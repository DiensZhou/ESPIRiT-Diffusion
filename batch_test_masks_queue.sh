#!/bin/bash
set -euo pipefail

# Dynamic GPU queue for running multiple examples with multiple masks.
#
# Usage:
#   bash batch_test_masks_queue.sh
#   GPUS=0,1,2 bash batch_test_masks_queue.sh

MODEL=${MODEL:-espiritdiff}
PYTHON_BIN=${PYTHON_BIN:-python}

EXAMPLES=(example1.mat example2.mat)
MASKS=(
    mask_cartesian/cartesian_302x256acc_8.8acs_48.mat
    mask_random/random_302x256acc_8.8acs_48.mat
)

IFS=',' read -r -a GPU_IDS <<< "${GPUS:-0,1,2}"

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
LOG_DIR="${SCRIPT_DIR}/logs/batch_masks_$(date +%Y%m%d_%H%M%S)"
CONFIG_DIR="${LOG_DIR}/configs"
mkdir -p "${CONFIG_DIR}"

cleanup() {
    echo
    echo "Caught interrupt, stopping queued reconstruction jobs..."
    trap - INT TERM
    kill -- -$$ 2>/dev/null || true
    wait 2>/dev/null || true
    exit 130
}
trap cleanup INT TERM

JOB_FILE="${LOG_DIR}/jobs.tsv"
NEXT_FILE="${LOG_DIR}/next_job.txt"
LOCK_FILE="${LOG_DIR}/queue.lock"
STATUS_FILE="${LOG_DIR}/status.tsv"

echo 2 > "${NEXT_FILE}"
printf "job_id\texample\tmask\tconfig\n" > "${JOB_FILE}"
printf "time\tevent\tgpu\tjob_id\texample\tmask\tpid\texit_code\tlog\n" > "${STATUS_FILE}"

make_mask_config() {
    local mask_path=$1
    local tag=$2
    local config_path="${CONFIG_DIR}/${tag}.py"

    cat > "${config_path}" <<EOF
from configs.espiritdiff.ncsnpp_continuous import get_config as _base_get_config


def get_config():
    config = _base_get_config()
    config.sampling.mask_file = "${mask_path}"
    return config
EOF
    printf "%s" "${config_path}"
}

job_id=0
for mask in "${MASKS[@]}"; do
    mask_base=$(basename "${mask}" .mat)
    mask_tag=$(printf "%s" "${mask_base}" | tr '.-' '__')
    config_path=$(make_mask_config "${mask}" "${mask_tag}")
    for example_file in "${EXAMPLES[@]}"; do
        job_id=$((job_id + 1))
        printf "%s\t%s\t%s\t%s\n" "${job_id}" "${example_file}" "${mask}" "${config_path}" >> "${JOB_FILE}"
    done
done

record_status() {
    local event=$1
    local gpu=$2
    local jid=$3
    local example_file=$4
    local mask=$5
    local pid=$6
    local exit_code=$7
    local log_file=$8

    flock "${LOCK_FILE}" printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$(date '+%F %T')" "${event}" "${gpu}" "${jid}" "${example_file}" "${mask}" \
        "${pid}" "${exit_code}" "${log_file}" >> "${STATUS_FILE}"
}

get_next_job() {
    local line_no
    local line

    flock "${LOCK_FILE}" bash -c '
        line_no=$(cat "$1")
        line=$(sed -n "${line_no}p" "$2")
        if [ -n "${line}" ]; then
            echo $((line_no + 1)) > "$1"
            printf "%s" "${line}"
        fi
    ' _ "${NEXT_FILE}" "${JOB_FILE}"
}

run_worker() {
    local gpu_id=$1
    local line
    local jid
    local example_file
    local mask
    local config_path
    local mask_name
    local log_file
    local pid
    local exit_code

    while true; do
        line=$(get_next_job)
        if [ -z "${line}" ]; then
            break
        fi

        IFS=$'\t' read -r jid example_file mask config_path <<< "${line}"
        mask_name=$(basename "${mask}" .mat)
        log_file="${LOG_DIR}/gpu${gpu_id}_job${jid}_${example_file%.mat}_${mask_name}.log"

        echo "[$(date '+%F %T')] GPU ${gpu_id} start job ${jid}: ${example_file}, ${mask}"
        CONFIG_ESPIRITDIFF="${config_path}" PYTHON_BIN="${PYTHON_BIN}" \
            bash "${SCRIPT_DIR}/test_fastMRI.sh" "${MODEL}" "${example_file}" "${gpu_id}" \
            > "${log_file}" 2>&1 &
        pid=$!
        record_status "start" "${gpu_id}" "${jid}" "${example_file}" "${mask}" "${pid}" "-" "${log_file}"

        set +e
        wait "${pid}"
        exit_code=$?
        set -e

        record_status "done" "${gpu_id}" "${jid}" "${example_file}" "${mask}" "${pid}" "${exit_code}" "${log_file}"
        echo "[$(date '+%F %T')] GPU ${gpu_id} done job ${jid}: ${example_file}, exit=${exit_code}"
    done
}

echo "Model: ${MODEL}"
echo "GPUs: ${GPU_IDS[*]}"
echo "Examples: ${EXAMPLES[*]}"
echo "Masks: ${MASKS[*]}"
echo "Logs: ${LOG_DIR}"
echo "Status: ${STATUS_FILE}"

for gpu_id in "${GPU_IDS[@]}"; do
    run_worker "${gpu_id}" &
done

wait
echo "All mask-example jobs finished."
