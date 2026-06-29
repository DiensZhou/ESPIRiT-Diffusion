#!/bin/bash
set -euo pipefail

# Run examples in parallel across GPUs, with one example at a time per GPU.
#
# Usage:
#   bash batch_test_fastMRI.sh espiritdiff example1.mat example2.mat

MODEL=${1:-}
if [ -z "${MODEL}" ]; then
    echo "Usage: bash batch_test_fastMRI.sh <espiritdiff|vp|spirit> <example1.mat> [example2.mat ...]"
    exit 1
fi
shift

if [ "$#" -eq 0 ]; then
    echo "Usage: bash batch_test_fastMRI.sh <espiritdiff|vp|spirit> <example1.mat> [example2.mat ...]"
    exit 1
fi

IFS=',' read -r -a GPU_IDS <<< "${GPUS:-0,1,2}"
EXAMPLES=("$@")

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
LOG_DIR="${SCRIPT_DIR}/logs/batch_test_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"

echo "Model: ${MODEL}"
echo "GPUs: ${GPU_IDS[*]}"
echo "Examples: ${EXAMPLES[*]}"
echo "Logs: ${LOG_DIR}"
echo "Status: ${LOG_DIR}/status.tsv"
printf "time\tgpu\texample\tpid\tlog\n" > "${LOG_DIR}/status.tsv"

run_gpu_queue() {
    local gpu_id=$1
    local gpu_index=$2
    local gpu_count=$3
    local example_file
    local log_file
    local pid
    local i

    for i in "${!EXAMPLES[@]}"; do
        if [ $((i % gpu_count)) -ne "${gpu_index}" ]; then
            continue
        fi

        example_file="${EXAMPLES[$i]}"
        log_file="${LOG_DIR}/gpu${gpu_id}_${example_file%.mat}.log"
        echo "[$(date '+%F %T')] GPU ${gpu_id} start example ${example_file}"
        bash "${SCRIPT_DIR}/test_fastMRI.sh" "${MODEL}" "${example_file}" "${gpu_id}" \
            > "${log_file}" 2>&1 &
        pid=$!
        printf "%s\t%s\t%s\t%s\t%s\n" "$(date '+%F %T')" "${gpu_id}" "${example_file}" "${pid}" "${log_file}" \
            >> "${LOG_DIR}/status.tsv"
        echo "[$(date '+%F %T')] GPU ${gpu_id} example ${example_file} pid ${pid} log ${log_file}"
        wait "${pid}"
        echo "[$(date '+%F %T')] GPU ${gpu_id} done example ${example_file}"
    done
}

GPU_COUNT=${#GPU_IDS[@]}
for gpu_index in "${!GPU_IDS[@]}"; do
    run_gpu_queue "${GPU_IDS[$gpu_index]}" "${gpu_index}" "${GPU_COUNT}" &
done

wait
echo "All examples finished."
