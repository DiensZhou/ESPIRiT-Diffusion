#!/bin/bash
# Usage:
#   bash tune_sampling_params.sh espiritdiff example1.mat 0
#   bash tune_sampling_params.sh espiritdiff example2.mat 0 1

MODEL=${1:-espiritdiff}
EXAMPLE_FILE=${2:-example1.mat}
GPU_ID=${3:-0}
NUM_SLICES=${4:-1}
PYTHON_BIN=${PYTHON_BIN:-python}

export CUDA_VISIBLE_DEVICES=${GPU_ID}
export TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0;10.0;12.0"

CONFIG="configs/espiritdiff/ncsnpp_continuous.py"
if [ "${MODEL}" = "vp" ]; then
    CONFIG="configs/vp/ddpm_continuous.py"
elif [ "${MODEL}" = "spirit" ]; then
    CONFIG="configs/SPIRiT/ncsnpp_continuous.py"
fi

LOG_DIR="tuning_logs/${EXAMPLE_FILE%.mat}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"

"${PYTHON_BIN}" - <<PY > "${LOG_DIR}/slices.txt"
import random
random.seed(20260616)
for x in sorted(random.sample(range(0, 736), int("${NUM_SLICES}"))):
    print(x)
PY

echo "================ tuning ${MODEL} ${EXAMPLE_FILE} on GPU ${GPU_ID} ================"
echo "slices: $(tr '\n' ' ' < "${LOG_DIR}/slices.txt")"
echo "logs: ${LOG_DIR}"

for SNR in 0.5 0.7 0.9; do
  for MSE in 0.3 0.5 0.7; do
    for CMSE in 0.1 0.2 0.3; do
      for SLICE in $(cat "${LOG_DIR}/slices.txt"); do
        END=$((SLICE + 1))
        LOG="${LOG_DIR}/snr${SNR}_mse${MSE}_cmse${CMSE}_slice${SLICE}.log"
        echo "==== snr=${SNR} mse=${MSE} corrector_mse=${CMSE} slice=${SLICE} ===="
        PYTHONDONTWRITEBYTECODE=1 "${PYTHON_BIN}" main.py \
          --config="${CONFIG}" \
          --mode=sample \
          --workdir=results \
          --sample_file="${EXAMPLE_FILE}" \
          --sampling_snr="${SNR}" \
          --sampling_mse="${MSE}" \
          --sampling_corrector_mse="${CMSE}" \
          --sample_slice_start="${SLICE}" \
          --sample_slice_end="${END}" \
          > "${LOG}" 2>&1
        grep -E "PSNR:|SSIM|NMSE|saved:" "${LOG}" || true
      done
    done
  done
done

grep -H "PSNR:" "${LOG_DIR}"/*.log > "${LOG_DIR}/metrics_summary.txt" || true
echo "summary: ${LOG_DIR}/metrics_summary.txt"
