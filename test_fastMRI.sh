#!/bin/sh
# sh test_fastMRI.sh "vp"
# sh test_fastMRI.sh "espiritdiff" "example1.mat" "0"
# sh test_fastMRI.sh "espiritdiff" "example2.mat" "0"

MODEL=$1
EXAMPLE_FILE=${2:-}
GPU_ID=${3:-0}
if [ "$#" -ge 3 ]; then
    shift 3
else
    shift "$#"
fi
PYTHON_BIN=${PYTHON_BIN:-python}
CONFIG_VP=${CONFIG_VP:-configs/vp/ddpm_continuous.py}
CONFIG_ESPIRITDIFF=${CONFIG_ESPIRITDIFF:-configs/espiritdiff/ncsnpp_continuous.py}
CONFIG_SPIRIT=${CONFIG_SPIRIT:-configs/SPIRiT/ncsnpp_continuous.py}
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "${SCRIPT_DIR}" || exit 1
export CUDA_VISIBLE_DEVICES=${GPU_ID}
echo "================ CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} ================"
"${PYTHON_BIN}" -c "import torch; print('torch cuda available:', torch.cuda.is_available()); print('torch visible device count:', torch.cuda.device_count()); print('torch cuda:0 name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
GPU_ARCH=$("${PYTHON_BIN}" -c "import torch; print('%d.%d' % torch.cuda.get_device_capability(0) if torch.cuda.is_available() else '')")
if [ -n "${GPU_ARCH}" ]; then
    export TORCH_CUDA_ARCH_LIST="${GPU_ARCH}"
    TORCH_EXT_ARCH=$(printf "%s" "${GPU_ARCH}" | tr "." "_")
    export TORCH_EXTENSIONS_DIR="/tmp/torch_extensions_${USER}_${TORCH_EXT_ARCH}"
    mkdir -p "${TORCH_EXTENSIONS_DIR}"
    echo "================ TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} ================"
    echo "================ TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR} ================"
fi

if [ "${MODEL}" = "vp" ]
then
    echo "================ run configs/vp/ddpm_continuous.py ================"
    "${PYTHON_BIN}" main.py \
        --config="${CONFIG_VP}" \
        --sample_file="${EXAMPLE_FILE}" \
        --mode='sample'  \
        --workdir=results \
        "$@"
elif [ "${MODEL}" = "espiritdiff" ]
then
    echo "================ run ${CONFIG_ESPIRITDIFF} example=${EXAMPLE_FILE:-all} ================"
    "${PYTHON_BIN}" main.py \
        --config="${CONFIG_ESPIRITDIFF}" \
        --sample_file="${EXAMPLE_FILE}" \
        --mode='sample'  \
        --workdir=results \
        "$@"
elif [ "${MODEL}" = "spirit" ]
then
    echo "================ run configs/spirit/ncsnpp_continuous.py ================"
    "${PYTHON_BIN}" main.py \
        --config="${CONFIG_SPIRIT}" \
        --sample_file="${EXAMPLE_FILE}" \
        --mode='sample'  \
        --workdir=results \
        "$@"
else
    echo "================ Usage: sh test_fastMRI.sh espiritdiff example1.mat 0 ================"
    echo "================ Arguments: model(espiritdiff/vp/spirit) example_file gpu_id [extra main.py flags] ================"
fi
