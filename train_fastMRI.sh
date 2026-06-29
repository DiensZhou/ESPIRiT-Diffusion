#!/bin/bash
# sh train_fastMRI.sh "vp" 

export CUDA_VISIBLE_DEVICES=2
# export TF_CPP_MIN_LOG_LEVEL=2echo $SHELL
# export PATH=/usr/local/cuda-12.2/bin:$PATH

if [ "$1" = "vp" ]
then
    echo "================ run configs/vp/ddpm_continuous.py ================"
    python main.py \
        --config=configs/vp/ddpm_continuous.py \
        --mode='train'  \
        --workdir=results
elif [ "$1" = "espiritdiff" ]
then
    echo "================ run configs/espiritdiff/ncsnpp_continuous.py ================"
    python main.py \
        --config=configs/espiritdiff/ncsnpp_continuous.py \
        --mode='train'  \
        --workdir=results
elif [ "$1" = "spirit" ]
then
    echo "================ run configs/SPIRiT/ncsnpp_continuous.py ================"
    python main.py \
        --config=configs/SPIRiT/ncsnpp_continuous.py \
        --mode='train'  \
        --workdir=results
else
    echo "================ You must input one argument: espiritdiff or vp ================"
fi
