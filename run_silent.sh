#!/bin/bash

set -x
if [ "$#" -ne 8 ]
then
    echo "usage:" $0 "exp_name model_name t p d gbs mbs nodelist"
    exit 1
fi

export MASTER_PORT=$(expr $RANDOM % 10000 + 10000)

export EXP_NAME=$1
export MODEL_NAME=$2
export TENSOR_PARALLEL_SIZE=$3
export PIPELINE_PARALLEL_SIZE=$4
export DATA_PARALLEL_SIZE=$5
export GLOBAL_BATCH_SIZE=$6
export MICRO_BATCH_SIZE=$7
export NODELIST=$8
export GPUS_PER_NODE=8

export NUM_LAYERS=-1
export HIDDEN_SIZE=-1
export NUM_ATTN_HEADS=-1

if [ ${MODEL_NAME} == "GPT-760M" ];then
    export NUM_LAYERS=24
    export HIDDEN_SIZE=1536
    export NUM_ATTN_HEADS=16
fi

if [ ${MODEL_NAME} == "GPT-1.3B" ];then
    export NUM_LAYERS=24
    export HIDDEN_SIZE=2048
    export NUM_ATTN_HEADS=16
fi

if [ ${NUM_LAYERS} == -1 ];then
    echo "model name not found."
    exit -1
fi

mkdir -p ./logs
mkdir -p ./logs/${EXP_NAME}

LOG_DIR=$(pwd)/logs/${EXP_NAME}

LOG_PREFIX=${MODEL_NAME}\_t$TENSOR_PARALLEL_SIZE\_p$PIPELINE_PARALLEL_SIZE\_d$DATA_PARALLEL_SIZE\_gbs$GLOBAL_BATCH_SIZE\_mbs$MICRO_BATCH_SIZE\_$(date -Iseconds)
LOG_NAME=${LOG_PREFIX}.log

export PROFILER_LOG_PATH=${LOG_DIR}/${LOG_PREFIX}.prof

mkdir -p $PROFILER_LOG_PATH

NNODES=$(scontrol show hostnames ${NODELIST} | wc -l)

nohup srun \
    --exclusive=user \
    -p Big \
    -N $NNODES \
    -K \
    -w $NODELIST \
    --time 20:00 \
    --job-name=mmpret \
	--ntasks-per-node=$GPUS_PER_NODE \
    --gres=gpu:$GPUS_PER_NODE \
    --export=ALL \
	bash pretrain.sh  > ${LOG_DIR}/${LOG_NAME} 2>&1
