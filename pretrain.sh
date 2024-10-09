#! /bin/bash
set -x



export NNODES=$(( $DATA_PARALLEL_SIZE * $TENSOR_PARALLEL_SIZE * $PIPELINE_PARALLEL_SIZE / $GPUS_PER_NODE))
export MASTER_ADDR=$(scontrol show jobid=$SLURM_JOB_ID | tr '=' ' ' | grep BatchHost | awk '{print $2}')
export NODE_RANK=$(expr $SLURM_PROCID / $NNODES)
export WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
export RANK=$SLURM_PROCID
export CUDA_DEVICE_MAX_CONNECTIONS=1 # for async gradient all reduce

DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`

source ~/workspace/mega-env/bin/activate
cd /home/zanzong/workspace/Megatron-LM

VOCAB_FILE=~/datasets/wikidataset/gpt2-vocab.json
MERGE_FILE=~/datasets/wikidataset/gpt2-merges.txt
DATA_PATH=~/datasets/wikidataset/my-bert_text_sentence

# TRAIN_SAMPLES=$(( $GLOBAL_BATCH_SIZE * 50))
TRAIN_SAMPLES=$(( $GLOBAL_BATCH_SIZE * 15))

PROFILER_LOG_PATH=$PROFILER_LOG_PATH \
exec python \
        ./pretrain_gpt.py \
        --vocab-file $VOCAB_FILE \
	--merge-file $MERGE_FILE \
        --transformer-impl local \
	--tensor-model-parallel-size $TENSOR_PARALLEL_SIZE \
	--pipeline-model-parallel-size $PIPELINE_PARALLEL_SIZE \
	--num-layers $NUM_LAYERS \
	--hidden-size $HIDDEN_SIZE \
        --num-attention-heads $NUM_ATTN_HEADS \
        --seq-length 1024 \
        --max-position-embeddings 1024 \
        --micro-batch-size $MICRO_BATCH_SIZE \
        --global-batch-size $GLOBAL_BATCH_SIZE \
        --train-samples $TRAIN_SAMPLES \
	--lr-decay-samples 4882800 \
        --lr 0.0001 \
        --min-lr 0.00001 \
        --lr-decay-style cosine \
        --log-interval 1 \
        --eval-iters -1 \
        --data-path ${DATA_PATH} \
        --split 100,0,0 \
        --clip-grad 1.0 \
        --weight-decay 0.01 \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --init-method-std 0.002 \
        --fp16
