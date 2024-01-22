set -x
CFG=$1
GPUS=$2
JOB_NAME=$3
PART_NAME=${4:-RTX3090}

GPUS_PER_NODE=${GPUS_PER_NODE:-8}
CPUS_PER_TASK=${CPUS_PER_TASK:-8}

export TORCH_CUDNN_V8_API_ENABLED=1

srun -p ${PART_NAME} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    python -u slurm_main.py --cfg="configs/${CFG}.yaml" --data-path="/home/data/ImageNet" --amp --use-bf16 --tag="${CFG}"
