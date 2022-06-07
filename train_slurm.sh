set -x
CFG=$1
GPUS=$2
JOB_NAME=$3

GPUS_PER_NODE=${GPUS_PER_NODE:-8}
CPUS_PER_TASK=${CPUS_PER_TASK:-8}
srun -p RTX3090 \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    python -u slurm_main.py --cfg="configs/${CFG}.yaml" --data-path="/home/data/ImageNet" --amp --tag="${CFG}"
