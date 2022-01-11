PORT=30001
GPU=$2
NODE=$1
JOB=TEST_DAT
CFG=$3
TAG=${4:-'default'}

srun -p $NODE -N 1 -J $JOB --gres gpu:$GPU \
    python -m torch.distributed.launch --nproc_per_node $GPU --master_port $PORT --use_env main.py --cfg $CFG --data-path /home/data/imagenet --amp --tag $TAG