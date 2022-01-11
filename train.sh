PORT=30001
GPU=$1
CFG=$2
TAG=${3:-'default'}

python -m torch.distributed.launch --nproc_per_node $GPU --master_port $PORT --use_env main.py --cfg $CFG --data-path <path-to-imagenet> --amp --tag $TAG