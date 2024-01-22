PORT=30001
GPU=$1
CFG=$2
TORCH_CUDNN_V8_API_ENABLED=1 \
torchrun --nproc_per_node=$GPU --master_port=$PORT main.py --cfg $CFG --data-path "/home/data/ImageNet" --amp --use-bf16 --tag $CFG