PORT=30001
CFG=$1
BS=$2

TORCH_LOGTORCH_LOGS="+dynamo" TORCHDYNAMO_VERBOSE=1 \
torchrun --nproc_per_node 1 --master_port $PORT main.py --cfg $CFG --data-path "/root/autodl-tmp/data/imagenet" --batch-size $BS --throughput