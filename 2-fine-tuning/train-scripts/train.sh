export WANDB_DISABLED=1

bash 2-fine-tuning/train-scripts/set_env.sh
ps -ef | grep python | grep -v grep | awk '{print $2}' | xargs kill -9
cd 2-fine-tuning/LLaMA-Factory

FILE=2-fine-tuning/train-scripts/1.5b/OXA_MLE.yaml
MODEL_NAME=OXA_MLE-1.5b
OUTPUT_DIR=/output/dir/$MODEL_NAME
mkdir -p $OUTPUT_DIR
cp $FILE $OUTPUT_DIR

FORCE_TORCHRUN=1 NNODES=${WORLD_SIZE} NODE_RANK=${RANK} nohup llamafactory-cli train $FILE > $OUTPUT_DIR/$MODEL_NAME.${RANK}.log 2>&1 &