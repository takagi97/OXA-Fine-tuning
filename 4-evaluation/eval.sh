# set -e

cd ./4-evaluation
export VLLM_USE_V1=0

MODEL=$1
OUTPUT_DIR=$2
DATA=$3
REPEAT=$4
CONCURRENCY=$5
PORTS=(8010 8011 8012 8013 8014 8015 8016 8017)
DEVICES=(0 1 2 3 4 5 6 7)

if [[ -d "$OUTPUT_DIR" ]]; then
    echo "Output directory $OUTPUT_DIR already exists. Exiting..."
    exit 1
fi

python modify_generation_config.py $MODEL

for i in "${!DEVICES[@]}"; do
    PORT="${PORTS[$i]}"
    DEVICE="${DEVICES[$i]}"

    CUDA_VISIBLE_DEVICES=$DEVICE vllm serve $MODEL \
    --max_model_len 32768 \
    --enforce-eager \
    --gpu-memory-utilization 0.9 \
    --port $PORT &
done

sleep 5m

mkdir -p $OUTPUT_DIR
cp eval.sh eval.py $OUTPUT_DIR
python eval.py \
    --model $MODEL \
    --file 4-evaluation/benchmarks/$DATA.json \
    --ports 8010,8011,8012,8013,8014,8015,8016,8017 \
    --repeat $REPEAT \
    --concurrency $CONCURRENCY \
    --output_dir $OUTPUT_DIR


ps -ef | grep python | grep -v grep | awk '{print $2}' | xargs kill -9