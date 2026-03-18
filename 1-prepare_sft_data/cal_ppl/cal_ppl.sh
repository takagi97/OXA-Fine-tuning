#!/bin/bash


cd 1-prepare_sft_data/1-cal_ppl

input_dir=/path/to/part_XXXX.jsonl
output_dir=/output/path

model_path=/path/to/base-LLMs
max_seq_len=32768
batch_size=8

# 1 台机器 共1*8=8个serve 一个serve1个并发 总共8*1=8个进程 还剩 268835 条数据，一个进程 33605 行
START_ID=$(($1 * 8))
for gpu_id in {0..7}; do
    part_idx=$(($START_ID + $gpu_id))
    echo "Running part_idx=$part_idx on gpu_id=$gpu_id"
    python3 cal_ppl.py $part_idx $input_dir $output_dir $model_path $max_seq_len $gpu_id $batch_size &
done
