#!/bin/bash
set -x
ps -ef | grep python | grep -v grep | awk '{print $2}' | xargs kill -9

pip install -e 3-reinforcement-learning/verl
pip install liger_kernel==0.5.5
pip install math-verify
pip install antlr4-python3-runtime==4.9.3
pip install nvidia-cublas-cu12==12.4.5.8
pip uninstall -y megatron_core
pip install pyext==0.7
pip install fastparquet==2024.11.0

ulimit -c 0
export VLLM_USE_V1=0
which python

cd 3-reinforcement-learning

export MODEL_PATH="/path/to/OXA_MLE/checkpoint"
export SIGNATURE="RL_model_name"
export OUTPUT_DIR="/output/dir/${SIGNATURE}"
mkdir -p $OUTPUT_DIR

wandb offline
export WANDB_MODE=offline
export WANDB_DIR=$OUTPUT_DIR

cp -r 3-reinforcement-learning/train-scripts/7b/RL-for-OXA_MLE.sh $OUTPUT_DIR

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=3-reinforcement-learning/train-data/DeepScaleR-10K.prompt_v1.parquet \
    data.val_files=3-reinforcement-learning/train-data/aime24.prompt_v1.parquet \
    data.train_batch_size=64 \
    data.max_prompt_length=1536 \
    data.max_response_length=16384 \
    data.filter_overlong_prompts=True \
    actor_rollout_ref.model.path=$MODEL_PATH  \
    actor_rollout_ref.actor.optim.lr=2e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_liger=True \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size=64 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=0.85 \
    actor_rollout_ref.rollout.val_kwargs.n=8 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.kl_ctrl.kl_coef=0 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='SFT_for_RL' \
    trainer.experiment_name=${SIGNATURE} \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=${WORLD_SIZE} \
    trainer.save_freq=20 \
    trainer.test_freq=-1 \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=20 "${@:1}" \
    trainer.default_local_dir=${OUTPUT_DIR} \
    2>&1 | tee ${OUTPUT_DIR}/train-$(date "+%Y-%m-%d-%H:%M:%S").log