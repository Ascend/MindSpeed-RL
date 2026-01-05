#!/usr/bin/env bash
set -x

project_name='GRPO'
exp_name='GRPO-Qwen2.5-32B-tp4_sp_4_bsz_32_response_8k_sglang-deepscaler'

# Performance Related Parameter
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=True
kl_loss_coef=0.001

max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 8))
enable_overlong_buffer=True
overlong_buffer_len=$((1024 * 4))
overlong_penalty_factor=1.0

loss_agg_mode="token-mean"

enable_filter_groups=False
filter_groups_metric=acc
max_num_gen_batches=8
train_prompt_bsz=32
n_resp_per_prompt=8
train_prompt_mini_bsz=32
gen_tp=4
sp_size=4
adv_estimator=grpo

# Algorithm
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for SGLang rollout

# Ray
NNODES=${NNODES:-2}
# Paths
RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/sglang"}
MODEL_PATH=${MODEL_PATH:-"${RAY_DATA_HOME}/models/Qwen2.5-32B"}
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/datasets/deepscaler/deepscaler.parquet"}
TEST_FILE=${TEST_FILE:-"${RAY_DATA_HOME}/datasets/deepscaler/deepscaler.parquet"}


PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.train_batch_size=${train_prompt_bsz} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.prompt_key=prompt \
    data.filter_overlong_prompts=${enable_overlong_buffer} \
    data.truncation='error' \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend="ascend" \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.multi_stage_wake_up=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.use_torch_compile=False \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.nccl_timeout=1800 \
    algorithm.use_kl_in_reward=False \
    algorithm.adv_estimator="${adv_estimator}" \
    trainer.critic_warmup=0 \
    trainer.logger=console \
    trainer.val_before_train=False \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes="${NNODES}" \
    trainer.save_freq=100 \
    trainer.test_freq=-1 \
    trainer.device=npu \
    trainer.total_epochs=5 2>&1 | tee "./logs/verl-sglang_tp4_sp_4_bsz_32_response_8k_deepscaler_A2_$(date +%Y%m%d_%H%M).log"
