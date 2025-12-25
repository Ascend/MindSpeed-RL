set -x

export HCCL_CONNECT_TIMEOUT=3000
export HCCL_HOST_SOCKET_PORT_RANGE="HCCL HOST SOCKET PORT RANGE"
export HCCL_NPU_SOCKET_PORT_RANGE="HCCL NPU SOCKET PORT RANGE"

export VERL_LOGGING_LEVEL=DEBUG
export VLLM_LOGGING_LEVEL=DEBUG
export VLLM_CONFIGURE_LOGGING=1
export FLASHRL_LOGGING_LEVEL=DEBUG
export FLASHRL_CONFIG='./.flashrl_config.7b.yaml'
export FLASHRL_LMHEAD_FP32=0   # if set to 1, forcing vLLM conducting lm head compute in bf16

train_prompt_bsz=32
n_resp_per_prompt=4

train_prompt_mini_bsz=32
actor_micro_batch_size=4
ref_micro_batch_size=4
rollout_micro_batch_size=4

max_prompt_length=512
max_response_length=2048

RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
MODEL_PATH=${MODEL_PATH:-"${RAY_DATA_HOME}/models/qwen25-7b"}
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/data/gsm8k/train.parquet"}
TEST_FILE=${TEST_FILE:-"${RAY_DATA_HOME}/data/gsm8k/test.parquet"}

# Importance Sampling (IS) weights configuration
rollout_is="sequence"                     # Self-normalized sequence-level IS
rollout_is_threshold=2.0                  # Upper threshold for IS weights
rollout_is_batch_normalize="true"         # Self-normalization (mean=1.0)

# Rejection Sampling (RS) configuration
rollout_rs="null"                         # No rejection sampling for basic RLOO
rollout_rs_threshold="null"               # RS upper threshold
rollout_rs_threshold_lower="null"         # RS lower threshold

# Veto mechanism (optional, independent of IS/RS)
rollout_token_veto_threshold="null"       # Per-token veto threshold (null to disable)

# Policy Gradient loss mode (bypass mode with policy gradient loss, no PPO clipping)
bypass_mode="true"     # Required for policy gradient mode
use_policy_gradient="true"        # Use policy gradient loss (works with IS/RS/both)

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=${TRAIN_FILE} \
    data.val_files=${TEST_FILE} \
    data.train_batch_size=${train_prompt_bsz} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=5e-8 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${actor_micro_batch_size} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${rollout_micro_batch_size} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${ref_micro_batch_size} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    algorithm.rollout_correction.rollout_is=${rollout_is} \
    algorithm.rollout_correction.rollout_is_threshold=${rollout_is_threshold} \
    algorithm.rollout_correction.rollout_is_batch_normalize=${rollout_is_batch_normalize} \
    algorithm.rollout_correction.rollout_rs=${rollout_rs} \
    algorithm.rollout_correction.rollout_rs_threshold=${rollout_rs_threshold} \
    algorithm.rollout_correction.rollout_rs_threshold_lower=${rollout_rs_threshold_lower} \
    algorithm.rollout_correction.rollout_token_veto_threshold=${rollout_token_veto_threshold} \
    algorithm.rollout_correction.bypass_mode=${bypass_mode} \
    algorithm.rollout_correction.use_policy_gradient=${use_policy_gradient} \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=console \
    trainer.project_name='verl_grpo_example_gsm8k' \
    trainer.experiment_name='qwen2_5_7b_function_rm' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    trainer.device=npu $@
