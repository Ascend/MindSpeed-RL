# vllm-ascend has not yet been adapted for sink.
# To run this model, please：
# switch the vllm commit ID to 2918c1b49c88c29783c86f78d2c4221cb9622379
# switch the vllm—ascend commit ID to 941d54a2ce1ce387e4bf5d80003c098ff6d44841
# import verl_npu will patch this commit.
# Some files may fail to patch due to differences in system line endings.
# If this happens, it is recommended to add them manually.
# We will adapt as soon as vllm-ascend is supported.

set -x

project_name='GRPO'
exp_name='gpt-oss-20b'

export HCCL_CONNECT_TIMEOUT=3000
export HCCL_HOST_SOCKET_PORT_RANGE="HCCL HOST SOCKET PORT RANGE"
export HCCL_NPU_SOCKET_PORT_RANGE="HCCL NPU SOCKET PORT RANGE"

export VLLM_ASCEND_ENABLE_NZ=0
export VLLM_USE_V1=1

train_prompt_bsz=32
n_resp_per_prompt=4

train_prompt_mini_bsz=32
actor_micro_batch_size=4
ref_micro_batch_size=4
rollout_micro_batch_size=4

max_prompt_length=512
max_response_length=2048

RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
MODEL_PATH=${MODEL_PATH:-"${RAY_DATA_HOME}/models/gpt-oss-20b-BF16"}
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/data/gsm8k/train.parquet"}
TEST_FILE=${TEST_FILE:-"${RAY_DATA_HOME}/data/gsm8k/test.parquet"}

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=${TRAIN_FILE} \
    data.val_files=${TEST_FILE} \
    data.train_batch_size=${train_prompt_bsz} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${actor_micro_batch_size} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    ++actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${ref_micro_batch_size} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=8 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${rollout_micro_batch_size} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${exp_name} \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=10 \
    trainer.total_epochs=2 \
    trainer.device=npu $@
