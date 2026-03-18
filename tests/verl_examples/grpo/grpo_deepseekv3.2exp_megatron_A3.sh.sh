set -x

# # 0. download HF checkpoint
# # remove the `quantization_config` in the `config.json`
# # set `num_nextn_predict_layers=0` to disable MTP, which is not currently supported
# huggingface-cli download deepseek-ai/DeepSeek-V3-0324

# no offline dist checkpoint needed, now with mbridge>=0.13.0, we can directly init model from huggingface downloaded fp8 weights
# tested on docker://verlai/verl:app-verl0.5-transformers4.55.4-vllm0.10.0-mcore0.13.0-te2.2
LLM="DeepSeek-V3.2-Exp-bf16"
DIST_CKPT_PATH=""

export RAY_DEDUP_LOGS="0"
export VLLM_ASCEND_ENABLE_NZ=0

# 2. run the script

train_files=/dapo-math-17k.parquet
test_files=/dapo-math-17k.parquet

ALL_OFFLOAD=${ALL_OFFLOAD:-True}
COMMON_PARAM_OFFLOAD=${COMMON_PARAM_OFFLOAD:-$ALL_OFFLOAD}
COMMON_GRAD_OFFLOAD=${COMMON_GRAD_OFFLOAD:-$ALL_OFFLOAD}
COMMON_OPTIMIZER_OFFLOAD=${COMMON_OPTIMIZER_OFFLOAD:-$ALL_OFFLOAD}

ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-$COMMON_PARAM_OFFLOAD}
ACTOR_GRAD_OFFLOAD=${ACTOR_GRAD_OFFLOAD:-$COMMON_GRAD_OFFLOAD}
ACTOR_OPTIMIZER_OFFLOAD=${ACTOR_OPTIMIZER_OFFLOAD:-$COMMON_OPTIMIZER_OFFLOAD}
REF_PARAM_OFFLOAD=${REF_PARAM_OFFLOAD:-$COMMON_PARAM_OFFLOAD}
CRITIC_PARAM_OFFLOAD=${CRITIC_PARAM_OFFLOAD:-$COMMON_PARAM_OFFLOAD}
CRITIC_GRAD_OFFLOAD=${CRITIC_GRAD_OFFLOAD:-$COMMON_GRAD_OFFLOAD}
CRITIC_OPTIMIZER_OFFLOAD=${CRITIC_OPTIMIZER_OFFLOAD:-$COMMON_OPTIMIZER_OFFLOAD}
RM_PARAM_OFFLOAD=${RM_PARAM_OFFLOAD:-$COMMON_PARAM_OFFLOAD}



first_layer=3
last_layer=2
# PP=16，[3, 4×14, 2]

NNODES=16
PP=16
TP=8
EP=16
CP=1
ETP=1
INFER_TP=64
max_num_seqs=128

experiment_name='dsv3-32nodes'  
n_gpus_per_node=16


train_batch_size=128
ppo_mini_batch_size=64
n_resp_per_prompt=4

balance_batch=False

# NPU 910C
max_prompt_length=$((1024 * 16))
max_response_length=$(( 1024 * 8 ))
total_length=$(($max_prompt_length+$max_response_length))

use_dynamic_bsz=False
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 1))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 1))

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=True
kl_loss_coef=0.001

clip_ratio_low=0.2
clip_ratio_high=0.28

exp_name="685B-${NNODES}-train-pp${PP}-tp${TP}-ep${EP}-CP${CP}-actor-length${actor_ppo_max_token_len}_final"
CKPTS_DIR=${CKPTS_DIR:-"${exp_name}"}

python3 -m verl.trainer.main_ppo \
    --config-path=./config --config-name='ppo_megatron_trainer'\
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=$train_batch_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=False \
    data.truncation='left' \
    actor_rollout_ref.nccl_timeout=7200 \
    actor_rollout_ref.model.path=$LLM \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$INFER_TP \
    actor_rollout_ref.rollout.load_format='dummy' \
    actor_rollout_ref.rollout.max_num_seqs=$max_num_seqs \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    algorithm.use_kl_in_reward=False \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name='verl_megatron_gsm8k_examples' \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.nnodes=$NNODES \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.megatron.use_remove_padding=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.model.use_fused_kernels=False \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.actor.megatron.override_transformer_config.attention_backend='fused' \
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_distributed_optimizer=True \
    actor_rollout_ref.actor.strategy=megatron \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=$PP \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=$PP \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=$TP \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=$TP \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=$EP \
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=$EP \
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=$ETP \
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=$ETP \
    actor_rollout_ref.actor.megatron.context_parallel_size=$CP \
    +actor_rollout_ref.actor.megatron.override_transformer_config.sequence_parallel=True \
    actor_rollout_ref.actor.megatron.param_offload=True \
    actor_rollout_ref.actor.megatron.optimizer_offload=True \
    actor_rollout_ref.actor.megatron.grad_offload=True \
    actor_rollout_ref.ref.megatron.param_offload=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1 \
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=False \
    +actor_rollout_ref.actor.megatron.override_transformer_config.masked_softmax_fusion=False \
    +actor_rollout_ref.actor.megatron.override_transformer_config.bias_dropout_fusion=False \
    +actor_rollout_ref.actor.megatron.override_transformer_config.num_layers_in_first_pipeline_stage=$first_layer \
    +actor_rollout_ref.actor.megatron.override_transformer_config.num_layers_in_last_pipeline_stage=$last_layer \
    +actor_rollout_ref.actor.megatron.override_transformer_config.attention_softmax_in_fp32=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$total_length \
    actor_rollout_ref.actor.megatron.use_mbridge=True \
    actor_rollout_ref.actor.megatron.vanilla_mbridge=False \
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=False \
    actor_rollout_ref.ref.megatron.use_dist_checkpointing=False \
    actor_rollout_ref.actor.megatron.dist_checkpointing_path=$DIST_CKPT_PATH \
    actor_rollout_ref.ref.megatron.dist_checkpointing_path=$DIST_CKPT_PATH \
    trainer.default_local_dir=$CKPTS_DIR \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.normalization=RMSNorm \
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_rmsnorm=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.swiglu=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_swiglu=True \
    trainer.rollout_data_dir="/rollout_data_dir/$(date +%Y%m%d_%H%M%S)" \
    +actor_rollout_ref.actor.megatron.override_transformer_config.experimental_attention_variant="dsa" \
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_dsa_absorb=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.dsa_indexer_use_sparse_loss=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.dsa_indexer_loss_coeff=0.001 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_lightning_indexer=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_sparse_flash_attention=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_lightning_indexer_kl_loss=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_enable_expert_bias=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.context_parallel_size=${CP} \
    +actor_rollout_ref.actor.megatron.override_transformer_config.context_parallel_algo=kvallgather_cp_algo \
    +actor_rollout_ref.actor.megatron.override_transformer_config.reset_position_ids=False \
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_ascend_mc2=False \
    trainer.resume_mode="disable" \
    trainer.balance_batch=${balance_batch} \
    trainer.device=npu \
    trainer.val_before_train=False \
    trainer.total_epochs=100 2>&1 | tee logs/$(date +%Y%m%d_%H%M%S).log
