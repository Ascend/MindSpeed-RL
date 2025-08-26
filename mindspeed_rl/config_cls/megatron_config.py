# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.

import os
from pathlib import Path

from typing import Dict

from mindspeed_rl.config_cls.base_config import BaseConfig


cur_file_dir = Path(__file__).absolute().parent
TEMPLATES_DIR = os.path.join(cur_file_dir.parent.parent, "configs/model/templates.json")


class MegatronConfig(BaseConfig):
    '''
    Model configuration class.
    Initialize model configuration from the provided config dictionary.
    All instance attributes are initialized using the dictionary keys.

    models_parameters:
    use_mcore_models: Whether to use MCore models (default: False)
    spec: Specify the <module_location function_name> pair (default: None)
    sequence_parallel: Whether to use sequence parallelism (default: False)
    use_flash_attn: Whether to use flash attention (default: False)
    use_rotary_position_embeddings: Whether to use rotary position embeddings (default: False)
    use_fused_rmsnorm: Whether to use fused RMSNorm (default: False)
    use_fused_swiglu: Whether to use fused Swiglu (default: False)
    shape_order: Input shape order used by Flash attention (default: 'SBH')
    no_bias_dropout_fusion: Disable bias and dropout fusion (default: False)
    rope_scaling_type: Type of rope scaling used (default: None)
    rope_scaling_factor: Scaling factor for rope (default: 1.0)
    low_freq_factor: Low frequency factor (default: None)
    high_freq_factor: High frequency factor (default: None)
    original_max_position_embeddings: Original maximum position embeddings (default: None)
    max_position_embeddings: Maximum position embeddings (default: None)
    beta_fast: Yarn rope: rope beta fast (default: 32)
    beta_slow: Yarn rope: rope beta slow (default: 1)
    rope_scaling_mscale: Yarn rope: rope mscale (default: 1.0)
    rope_scaling_mscale_all_dim: Yarn rope: rope mscale all dim (default: 0.0)
    rope_scaling_original_max_position_embeddings: Yarn rope: rope original max position embeddings (default: None)
    num_layers: Number of layers in the model (default: None)
    hidden_size: Size of the hidden layers (default: None)
    ffn_hidden_size: Size of the feed-forward layers (default: None)
    num_attention_heads: Number of attention heads (default: None)
    kv_channels: Projection weights dimension in multi-head attention. This is set to args.hidden_size //
         args.num_attention_heads if not provided (default: None)
    group_query_attention: Whether to use group query attention (default: False)
    num_query_groups: Number of query groups (default: 1)
    make_vocab_size_divisible_by: Divisibility constraint for vocab size (default: 128)
    padded_vocab_size: Padded vocabulary size (default: None)
    add_qkv_bias: Enable bias only in the QKV linear layers (default: False)
    disable_bias_linear: Whether to disable bias in the linear layer (default: False)
    attention_dropout: Dropout rate for attention (default: 0.1)
    init_method_std: Standard deviation for initialization (default: 0.02)
    hidden_dropout: Dropout rate for hidden layers (default: 0.1)
    position_embedding_type: Type of position embedding used (default: 'learned_absolute')
    rotary_base: Base value for rotary embedding (default: 10000)
    normalization: Normalization method (default: 'LayerNorm')
    norm_epsilon: Epsilon value for normalization (default: 1e-5)
    swiglu: Whether to use SwiGLU activation (default: False)
    no_masked_softmax_fusion: Whether to disable masked softmax fusion (default: False)
    attention_softmax_in_fp32: Whether to use FP32 for attention softmax (default: False)
    no_gradient_accumulation_fusion: Whether to disable gradient accumulation fusion (default: False)
    bf16: Whether to use BF16 precision (default: False)
    untie_embeddings_and_output_weights: Untie embeddings and output weights (default: False)

    multi_latent_attention: Use Multi-head Latent Attention(MLA) (default: False)
    qk_pos_emb_head_dim: The qk head dim for rope (default: None)
    qk_head_dim: The qk head dim for only self-attn (default: None)
    q_lora_rank: The low rank of q (default: None)
    kv_lora_rank: The low rank of k and v (default: None)
    v_head_dim: The head dim of v (default: None)
    qk_layernorm: Whether to layer normalize the q and k attention embeddings (default: False)

    moe_grouped_gemm: Use moe grouped gemm (default: False)
    moe_permutation_async_comm: overlap moe permutation all gather communications (default: False)
    use_fused_moe_token_permute_and_unpermute: Use fused moe permute and unpermute (default: False)
    moe_token_dispatcher_type: The dispatcher type for moe token dispatching (default: None)
    seq_aux: Compute aux loss in seq_aux (default: False)
    first_k_dense_replace: First k layer as dense layer (default: None)
    moe_layer_freq: The occurrence frequency of the moe layer (default: None)
    moe_router_topk: Number of experts to route to for each token (default: 2)
    num_experts: Number of Experts in MoE (None means no MoE) (default: None)
    n_shared_experts: This value is the number of shared experts (default: None)
    moe_ffn_hidden_size: The ffn hidden size of MoE layer (default: None)
    moe_router_load_balancing_type: Determines the load balancing strategy for the router (default: aux_loss)
    moe_router_num_groups: Number of groups for routed experts (default: None)
    moe_router_group_topk: Choose topK group experts in group_limited_greedy_topK method (default: None)
    moe_router_topk_scaling_factor: The routed scaling factor (default: None)
    norm_topk_prob: Normalize the topk weight (default: False)
    moe_router_score_function: Score function for MoE TopK routing. Can be 'softmax' or 'sigmoid' (default: softmax)
    moe_router_enable_expert_bias: TopK routing with dynamic expert bias in the aux-loss-free load balancing strategy (default: False)

    training_parameters:
    global_batch_size: Global batch size for training (default: None)
    seq_length: Sequence length for training (default: None)
    tokenizer_type: Type of tokenizer used (default: None)
    tokenizer_name_or_path: Path or name of the tokenizer (default: None)
    train_iters: Number of training iterations (default: None)
    eval_iters: Number of iterations to run for evaluation validation/test for (default: 100)
    distributed_backend: Distributed backend for training (default: 'nccl')
    no_shared_storage: Whether to use shared storage (default: False)
    save_interval: Interval for saving models (default: None)
    no_load_optim: Whether to skip loading optimizer (default: None)
    no_load_rng: Whether to skip loading RNG state (default: None)
    no_save_optim: Whether to skip saving optimizer (default: None)
    no_save_rng: Whether to skip saving RNG state (default: None)
    bf16: Whether to use BF16 (default: False)
    use_distributed_optimizer: Use distributed optimizer (default: False)
    is_instruction_dataset: Whether the dataset is instruction-based (default: False)
    is_pairwise_dataset: Whether the dataset is pairwise format that has a chosen sequence and rejected
        sequence, which usually used in reinforce learning (default: False)
    variable_seq_lengths: Whether to use variable sequence lengths (default: False)
    no_shuffle: Whether to shuffle the dataset (default: False)
    stage: Stage of the model (default: None)
    sequence_parallel: Whether to use sequence parallelism (default: False)
    micro_batch_size: Micro batch size for actor (default: 1)
    tensor_model_parallel_size: Size of tensor model parallelism (default: 1)
    pipeline_model_parallel_size: Size of pipeline model parallelism (default: 1)
    expert_model_parallel_size: Degree of expert model parallelism (default: 1)
    num_layers_per_virtual_pipeline_stage: Degree of vpp (default: None)
    lr: Learning rate (default: None)
    lr_decay_style: Learning rate decay style (default: 'linear')
    min_lr: Minimum learning rate (default: 0.0)
    weight_decay: Weight decay for regularization (default: 0.01)
    lr_warmup_fraction: Fraction of steps for learning rate warmup (default: None)
    clip_grad: Gradient clipping value (default: 1.0)
    adam_beta1: Adam optimizer beta1 value (default: 0.9)
    adam_beta2: Adam optimizer beta2 value (default: 0.999)
    initial_loss_scale: Initial loss scale (default: 2**32)
    finetune: Whether to fine-tune the model (default: False)
    load: Path to load the model from (default: None)
    save: Path to save the model (default: None)
    pad_to_multiple_of: Padding to multiple of (default: 8)
    data_path: Path to the dataset (default: None)
    split: Data split for training, validation, and test (default: None)
    dataloader_type: Single pass vs multiple pass data loader (default: None)
    enable_high_availability: Switch of the high availability feature (default: False)
    context_parallel_size: Degree of context parallelism (default: 1)
    context_parallel_algo: Algorithm of context parallelism (default: ulysses_cp_algo)
    reset_position_ids: Reset posistion ids after end-of-document token (default: False)
    optimizer: Optimizer function (default: 'adam')
    do_sample: Enable doing sample in actor generations (default: False)
    prompt_type: Which template to use for constructing prompts in training/inference  'e.g., "qwen (default None)"
    prompt_type_path:Path to the json file of templates (default: TEMPLATES_DIR).
    tokenizer_not_use_fast: HuggingFace tokenizer not use the fast version (default: False)
    use_fused_rotary_pos_emb: Use new fused rotary-pos-emb (default False)
    full_shuffle_instruction_dataset: Full shuffle instruction dataset or not (default: False)
    tokenizer_padding_side: Tokenizer padding side (default: 'right')
    num_workers: Dataloader number of workers (default: 2)
    skip_train: If set, bypass the training loop, optionally do evaluation for validation/test, and exit (default: False)
    eval_interval: Interval between running evaluation on validation set (default: 1000)
    seed: Random seed used for python, numpy, pytorch, and cuda (default: 1234)
    vocab_extra_ids: Number of additional vocabulary tokens. They are used for span masking in the T5 model (default: 0)
    use_tp_pp_dp_mapping: If set, distributed ranks initialize order is changed from tp-dp-pp to tp-pp-dp.
        Make sure EP and CP aren't used with this option enabled with this option enabled (default: False)
    log_interval: Report loss and timing interval (default: 100)
    load_checkpoint_loosely: Enable loading checkpoint not strictly (default: False)
    ddp_bucket_size: Bucket size for data-parallel communication (default: None)
    no_check_for_nan_in_loss_and_grad: no check for NaNs in loss and grad (default: False)
    overlap_grad_reduce: If set, overlap DDP grad reduce (default: False)
    accumulate_allreduce_grads_in_fp32: Gradient accumulation and all-reduce in fp32 (default: False)
    pretrained_checkpoint: Directory containing a pretrained model checkp oint for finetuning (default: None)
    reuse_fp32_param: The distributed training optimizer frees up param copies of FP32 to save memory (default: False)
    recompute_granularity: Checkpoint activations to allow for training (default: None)
    recompute_method: Checkpoint activations methods, support 'uniform' and 'block' (default: None)
    recompute_num_layers: 1) uniform: the number of Transformer layers in each uniformly divided recompute unit,
                          2) block: the number of individual Transformer layers to recompute within each pipeline stage
                          (default: None)
    ai_framework: choose mindspore or not (default: None) choices=[None, 'mindspore']
    num_layer_list: a list of number of layers, seperated by comma; e.g., 4,4,4,4 (default: None)
    dataset_additional_keys: Additional keys need to be add from dataset (default: [])
    dpo_loss_type: DPO loss computation method type (default: sigmoid)

    inference_parameters:
    use_kv_cache: Use kv cache to accelerate inference

    log_parameters:
    log_throughput:'If set, calculate and log throughput per GPU.'(default: False)

    memory_parameters:
    recompute_activation_function: Recompute the activation function in MLP layers (default: False)
    recompute_granularity: 'Checkpoint activations to allow for training '
                           'with larger models, sequences, and batch sizes. '
                           'It is supported at two granularities 1) full: '
                           'whole transformer layer is recomputed, '
                           '2) selective: core attention part of the transformer '
                           'layer is recomputed.' (default: None) choices=['full', 'selective'],
    recompute_method:  '1) uniform: uniformly divide the total number of '
                       'Transformer layers and recompute the input activation of '
                       'each divided chunk at specified granularity, '
                       '2) recompute the input activations of only a set number of '
                       'individual Transformer layers per pipeline stage and do the '
                       'rest without any recomputing at specified granularity'  (default: None)  choices=['uniform', 'block'],
    recompute_num_layers: '1) uniform: the number of Transformer layers in each '
                       'uniformly divided recompute unit, '
                       '2) block: the number of individual Transformer layers '
                       'to recompute within each pipeline stage.'  (default: None)
    swap_attention: switch to open swap-attention feature (default: False)
    reuse_fp32_param: The distributed training optimizer frees up 'param copies of FP32 to save memory. (default: False)
    moe_tp_extend_ep: use tp group to extend experts parallelism instead of sharding weight tensor of experts in tp group
    moe_alltoall_overlap_comm: moe_alltoall_overlap_comm
    noop_layers:  noop layers string ，PP切分时，层不够加的冗余层
    attention_mask_type: attention mask type in cp
    reset_attention_mask: reset attention_mask in cp
    use_cp_send_recv_overlap: To support send receive overlap in cp, suggest to be true
    use_fused_ring_attention_update:  use fused_ring_attention_update in cp, suggest to be true
    use_ascend_coc: switch to open CoC feature (default: False)
    coc_mode: 0=original, 1=rewrite, 2=coc default
    coc_parallel_num: number of parallel in CoC features (default: 1)
    coc_fused_kernel: switch to use fused kernel in CoC (default: False)
    swap_optimizer: switch to swap-optimizer feature (default: False)
    mm_model: config for multimodal models
    '''

    def __init__(self, training_config: Dict, model_config: Dict):
        '''
        param config_dict: Dictionary containing the configuration parameters
        '''
        # Default values can still be defined if no config is provided
        self.use_mcore_models = False
        self.spec = None
        self.sequence_parallel = False
        self.use_flash_attn = False
        self.use_rotary_position_embeddings = False
        self.use_fused_rmsnorm = False
        self.use_fused_swiglu = False
        self.shape_order = 'SBH'
        self.no_bias_dropout_fusion = False

        self.rope_scaling_type = None
        self.rope_scaling_factor = 1.0
        self.low_freq_factor = None
        self.high_freq_factor = None
        self.original_max_position_embeddings = None
        self.max_position_embeddings = None
        self.beta_fast = 32
        self.beta_slow = 1
        self.rope_scaling_mscale = 1.0
        self.rope_scaling_mscale_all_dim = 0.0
        self.rope_scaling_original_max_position_embeddings = None

        self.num_layers = None
        self.hidden_size = None
        self.ffn_hidden_size = None
        self.num_attention_heads = None
        self.kv_channels = None
        self.group_query_attention = False
        self.num_query_groups = 1
        self.untie_embeddings_and_output_weights = False

        self.multi_latent_attention = False
        self.qk_pos_emb_head_dim = None
        self.qk_head_dim = None
        self.q_lora_rank = None
        self.kv_lora_rank = None
        self.v_head_dim = None
        self.qk_layernorm = False

        self.moe_grouped_gemm = False
        self.moe_permutation_async_comm = False
        self.moe_tp_extend_ep = False
        self.moe_alltoall_overlap_comm = False
        self.use_fused_moe_token_permute_and_unpermute = False
        self.moe_token_dispatcher_type = None
        self.seq_aux = False
        self.first_k_dense_replace = None
        self.moe_layer_freq = None
        self.moe_router_topk = 2
        self.num_experts = None
        self.n_shared_experts = None
        self.moe_ffn_hidden_size = None
        self.moe_router_load_balancing_type = 'aux_loss'
        self.moe_router_num_groups = None
        self.moe_router_group_topk = None
        self.moe_router_topk_scaling_factor = None
        self.norm_topk_prob = False
        self.moe_router_score_function = 'softmax'
        self.moe_router_enable_expert_bias = False
        self.log_throughput = False
        self.make_vocab_size_divisible_by = 128
        self.padded_vocab_size = None
        self.add_qkv_bias = False
        self.disable_bias_linear = False
        self.attention_dropout = 0.1
        self.init_method_std = 0.02
        self.hidden_dropout = 0.1
        self.position_embedding_type = 'learned_absolute'
        self.rotary_base = 10000
        self.normalization = 'LayerNorm'
        self.norm_epsilon = 1e-5
        self.swiglu = False
        self.no_masked_softmax_fusion = False
        self.attention_softmax_in_fp32 = False
        self.no_gradient_accumulation_fusion = False
        self.bf16 = True
        self.use_distributed_optimizer = False
        self.global_batch_size = None
        self.seq_length = None
        self.tokenizer_type = None
        self.tokenizer_name_or_path = None
        self.train_iters = None
        self.eval_iters = 100
        self.distributed_backend = 'nccl'
        self.no_shared_storage = False
        self.save_interval = None
        self.no_load_optim = None
        self.no_load_rng = None
        self.no_save_optim = None
        self.no_save_rng = None
        self.is_instruction_dataset = False
        self.is_pairwise_dataset = False
        self.variable_seq_lengths = False
        self.no_shuffle = False
        self.stage = None
        self.sequence_parallel = False
        self.micro_batch_size = None
        self.tensor_model_parallel_size = 1
        self.pipeline_model_parallel_size = 1
        self.expert_model_parallel_size = 1
        self.num_layers_per_virtual_pipeline_stage = None
        self.lr = None
        self.lr_decay_style = 'linear'
        self.min_lr = 0.0
        self.weight_decay = 0.01
        self.lr_warmup_fraction = None
        self.clip_grad = 1.0
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.999
        self.initial_loss_scale = 2 ** 32
        self.finetune = False
        self.load = None
        self.save = None
        self.pad_to_multiple_of = 8
        self.data_path = None
        self.split = None
        self.dataloader_type = None
        self.enable_high_availability = False
        self.context_parallel_size = 1
        self.context_parallel_algo = "ulysses_cp_algo"
        self.reset_position_ids = False
        self.optimizer = 'adam'
        self.do_sample = False
        self.use_kv_cache = False
        self.use_tp_pp_dp_mapping = False
        self.log_interval = 100
        self.load_checkpoint_loosely = False
        self.prompt_type = None
        self.prompt_type_path = TEMPLATES_DIR
        self.tokenizer_not_use_fast = False
        self.use_fused_rotary_pos_emb = False
        self.full_shuffle_instruction_dataset = False
        self.tokenizer_padding_side = 'right'
        self.num_workers = 2
        self.skip_train = False
        self.eval_interval = 1000
        self.seed = 1234
        self.vocab_extra_ids = 0
        self.algorithm = None
        self.add_qkv_bias = False
        self.ddp_bucket_size = None
        self.no_check_for_nan_in_loss_and_grad = False
        self.overlap_grad_reduce = False
        self.accumulate_allreduce_grads_in_fp32 = False
        self.pretrained_checkpoint = None
        self.reuse_fp32_param = False
        self.recompute_granularity = None
        self.recompute_method = None
        self.recompute_num_layers = None
        self.num_layer_list = None
        self.dataset_additional_keys = []
        self.npu_deterministic = False
        self.overlap_param_gather = False
        self.recompute_activation_function = False
        self.swap_attention = False
        self.ai_framework = None
        self.noop_layers = None
        self.attention_mask_type = 'causal'
        self.reset_attention_mask = False
        self.use_cp_send_recv_overlap = False
        self.use_fused_ring_attention_update = False
        self.dpo_loss_type = 'sigmoid'
        self.ref_model = ''
        self.refer_model_iter = 1
        self.wandb_exp_name = ''
        self.wandb_project = ''
        self.wandb_save_dir = ''

        self.use_ascend_coc = False
        self.coc_mode = -1
        self.coc_parallel_num = 1
        self.coc_fused_kernel = False
        self.swap_optimizer = False

        # used for multimodal models
        self.mm_model = None
        self.attention_bias = False
        self.moe_aux_loss_coeff = 0.001
        self.gemm_gradient_accumulation_fusion = False

        self.update(training_config, model_config)

        self.pad_to_multiple_of = self.tensor_model_parallel_size * self.context_parallel_size
