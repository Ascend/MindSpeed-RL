# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.

import os
from pathlib import Path

from typing import Dict

from mindspeed_rl.config_cls.base_config import BaseConfig


cur_file_dir = Path(__file__).absolute().parent
TEMPLATES_DIR = os.path.join(cur_file_dir.parent.parent, "configs/model/templates.json")


class MegatronConfig(BaseConfig):
    """Megatron model configuration class for large-scale transformer training.

    This class manages comprehensive model architecture, training, and optimization
    parameters for distributed training of large language models using Megatron framework.

    Attributes:
        use_mcore_models (bool): Whether to use Megatron Core (MCore) models instead of legacy models.
        spec (str): Specify the model specification as <module_location:function_name> pair for custom models.
        sequence_parallel (bool): Whether to use sequence parallelism within tensor parallel groups.
        use_flash_attn (bool): Whether to use Flash Attention for memory-efficient attention computation.
        use_rotary_position_embeddings (bool): Whether to use Rotary Position Embeddings (RoPE).
        use_fused_rmsnorm (bool): Whether to use fused RMSNorm kernel for better performance.
        use_fused_swiglu (bool): Whether to use fused SwiGLU kernel for better performance.
        shape_order (str): Input tensor shape ordering for Flash Attention. Options: 'SBH', 'BSH'.
        no_bias_dropout_fusion (bool): Whether to disable fusion of bias and dropout operations.
        rope_scaling_type (str): Type of RoPE scaling used for long sequences. Options: 'linear', 'yarn'.
        rope_scaling_factor (float): Scaling factor for RoPE extension. Defaults to 1.0 (no scaling).
        low_freq_factor (float): Low frequency adjustment factor for yarn RoPE scaling.
        high_freq_factor (float): High frequency adjustment factor for yarn RoPE scaling.
        original_max_position_embeddings (int): Original maximum position embeddings before scaling.
        max_position_embeddings (int): Maximum position embeddings (context length) supported by the model.
        beta_fast (int): Fast beta parameter for yarn RoPE scaling. Defaults to 32.
        beta_slow (int): Slow beta parameter for yarn RoPE scaling. Defaults to 1.
        rope_scaling_mscale (float): Magnitude scaling factor for yarn RoPE. Defaults to 1.0.
        rope_scaling_mscale_all_dim (float): All-dimension scaling factor for yarn RoPE. Defaults to 0.0.
        rope_scaling_original_max_position_embeddings (int): Original max length reference for yarn RoPE scaling.
        num_layers (int): Total number of transformer layers in the model.
        hidden_size (int): Dimension of hidden layers (embedding and hidden states).
        ffn_hidden_size (int): Dimension of feed-forward network intermediate layers.
        num_attention_heads (int): Number of attention heads for multi-head attention.
        kv_channels (int): Dimension of projection weights in multi-head attention. 
            Auto-calculated as hidden_size // num_attention_heads if not provided.
        group_query_attention (bool): Whether to use Group Query Attention (GQA) for efficient inference.
        num_query_groups (int): Number of query groups for GQA. Defaults to 1 (standard MHA).
        untie_embeddings_and_output_weights (bool): Whether to untie input embedding and output linear weights.
        multi_latent_attention (bool): Whether to use Multi-head Latent Attention (MLA) like DeepSeek.
        qk_pos_emb_head_dim (int): Head dimension for QK position embeddings in MLA.
        qk_head_dim (int): Head dimension for QK projections in MLA (self-attention only).
        q_lora_rank (int): Low-rank dimension for Q projection compression in MLA.
        kv_lora_rank (int): Low-rank dimension for KV projection compression in MLA.
        v_head_dim (int): Head dimension for V projection in MLA.
        qk_layernorm (bool): Whether to apply LayerNorm to QK attention embeddings in MLA.
        moe_grouped_gemm (bool): Whether to use grouped GEMM for efficient MoE computation.
        moe_permutation_async_comm (bool): Whether to use async communication for MoE permutation.
        moe_tp_extend_ep (bool): Whether to extend expert parallelism using TP groups instead of sharding.
        moe_alltoall_overlap_comm (bool): Whether to overlap all-to-all communication with MoE computation.
        use_fused_moe_token_permute_and_unpermute (bool): Whether to use fused token permute/unpermute kernels.
        moe_token_dispatcher_type (str): Token dispatching implementation type for MoE.
        seq_aux (bool): Whether to compute auxiliary loss at sequence level instead of batch level.
        first_k_dense_replace (int): Number of initial layers to keep dense (non-MoE).
        moe_layer_freq (int): Frequency of MoE layers (every N-th layer is MoE).
        moe_router_topk (int): Number of experts to route each token to. Defaults to 2.
        num_experts (int): Number of experts in MoE layers. None means dense model (no MoE).
        n_shared_experts (int): Number of shared experts accessible to all tokens.
        moe_ffn_hidden_size (int): Hidden dimension for MoE feed-forward layers.
        moe_router_load_balancing_type (str): Load balancing method for MoE routing. Options: 'aux_loss', 'sinkhorn'.
        moe_router_num_groups (int): Number of expert groups for hierarchical routing.
        moe_router_group_topk (int): Number of groups to select in group-limited greedy routing.
        moe_router_topk_scaling_factor (float): Scaling factor for routing scores.
        norm_topk_prob (bool): Whether to normalize top-K routing probabilities.
        moe_router_score_function (str): Score function for routing. Options: 'softmax', 'sigmoid'.
        moe_router_enable_expert_bias (bool): Whether to use dynamic expert bias for load balancing.
        log_throughput (bool): Whether to calculate and log training throughput per GPU.
        make_vocab_size_divisible_by (int): Padding value to make vocabulary size divisible by this number.
            Defaults to 128 for computational efficiency.
        padded_vocab_size (int): Actual padded vocabulary size after applying divisibility constraint.
        add_qkv_bias (bool): Whether to add bias specifically to QKV linear layers.
        disable_bias_linear (bool): Whether to disable bias in all linear layers.
        attention_dropout (float): Dropout rate for attention weights. Defaults to 0.1.
        init_method_std (float): Standard deviation for weight initialization. Defaults to 0.02.
        hidden_dropout (float): Dropout rate for hidden states. Defaults to 0.1.
        position_embedding_type (str): Position encoding method. Options: 'learned_absolute', 'rope'.
        rotary_base (int): Base period for Rotary Position Embedding (RoPE). Defaults to 10000.
        normalization (str): Normalization method. Options: 'LayerNorm', 'RMSNorm'.
        norm_epsilon (float): Epsilon value for numerical stability in normalization. Defaults to 1e-5.
        swiglu (bool): Whether to use SwiGLU activation function in feed-forward networks.
        no_masked_softmax_fusion (bool): Whether to disable fusion of masked softmax operations.
        attention_softmax_in_fp32 (bool): Whether to compute attention softmax in FP32 for numerical stability.
        no_gradient_accumulation_fusion (bool): Whether to disable fusion of gradient accumulation operations.
        bf16 (bool): Whether to use Brain Float 16 (BF16) mixed precision training. Defaults to True.
        use_distributed_optimizer (bool): Whether to use distributed optimizer for large-scale training.
        global_batch_size (int): Total batch size across all data parallel ranks.
        seq_length (int): Training sequence length (context window size).
        tokenizer_type (str): Tokenizer implementation type.
        tokenizer_name_or_path (str): HuggingFace tokenizer name or local path.
        train_iters (int): Total number of training iterations.
        eval_iters (int): Number of iterations for evaluation. Defaults to 100.
        distributed_backend (str): Distributed communication backend. Options: 'nccl', 'gloo'. Defaults to 'nccl'.
        no_shared_storage (bool): Whether nodes do not share common storage (affects checkpointing).
        save_interval (int): Interval (in iterations) for saving checkpoints.
        no_load_optim (bool): Whether to skip loading optimizer state from checkpoint.
        no_load_rng (bool): Whether to skip loading RNG state from checkpoint.
        no_save_optim (bool): Whether to exclude optimizer state from checkpoints.
        no_save_rng (bool): Whether to exclude RNG state from checkpoints.
        is_instruction_dataset (bool): Whether dataset follows instruction-following format.
        is_pairwise_dataset (bool): Whether dataset is pairwise (chosen/rejected) for RLHF training.
        variable_seq_lengths (bool): Whether to support variable sequence lengths in batch.
        no_pad_to_seq_lengths (bool): Whether to disable padding to fixed sequence length.
        no_shuffle (bool): Whether to disable dataset shuffling.
        stage (str): Training stage identifier for multi-stage training.
        micro_batch_size (int): Batch size per forward pass on single device.
        tensor_model_parallel_size (int): Degree of tensor model parallelism. Defaults to 1.
        pipeline_model_parallel_size (int): Degree of pipeline model parallelism. Defaults to 1.
        expert_model_parallel_size (int): Degree of expert model parallelism for MoE. Defaults to 1.
        num_layers_per_virtual_pipeline_stage (int): Number of layers per virtual pipeline stage (VPP).
        lr (float): Initial learning rate.
        lr_decay_style (str): Learning rate decay schedule. Options: 'linear', 'cosine'. Defaults to 'linear'.
        min_lr (float): Minimum learning rate at end of training. Defaults to 0.0.
        weight_decay (float): L2 regularization coefficient. Defaults to 0.01.
        lr_warmup_fraction (float): Fraction of training steps for learning rate warmup.
        clip_grad (float): Gradient clipping threshold. Defaults to 1.0.
        adam_beta1 (float): Adam optimizer beta1 parameter. Defaults to 0.9.
        adam_beta2 (float): Adam optimizer beta2 parameter. Defaults to 0.999.
        initial_loss_scale (float): Initial loss scale for loss scaling. Defaults to 2**32.
        finetune (bool): Whether to finetune from a pretrained checkpoint.
        load (str): Directory path to load checkpoint from.
        save (str): Directory path for saving checkpoints.
        pad_to_multiple_of (int): Pad sequence length to be multiple of this value. Defaults to 8.
        data_path (str): Path to training dataset directory or file.
        split (str): Data split ratios for train/valid/test, e.g., "98,1,1".
        dataloader_type (str): Data loading strategy. Options: 'single', 'cyclic'.
        enable_high_availability (bool): Whether to enable high availability features for fault tolerance.
        context_parallel_size (int): Degree of context parallelism for long sequences. Defaults to 1.
        context_parallel_algo (str): Context parallelism algorithm. Defaults to 'ulysses_cp_algo'.
        reset_position_ids (bool): Whether to reset position IDs after end-of-document token.
        optimizer (str): Optimizer type. Options: 'adam', 'sgd'. Defaults to 'adam'.
        do_sample (bool): Whether to enable sampling during actor generation in RL training.
        use_kv_cache (bool): Whether to use key-value cache for autoregressive generation.
        use_tp_pp_dp_mapping (bool): Whether to use TP-PP-DP rank ordering instead of TP-DP-PP.
        log_interval (int): Interval (in iterations) for logging metrics. Defaults to 100.
        load_checkpoint_loosely (bool): Whether to allow partial/loose checkpoint loading.
        prompt_type (str): Prompt template type for instruction formatting (e.g., 'qwen').
        prompt_type_path (str): Path to JSON file containing prompt templates. Defaults to TEMPLATES_DIR.
        tokenizer_not_use_fast (bool): Whether to disable HuggingFace fast tokenizer implementation.
        use_fused_rotary_pos_emb (bool): Whether to use fused rotary position embedding kernel.
        full_shuffle_instruction_dataset (bool): Whether to fully shuffle instruction-formatted datasets.
        tokenizer_padding_side (str): Padding side for tokenization. Options: 'right', 'left'. Defaults to 'right'.
        num_workers (int): Number of data loading worker processes. Defaults to 2.
        skip_train (bool): Whether to skip training and only run evaluation.
        eval_interval (int): Interval (in iterations) between evaluations. Defaults to 1000.
        seed (int): Random seed for reproducibility. Defaults to 1234.
        vocab_extra_ids (int): Number of extra vocabulary IDs for special tokens. Defaults to 0.
        algorithm (str): Training algorithm type identifier.
        ddp_bucket_size (int): Gradient bucketing size for DDP all-reduce.
        no_check_for_nan_in_loss_and_grad (bool): Whether to disable NaN checking in loss and gradients.
        overlap_grad_reduce (bool): Whether to overlap gradient all-reduce with backward computation.
        accumulate_allreduce_grads_in_fp32 (bool): Whether to accumulate and all-reduce gradients in FP32.
        pretrained_checkpoint (str): Path to pretrained model for finetuning initialization.
        reuse_fp32_param (bool): Whether to reuse FP32 parameter copies to save memory.
        recompute_granularity (str): Activation checkpointing granularity. Options: 'full', 'selective'.
        recompute_method (str): Checkpointing distribution method. Options: 'uniform', 'block'.
        recompute_num_layers (int): Number of layers per checkpointing unit.
        num_layer_list (str): Comma-separated layer distribution across pipeline stages (e.g., "4,4,4,4").
        dataset_additional_keys (list): Additional data fields to load from dataset.
        npu_deterministic (bool): Whether to enable deterministic operations on NPU.
        overlap_param_gather (bool): Whether to overlap parameter gathering with forward computation.
        recompute_activation_function (bool): Whether to recompute activation functions in backward pass.
        swap_attention (bool): Whether to enable attention activation swapping to CPU memory.
        ai_framework (str): AI framework backend. Options: None, 'mindspore'.
        noop_layers (str): Layers to skip (no-op) for specific parallelism configurations.
        attention_mask_type (str): Attention mask type for context parallelism. Defaults to 'causal'.
        reset_attention_mask (bool): Whether to reset attention mask for each context parallel chunk.
        use_cp_send_recv_overlap (bool): Whether to overlap send/recv operations in context parallelism.
        use_fused_ring_attention_update (bool): Whether to use fused ring attention update in context parallelism.
        dpo_loss_type (str): DPO loss computation method. Defaults to 'sigmoid'.
        ref_model (str): Reference model path for DPO/RLHF training.
        refer_model_iter (int): Reference model update frequency. Defaults to 1.
        use_ascend_coc (bool): Whether to enable Ascend CoC (Chain of Command) memory optimization.
        coc_mode (int): CoC operation mode. 0=original, 1=rewrite, 2=CoC default. Defaults to -1 (disabled).
        coc_parallel_num (int): Parallelism degree for CoC operations. Defaults to 1.
        coc_fused_kernel (bool): Whether to use fused kernels in CoC mode.
        swap_optimizer (bool): Whether to enable optimizer state swapping to CPU memory.
        mm_model (str): Multimodal model configuration identifier.
        attention_bias (bool): Whether to use bias in attention layers.
        moe_aux_loss_coeff (float): Coefficient for MoE auxiliary load balancing loss. Defaults to 0.001.
        gemm_gradient_accumulation_fusion (bool): Whether to fuse GEMM with gradient accumulation.
        lora_ckpt_filter (bool): Whether to filter LoRA parameters when loading checkpoint.
        lora_r (int): LoRA rank dimension. Defaults to 8.
        lora_alpha (int): LoRA scaling alpha parameter. Defaults to 16.
        lora_fusion (bool): Whether to fuse LoRA weights into base weights.
        lora_target_modules (list): List of module names to apply LoRA adaptation.
    """

    def __init__(self, training_config: Dict, model_config: Dict):
        """Initialize MegatronConfig with training and model configurations.

        Args:
            training_config (Dict): Dictionary containing the training configuration parameters.
            model_config (Dict): Dictionary containing the model configuration parameters.
        """
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
        self.no_pad_to_seq_lengths = False
        self.no_shuffle = False
        self.stage = None
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
        self.use_ascend_coc = False
        self.coc_mode = -1
        self.coc_parallel_num = 1
        self.coc_fused_kernel = False
        self.swap_optimizer = False
        self.mm_model = None
        self.attention_bias = False
        self.moe_aux_loss_coeff = 0.001
        self.gemm_gradient_accumulation_fusion = False
        self.lora_ckpt_filter = False
        self.lora_r = 8
        self.lora_alpha = 16
        self.lora_fusion = False
        self.lora_target_modules = None
        self.transformer_impl = "local"

        # Update configuration from input dictionaries
        self.update(training_config, model_config)

        # Adjust padding to be multiple of tensor parallel size and context parallel size
        self.pad_to_multiple_of = self.tensor_model_parallel_size * self.context_parallel_size
