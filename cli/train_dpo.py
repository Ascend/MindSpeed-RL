import sys

import torch
import hydra
from omegaconf import OmegaConf

from mindspeed_rl.config_cls.megatron_config import MegatronConfig
from mindspeed_rl.datasets.prompt_dataset import PromptDataset
from mindspeed_rl.datasets.dataloader import PromptDataLoader
from mindspeed_rl.datasets.build_dataset import build_train_valid_test_datasets
from mindspeed_rl.datasets.utils import get_train_valid_test_num_samples
from mindspeed_rl.utils import get_tokenizer, Loggers, synchronize_time, seed_all, parse_args_from_config


logger = Loggers('train_dpo')


def add_config(rl_config_dict, profiler_config_dict, megatron_config):
    # add wandb config
    if 'use_wandb' in rl_config_dict:
        megatron_config.use_wandb = rl_config_dict['use_wandb']
    if 'wandb_exp_name' in rl_config_dict:
        megatron_config.wandb_exp_name = rl_config_dict['wandb_exp_name']
    if 'wandb_project' in rl_config_dict:
        megatron_config.wandb_project = rl_config_dict['wandb_project']
    if 'wandb_save_dir' in rl_config_dict:
        megatron_config.wandb_save_dir = rl_config_dict['wandb_save_dir']

    # add profile config
    for key, value in profiler_config_dict.items():
        if 'profile' in key:
            setattr(megatron_config, key, value)
        if key == 'profile_ranks' and value == 'all':
            setattr(megatron_config, key, -1)

    return megatron_config


def dpo_train():
    from megatron.core import parallel_state
    from megatron.core.utils import get_model_config
    from megatron.core.enums import ModelType
    from megatron.training import get_args
    from megatron.training.checkpointing import save_checkpoint
    from megatron.training.training import evaluate_and_print_results, setup_model_and_optimizer
    from megatron.training.utils import get_batch_on_this_cp_rank

    from mindspeed_llm.training.initialize import set_jit_fusion_options
    from mindspeed_llm.tasks.posttrain.dpo import DPOTrainer

    args = get_args()
    set_jit_fusion_options()

    start_time = synchronize_time()
    logger.info("dpo training starting time: {}".format(start_time))
    from mindspeed_llm.tasks.posttrain.base.base_trainer import BaseTrainer
    BaseTrainer.model_provider = model_provider_swap

    # build tokenizer
    tokenizer = get_tokenizer(args.tokenizer_name_or_path,
                              prompt_type=args.prompt_type, prompt_type_path=args.prompt_type_path)
    logger.info('after tokenizer is built')

    # build dataset
    train_valid_test_num_samples = get_train_valid_test_num_samples(
        train_samples=args.train_samples,
        train_iters=args.train_iters,
        global_batch_size=args.global_batch_size,
        eval_interval=args.eval_interval,
        eval_iters=args.eval_iters,
    )
    train_dataset, _, _ = build_train_valid_test_datasets(
        data_prefix=args.data_path,
        splits_string=args.split,
        seq_length=args.seq_length,
        train_valid_test_num_samples=train_valid_test_num_samples,
        dataset_cls=PromptDataset,
        tokenizer=tokenizer,
        full_shuffle_instruction_dataset=args.full_shuffle_instruction_dataset,
        parallel_state=parallel_state,
        no_shuffle=args.no_shuffle,
        reset_position_ids=args.reset_position_ids,
        prompt_type=args.prompt_type,
        prompt_type_path=args.prompt_type_path,
        seed=args.seed,
        extra_param=args
    )
    logger.info('after datasets are built')

    data_loader = PromptDataLoader(
        train_dataset, args.global_batch_size,
        args.num_workers, args.seed, args.dataset_additional_keys,
        args.no_shuffle, args.is_pairwise_dataset, tokenizer=tokenizer.tokenizer
    )
    data_iters = iter(data_loader)
    [next(data_iters) for _ in range(args.consumed_train_samples // args.global_batch_size)]

    logger.info('after dataloaders are built')

    trainer = DPOTrainer()

    trainer.train()


def gpt_model_provider(pre_process, post_process):
    """
    Builds the model.

    If you set the use_mcore_models to True, it will return the mcore GPT model and if not the legacy GPT model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss.
        Defaults to True.


    Returns:
        Union[GPTModel, megatron.legacy.model.GPTModel]: The returned model
    """
    from megatron.training import get_args
    from megatron.core.models.gpt import GPTModel
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    from megatron.core.transformer.spec_utils import import_module
    from megatron.training.arguments import core_transformer_config_from_args
    args = get_args()

    logger.info('building GPT model ...')
    # Experimental loading arguments from configs
    config = core_transformer_config_from_args(args)

    if args.spec is not None:
        transformer_layer_spec = import_module(args.spec)
    else:
        transformer_layer_spec = get_gpt_layer_local_spec(args.num_experts, args.moe_grouped_gemm, qk_layernorm=args.qk_layernorm)

    model = GPTModel(
        config=config,
        transformer_layer_spec=transformer_layer_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=True,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=args.rotary_percent,
        seq_len_interpolation_factor=args.rotary_seq_len_interpolation_factor
    )

    return model


def initialize_megatron(
        extra_args_provider=None,
        args_defaults=None,
        ignore_unknown_args=False,
        allow_no_cuda=False,
        skip_mpu_initialization=False,
        get_embedding_ranks=None,
        get_position_embedding_ranks=None,
        config=None,
):
    """Set global variables, initialize distributed, and
    set autoresume and random seeds.
    `allow_no_cuda` should not be set unless using megatron for cpu only
    data processing. In general this arg should not be set unless you know
    what you are doing.
    Returns a function to finalize distributed env initialization
    (optionally, only when args.lazy_mpu_init == True)
    """
    if args_defaults is None:
        args_defaults = {}

    origin_sys_argv = sys.argv
    sys.argv = [sys.argv[0]]
    parse_args_from_config(config)
    
    # Note: Importing this line activates the megatron_adapter 
    from mindspeed_llm.training.arguments import parse_args_decorator
    import megatron

    args = megatron.training.arguments.parse_args()
    sys.argv = origin_sys_argv

    if not allow_no_cuda:
        if not torch.cuda.is_available():
            raise ValueError("Megatron requires CUDA.")

    from megatron.core import parallel_state
    from megatron.training import get_args
    from megatron.training.arguments import validate_args
    from megatron.training.checkpointing import load_args_from_checkpoint
    from megatron.training.global_vars import set_global_variables
    from megatron.training.initialize import _set_random_seed, \
        _init_autoresume, _compile_dependencies, \
        _initialize_tp_communicators, _initialize_distributed

    if args.use_checkpoint_args or args_defaults.get("use_checkpoint_args", False):
        if args.load is None:
            raise ValueError("--use-checkpoints-args requires --load argument.")
        load_args_from_checkpoint(args)

    _ori_vari_length = None
    _ori_vari_length = args.variable_seq_lengths
    args.variable_seq_lengths = False
    validate_args(args, args_defaults)
    if _ori_vari_length is not None:
        args.variable_seq_lengths = _ori_vari_length

    set_global_variables(args)

    if args.npu_deterministic:
        seed_all(args.seed)
        logger.info("deterministic computing is applied for npu.")

    # torch.distributed initialization
    def finish_mpu_init():
        args = get_args()
        # Pytorch distributed.
        _initialize_distributed(get_embedding_ranks, get_position_embedding_ranks)

        # Random seeds for reproducibility.
        if args.rank == 0:
            logger.info("> setting random seeds to {} ...".format(args.seed))
        _set_random_seed(args.seed, args.data_parallel_random_init)
        if args.use_ascend_mc2:
            from mindspeed.core.tensor_parallel.ascend_turbo.initialize import initialize_cfg_from_args
            initialize_cfg_from_args(args)

    if skip_mpu_initialization:
        return None

    args = get_args()
    if args.lazy_mpu_init:
        args.use_cpu_initialization = True
        # delayed initialization of DDP-related stuff
        # We only set basic DDP globals
        parallel_state.set_tensor_model_parallel_world_size(args.tensor_model_parallel_size)
        # and return function for external DDP manager
        # to call when it has DDP initialized
        parallel_state.set_tensor_model_parallel_rank(args.rank)
        return finish_mpu_init
    else:
        # Megatron's MPU is the master. Complete initialization right away.
        finish_mpu_init()

        # Autoresume.
        _init_autoresume()

        # Compile dependencies.
        _compile_dependencies()

        if args.tp_comm_overlap:
            _initialize_tp_communicators()

        # No continuation function
        return None


def model_provider_swap(self, pre_process, post_process):
    from megatron.training import get_args, print_rank_0
    from megatron.training.arguments import core_transformer_config_from_args
    from megatron.training.yaml_arguments import core_transformer_config_from_yaml
    from megatron.core.transformer.spec_utils import import_module
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    from mindspeed_llm.core.models.gpt.gpt_model import GPTModel
    args = get_args()
    use_te = args.transformer_impl == "transformer_engine"

    print_rank_0('building GPT model ...')
    # Experimental loading arguments from yaml
    if args.yaml_cfg is not None:
        config = core_transformer_config_from_yaml(args, "language_model")
    else:
        config = core_transformer_config_from_args(args)

    if args.spec is not None:
        transformer_layer_spec = import_module(args.spec)
    else:
        transformer_layer_spec = get_gpt_layer_local_spec(args.num_experts, args.moe_grouped_gemm, qk_layernorm=args.qk_layernorm)
    mtp_block_spec = None

    model = GPTModel(
        config=config,
        transformer_layer_spec=transformer_layer_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=True,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=args.rotary_percent,
        seq_len_interpolation_factor=args.rotary_seq_len_interpolation_factor,
        mtp_block_spec=mtp_block_spec,
    )

    return model


def separate_config_and_parse_args(config):
    model_config = config.model
    dpo_config = config.megatron_training
    rl_config = config.get('rl_config', OmegaConf.create())
    profiler_config = config.get('profiler_config', OmegaConf.create()).get('integrated', OmegaConf.create())

    OmegaConf.set_struct(model_config, False)
    OmegaConf.set_struct(dpo_config, False)
    OmegaConf.set_struct(rl_config, False)
    OmegaConf.set_struct(profiler_config, False)

    dpo_config_dict = OmegaConf.to_container(dpo_config, resolve=True)
    model_config_dict = OmegaConf.to_container(model_config, resolve=True)
    rl_config_dict = OmegaConf.to_container(rl_config, resolve=True)
    profiler_config_dict = OmegaConf.to_container(profiler_config, resolve=True)

    megatron_config = MegatronConfig(dpo_config_dict, model_config_dict)
    megatron_config = add_config(rl_config_dict, profiler_config_dict, megatron_config)
    return megatron_config


@hydra.main(config_path='../configs', config_name='dpo_qwen3_30b_a3b_A3', version_base=None)
def main(config):
    megatron_config = separate_config_and_parse_args(config)
    initialize_megatron(config=megatron_config)
    dpo_train()


if __name__ == '__main__':
    main()