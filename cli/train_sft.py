# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
# Copyright (c) 2023; NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.
import sys

import torch
import hydra
from omegaconf import OmegaConf

from mindspeed_rl.trainer.sft_trainer import SFTTrainer
from mindspeed_rl.config_cls.megatron_config import MegatronConfig
from mindspeed_rl.datasets.instruction_dataset import InstructionDataset
from mindspeed_rl.datasets.dataloader import InstructionDataLoader
from mindspeed_rl.datasets.build_dataset import build_train_valid_test_datasets
from mindspeed_rl.datasets.utils import build_data_iter, get_train_valid_test_num_samples
from mindspeed_rl.utils import get_tokenizer, Loggers, synchronize_time, seed_all, parse_args_from_config

logger = Loggers('train_sft')


def sft_train():
    from megatron.core import parallel_state
    from megatron.core.utils import get_model_config
    from megatron.core.enums import ModelType
    from megatron.training import get_args
    from megatron.training.checkpointing import save_checkpoint
    from megatron.training.training import evaluate_and_print_results, setup_model_and_optimizer
    from megatron.training.utils import get_batch_on_this_cp_rank

    from mindspeed_llm.training import train
    from mindspeed_llm.training.initialize import set_jit_fusion_options
    from mindspeed_llm.training.utils import generate_actual_seq_len

    args = get_args()
    set_jit_fusion_options()

    start_time = synchronize_time()
    logger.info("sft training starting time: {}".format(start_time))

    model, optimizer, opt_param_scheduler = setup_model_and_optimizer(
        gpt_model_provider, ModelType.encoder_or_decoder)
    logger.info('after model, optimizer and learning rate scheduler are built')

    model_arch_config = get_model_config(model[0])

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
    train_dataset, valid_dataset, test_dataset = build_train_valid_test_datasets(
        data_prefix=args.data_path,
        splits_string=args.split,
        seq_length=args.seq_length + args.num_nextn_predict_layers,
        train_valid_test_num_samples=train_valid_test_num_samples,
        dataset_cls=InstructionDataset,
        tokenizer=tokenizer,
        parallel_state=parallel_state,
        full_shuffle_instruction_dataset=args.full_shuffle_instruction_dataset,
        no_shuffle=args.no_shuffle,
        reset_position_ids=args.reset_position_ids,
        prompt_type=args.prompt_type,
        prompt_type_path=args.prompt_type_path,
        seed=args.seed
    )
    logger.info('after datasets are built')

    # Backward compatibility, assume fixed batch size.
    if args.iteration > 0 and args.consumed_train_samples == 0:
        if args.train_samples is not None:
            raise ValueError('only backward compatiblity support for iteration-based training')
        args.consumed_train_samples = args.iteration * args.global_batch_size
    if args.iteration > 0 and args.consumed_valid_samples == 0:
        if args.train_samples is None:
            args.consumed_valid_samples = (args.iteration // args.eval_interval) * \
                                          args.eval_iters * args.global_batch_size

    #  build_dataloader
    train_dataloader = None
    if train_dataset is not None and len(train_dataset) > 0:
        train_dataloader = InstructionDataLoader(
            dataset=train_dataset,
            parallel_state=parallel_state,
            tokenizer=tokenizer,
            num_workers=args.num_workers,
            tokenizer_padding_side=args.tokenizer_padding_side,
            pad_to_multiple_of=args.pad_to_multiple_of if args.variable_seq_lengths else args.seq_length + args.num_nextn_predict_layers,
            variable_seq_lengths=args.variable_seq_lengths,
            num_nextn_predict_layers=args.num_nextn_predict_layers,
            micro_batch_size=args.micro_batch_size,
            comsumed_samples=args.consumed_train_samples,
            seed=args.seed
        )

    valid_dataloader = None
    if valid_dataset is not None and len(valid_dataset) > 0:
        valid_dataloader = InstructionDataLoader(
            dataset=valid_dataset,
            parallel_state=parallel_state,
            tokenizer=tokenizer,
            num_workers=args.num_workers,
            tokenizer_padding_side=args.tokenizer_padding_side,
            pad_to_multiple_of=args.pad_to_multiple_of if args.variable_seq_lengths else args.seq_length + args.num_nextn_predict_layers,
            variable_seq_lengths=args.variable_seq_lengths,
            num_nextn_predict_layers=args.num_nextn_predict_layers,
            micro_batch_size=args.micro_batch_size,
            comsumed_samples=args.consumed_valid_samples,
            seed=args.seed
        )

    test_dataloader = None
    if test_dataset is not None and len(test_dataset) > 0:
        test_dataloader = InstructionDataLoader(
            dataset=test_dataset,
            parallel_state=parallel_state,
            tokenizer=tokenizer,
            num_workers=args.num_workers,
            tokenizer_padding_side=args.tokenizer_padding_side,
            pad_to_multiple_of=args.pad_to_multiple_of if args.variable_seq_lengths else args.seq_length + args.num_nextn_predict_layers,
            variable_seq_lengths=args.variable_seq_lengths,
            num_nextn_predict_layers=args.num_nextn_predict_layers,
            micro_batch_size=args.micro_batch_size,
            comsumed_samples=0,
            seed=args.seed
        )

    # Flags to know if we need to do training/validation/testing.
    do_train = train_dataloader is not None and args.train_iters > 0
    do_valid = valid_dataloader is not None and args.eval_iters > 0
    do_test = test_dataloader is not None and args.eval_iters > 0
    flags = torch.tensor(
        [int(do_train), int(do_valid), int(do_test)],
        dtype=torch.long, device='cuda')

    torch.distributed.broadcast(flags, 0)

    args.do_train = getattr(args, "do_train", False) or flags[0].item()
    args.do_valid = getattr(args, "do_valid", False) or flags[1].item()
    args.do_test = getattr(args, "do_test", False) or flags[2].item()

    # build data_iterator
    train_data_iterator = []
    valid_data_iterator = []
    test_data_iterator_list = []
    if args.virtual_pipeline_model_parallel_size is not None:
        for i in range(len(model)):
            parallel_state.set_virtual_pipeline_model_parallel_rank(i)
            train_data_iterator.append(build_data_iter(train_dataloader, args.dataloader_type))
            valid_data_iterator.append(build_data_iter(valid_dataloader, args.dataloader_type))
            test_data_iterator_list.append(build_data_iter(test_dataloader, args.dataloader_type))
    else:
        train_data_iterator = build_data_iter(train_dataloader, args.dataloader_type)
        valid_data_iterator = build_data_iter(valid_dataloader, args.dataloader_type)
        test_data_iterator = build_data_iter(test_dataloader, args.dataloader_type)
        test_data_iterator_list = [test_data_iterator]

    logger.info('after dataloaders are built')

    # configure Trainer
    megatron_modules = {
        'train_func': train,
        'parallel_state': parallel_state,
        'save_checkpoint_func': save_checkpoint,
        'evaluate_fun': evaluate_and_print_results,
        "generate_seq_len_fun": generate_actual_seq_len,
        'batch_cp_func': get_batch_on_this_cp_rank
    }

    trainer = SFTTrainer(
        args=args,
        model=model,
        optimizer=optimizer,
        train_data_iterator=train_data_iterator,
        valid_data_iterator=valid_data_iterator,
        test_data_iterator_list=test_data_iterator_list,
        scheduler=opt_param_scheduler,
        process_non_loss_data_func=None,
        model_config=model_arch_config,
        **megatron_modules
    )

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
        transformer_layer_spec = get_gpt_layer_local_spec(args.num_experts, args.moe_grouped_gemm)

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
    from mindspeed_llm.training.arguments import parse_args_decorator
    import megatron

    parse_args = parse_args_decorator(megatron.training.arguments.parse_args)
    args = parse_args(extra_args_provider, ignore_unknown_args)
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

    validate_args(args, args_defaults)

    set_global_variables(args)

    if args.use_deter_comp:
        seed_all(args.seed)
        logger.info("deterministic computing is applied for npu.")

    # torch.distributed initialization
    def finish_mpu_init():
        args = get_args()
        # Pytorch distributed.
        _initialize_distributed()

        # Random seeds for reproducibility.
        if args.rank == 0:
            logger.info("> setting random seeds to {} ...".format(args.seed))
        _set_random_seed(args.seed, args.data_parallel_random_init)

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


def separate_config_and_parse_args(config):
    model_config = config.model
    sft_config = config.sft

    OmegaConf.set_struct(model_config, False)
    OmegaConf.set_struct(sft_config, False)

    sft_config_dict = OmegaConf.to_container(sft_config, resolve=True)
    model_config_dict = OmegaConf.to_container(model_config, resolve=True)

    megatron_config = MegatronConfig(sft_config_dict, model_config_dict)
    return megatron_config


@hydra.main(config_path='../configs', config_name='sft_qwen25_7b', version_base=None)
def main(config):
    megatron_config = separate_config_and_parse_args(config)
    initialize_megatron(config=megatron_config)
    sft_train()


if __name__ == '__main__':
    main()
