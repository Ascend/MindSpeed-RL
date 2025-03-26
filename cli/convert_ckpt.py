import argparse
import importlib
import logging as logger
import sys
import torch
import torch.multiprocessing as mp

MODULE_ROOT = "mindspeed_llm.tasks.checkpoint"


def load_plugin(plugin_type, name):
    if name == '':
        module_name = f"{MODULE_ROOT}.{plugin_type}"
    else:
        module_name = f"{MODULE_ROOT}.{plugin_type}_{name}"
    try:
        plugin = importlib.import_module(module_name)
    except ModuleNotFoundError:
        module_name = f"{MODULE_ROOT}.{name}"
        try:
            plugin = importlib.import_module(module_name)
        except ModuleNotFoundError:
            sys.exit(f"Unable to load {plugin_type} plugin {name}. Exiting.")

    if not hasattr(plugin, 'add_arguments'):
        sys.exit(f"{module_name} module is not a plugin. Exiting.")

    logger.info(f"Loaded {module_name} as the {plugin_type}.")
    return plugin


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


def rm_model_provider(pre_process, post_process):
    """
    Builds the model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embeddings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss.
        Defaults to True.

    Returns:
        GPTRewardModel: The returned model
    """
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    from megatron.core.transformer.spec_utils import import_module
    from megatron.training import get_args
    from megatron.training.arguments import core_transformer_config_from_args
    from mindspeed_llm.tasks.posttrain.orm.orm_model import GPTRewardModel
    args = get_args()
    
    logger.info('building RM GPT model ...')
    # Experimental loading arguments from configs
    config = core_transformer_config_from_args(args)

    if args.spec is not None:
        transformer_layer_spec = import_module(args.spec)
    else:
        transformer_layer_spec = get_gpt_layer_local_spec(args.num_experts, args.moe_grouped_gemm)

    if (not args.untie_embeddings_and_output_weights) and (args.pipeline_model_parallel_size > 1):
        args.untie_embeddings_and_output_weights = True
        logger.warning(
            "untie_embeddings_and_output_weights is set to True, "
            "since output_layer is not used in Outcome Reward model training."
        )

    model = GPTRewardModel(
        config=config,
        transformer_layer_spec=transformer_layer_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        post_layer_norm=not args.no_post_layer_norm,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=True,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=args.rotary_percent,
    )

    return model


def main():

    parser = argparse.ArgumentParser(description="Megatron Checkpoint Utility Arguments",
                                     allow_abbrev=False, conflict_handler='resolve')

    parser.add_argument('--model-type', type=str, required=True,
                        choices=['GPT', 'BERT'],
                        help='Type of the model')
    parser.add_argument('--loader', type=str, default='megatron',
                        help='Module name to load checkpoint, should be on python path')
    parser.add_argument('--load-model-type', type=str, nargs='?',
                        default=None, const=None, choices=['hf', 'mg', 'optim'],
                        help='Module name to load checkpoint, should be on python path')
    parser.add_argument('--saver', type=str, default='megatron',
                        help='Module name to save checkpoint, should be on python path')
    parser.add_argument('--load-dir', type=str, required=True,
                        help='Directory to load model checkpoint from')
    parser.add_argument('--save-dir', type=str, required=True,
                        help='Directory to save model checkpoint to')
    parser.add_argument('--max-queue-size', type=int, default=50,
                        help='Maximum number of tensors in the queue')
    parser.add_argument('--no-checking', action='store_false',
                        help='Do not perform checking on the name and ordering of weights',
                        dest='checking')
    parser.add_argument('--spec', type=str, default=None, nargs='*',
                       help='Specify the <module_location function_name> pair '
                            'that returns a spec to customize transformer layer, depending on the use case.')
    parser.add_argument('--model-type-hf', type=str, default="llama2",
                        choices=['baichuan', 'baichuan2', 'llama2', 'mixtral', 'chatglm3', 'gemma', 'gemma2',
                                 'bloom', 'bloom_3b', 'qwen', 'internlm2', 'deepseek2', 'minicpm', 'minicpm3', 'minicpm-moe',
                                 'deepseek2-lite', 'qwen2-moe', 'phi3.5', 'phi3.5-moe', 'hunyuan'],
                        help='model type of huggingface')
    parser.add_argument('--ckpt-cfg-path', type=str, default="configs/checkpoint/model_cfg.json",
                        help="Path to the config directory. If not specified, the default path in the repository will be used.")
    parser.add_argument('--qlora-nf4', action='store_true',
                       help='use bitsandbytes nf4 to quantize model.')
    parser.add_argument('--orm', action="store_true", default=False,
                        help='Specify the ORM ckpt conversion, convert additional rm_head layer in ORM.')
    known_args, _ = parser.parse_known_args()


    if known_args.load_model_type == 'optim':
        loader = load_plugin('loader', known_args.load_model_type)
        loader.add_arguments(parser)
        args = parser.parse_args()
        model_provider = gpt_model_provider
        loader.load_checkpoint(model_provider, args)
    else:
        use_saver = known_args.load_model_type is None
        if use_saver:
            loader = load_plugin('loader', known_args.loader)
            saver = load_plugin('saver', known_args.saver)
        else:
            loader = load_plugin('loader', known_args.load_model_type)
            saver = load_plugin('saver', '')

        loader.add_arguments(parser)
        saver.add_arguments(parser)

        args = parser.parse_args()

        queue = mp.Queue(maxsize=args.max_queue_size)
        model_provider = rm_model_provider if args.orm else gpt_model_provider
        if args.orm and not args.use_mcore_models:
            raise AssertionError("Currently Outcome Reward Model only support Mcore models")

        logger.info("Starting saver...")
        saver_proc = mp.Process(target=saver.save_model_checkpoint, args=(model_provider, queue, args))
        saver_proc.start()

        logger.info("Starting loader...")
        loader.load_checkpoint(model_provider, queue, args)

        logger.info("Waiting for saver to complete...")
        saver_proc.join()


if __name__ == '__main__':
    main()

