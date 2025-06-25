from abc import ABC, abstractmethod
import re
import torch


class BaseWeightAdaptor(ABC):
    def __init__(self, model_config):
        """
        Base class for weight adaptors.
        A weight adaptor provide a set of tools to transfer from training weight to inference weight.
        Currently, we support MegatronVLLMWeightAdaptor only.
        Args:
        """
        self.model_config = model_config
        pass

    @abstractmethod
    def adjust_megatron_param_dict(self, param_dict, train_tp_size):
        """
        adjust megatron param dict to remove mindspeed only features such
        as mla-split-mm in DSv3
        """
        pass

    @abstractmethod
    def replace_name_i2t(self, inference_name):
        """
        transfer inference weight name to training weight name
        """
        pass

    @abstractmethod
    def convert_weight_t2i(self, weight_name, weight):
        """
        Transfer weight format to inference engine's format.
        """
        pass

    @abstractmethod
    def get_weight_buffer_meta(self, param_dict, valid_names=None):
        """
        Given inference param_dict, build a weight buffer meta data in train weight style.
        Needs model specific coding when multiple inference params correspond to one training param,
         or one inference param corresponds to multiple training params.
        Return a dictionary containing name to a shape and dtype.
        """
        pass


class MegatronVLLMWeightAdaptor(BaseWeightAdaptor):
    def __init__(self, model_config):
        super(MegatronVLLMWeightAdaptor, self).__init__(model_config)
        self.meta_info = None
        self.params_mapping = [
            # (megatron core gpt model name, vllm model name)
            ("embedding.word_embeddings", "model.embed_tokens"),
            ("self_attention.linear_qkv", "self_attn.qkv_proj"),
            ("self_attention.linear_proj", "self_attn.o_proj"),
            ("input_layernorm", "input_layernorm"),
            ("pre_mlp_layernorm", "post_attention_layernorm"),
            ("mlp.linear_fc1", "mlp.gate_up_proj"),
            ("mlp.linear_fc2", "mlp.down_proj"),
            ("decoder.final_layernorm", "model.norm"),
            ("output_layer", "lm_head"),
        ]

    def replace_name_i2t(self, inference_name):
        """
        transfer inference weight name to training weight name
        """
        for m_name, v_name in self.params_mapping:
            if v_name not in inference_name:
                continue
            if "layers" in inference_name:  # deal with decoder layers
                inference_name = inference_name.replace("model", "decoder")
                vllm_name_list = inference_name.split(".")
                if "layer_norm_weight" in vllm_name_list or "layer_norm_bias" in vllm_name_list:
                    param_name_list = vllm_name_list[:3]
                    param_name_list.append(m_name)
                    param_name = ".".join(param_name_list)
                else:
                    param_name_list = vllm_name_list[:3]
                    weight_or_bias = vllm_name_list[-1]
                    param_name_list.append(m_name)
                    if weight_or_bias in ['weight', 'bias']:
                        param_name_list.append(weight_or_bias)
                    param_name = ".".join(param_name_list)
                return param_name
            else:
                param_name = inference_name.replace(v_name, m_name)
                return param_name

    # Identity operation here, can be rewritten model by model.
    def _transfer_loaded_weight(self, loaded_weight, name, infer_tp_size):
        return loaded_weight

    def convert_weight_t2i(self, actor_weights, vllm_model, **kargs):
        """
        Transfer weight format to inference engine's format, and load weight to inference engine.
        This will be implemented in the next version.
        """
        pass

    def convert_weight_name_meta(self, weight_names):
        return weight_names

    def get_weight_buffer_meta(self, model, valid_names=None):
        weight_buffer_meta = {}
        for name, param in sorted(model.named_parameters()):
            if valid_names and name not in valid_names:
                continue
            else:
                weight_buffer_meta[name] = {'shape': param.shape, 'dtype': param.dtype}
        return weight_buffer_meta

    def adjust_megatron_param_dict(self, param_dict, train_tp_size):
        for name in list(param_dict.keys()):
            # use mm split
            if name.endswith("linear_qk_nope.weight"):
                num_attention_heads_per_tp = int(self.model_config.num_attention_heads / train_tp_size)
        
                qk_nope_head_dim = self.model_config.qk_nope_head_dim
                qk_rope_head_dim = self.model_config.qk_rope_head_dim
                v_head_dim = self.model_config.v_head_dim
        
                prefix = name.replace("linear_qk_nope.weight", "")
                param_dict[prefix + "linear_qb.weight"] = torch.nn.Parameter(torch.concat([param_dict[prefix + "linear_qk_nope.weight"].reshape(num_attention_heads_per_tp, qk_nope_head_dim, -1), param_dict[prefix + "linear_qk_rope.weight"].reshape(num_attention_heads_per_tp, qk_rope_head_dim, -1)], dim=1).reshape(num_attention_heads_per_tp * (qk_nope_head_dim + qk_rope_head_dim), -1))
                param_dict[prefix + "linear_qb.weight"].tensor_model_parallel = True
                param_dict[prefix + "linear_qb.weight"].partition_dim = param_dict[prefix + "linear_qk_nope.weight"].partition_dim
                param_dict[prefix + "linear_kvb.weight"] = torch.nn.Parameter(torch.concat([param_dict[prefix + "linear_kv_nope.weight"].reshape(num_attention_heads_per_tp, qk_nope_head_dim, -1), param_dict[prefix + "linear_v.weight"].reshape(num_attention_heads_per_tp, v_head_dim, -1)], dim=1).reshape(num_attention_heads_per_tp * (qk_nope_head_dim + v_head_dim), -1))
                param_dict[prefix + "linear_kvb.weight"].tensor_model_parallel = True
                param_dict[prefix + "linear_kvb.weight"].partition_dim = param_dict[prefix + "linear_kv_nope.weight"].partition_dim
        
        return param_dict

    @staticmethod
    def global2local_layer(name, num_layer_list, vpp_rank=0, global2local_map=None):
        """
        Transform the model name in each model_chunk in global space to local space
        """
        layer_name = 'layers'
        num_layer_offset = vpp_rank * sum(num_layer_list)

        if layer_name in name:  # belong to an intermediate layer
            split_name = name.split('.')

            # find the num next to split_name
            for layer_num_idx, name in enumerate(split_name, start=1):
                if name == layer_name:
                    break

            # check the name
            if len(split_name) < layer_num_idx + 1 or not split_name[layer_num_idx].isdigit():
                raise ValueError(f'split_name = {split_name}')

            # increment layer_num_idx by layer_offset
            if global2local_map is None:
                global_idx = int(split_name[layer_num_idx]) - num_layer_offset
                for layers_in_pp in num_layer_list:
                    global_idx -= layers_in_pp
                    if global_idx < 0:
                        local_index = global_idx + layers_in_pp
                        break
            else:
                local_index = global2local_map[int(split_name[layer_num_idx])]

            split_name[layer_num_idx] = str(local_index)
            name = '.'.join(split_name)  # weight name in inference_tp_model

        return name

    @staticmethod
    def get_weight_names_per_pp(layer_list, vllm_names, layers_num=None, vpp_size=0, noop_layers=None):
        ## add protection for default kwargs
        if not layers_num:
            if vpp_size > 0:
                ValueError(f"layers_num is required with vpp_size = {vpp_size}")
            layers_num = sum(layer_list)

        end_layer = layers_num - 1

        def get_weight_names_in_range(layer_range, names: list, noop_layers=None, layer_name='layers') -> list:
            """
            Extract weights in a given range and also include the weights before and after the range as needed.
            """
            start, end = layer_range

            layer_idx_list = [layer_idx for layer_idx in range(start, end + 1)]
            if noop_layers:
                layer_idx_list = [
                    layer_idx - sum(1 for i in noop_layers if i <= layer_idx) for layer_idx in layer_idx_list if
                    layer_idx not in noop_layers
                ]
            last_layer_index = end_layer
            names_in_range = []

            # add names before decoder layers
            if start == 0:
                for name in names:
                    if layer_name not in name:
                        names_in_range.append(name)
                    else:
                        break

            for name in names:
                # Extract layer number from weight
                match = re.match(r'.*\.layers\.(\d+)', name)
                if match:
                    layer_num = int(match.group(1))
                    if layer_num in layer_idx_list:
                        names_in_range.append(name)

            # add names after decode layers
            if end == last_layer_index:
                for name in reversed(names):
                    if layer_name not in name:
                        names_in_range.append(name)
                    else:
                        break
            return names_in_range

        stage_layers_num = sum(layer_list)
        weight_names_per_vpp_combined = [[] for _ in layer_list]
        for vpp_rank in range(vpp_size):
            start_layer = vpp_rank * stage_layers_num
            for pp_rank, layers_in_vpp_rank in enumerate(layer_list):
                vpp_layers_range = (start_layer, start_layer + layers_in_vpp_rank - 1)
                weight_names_per_vpp = get_weight_names_in_range(vpp_layers_range, vllm_names, noop_layers)
                weight_names_per_vpp_combined[pp_rank].append(weight_names_per_vpp)

                start_layer += layers_in_vpp_rank

        return weight_names_per_vpp_combined


class DeepSeekMVWeightAdaptor(MegatronVLLMWeightAdaptor):
    """
    Megatron-vLLM WeightAdaptor for DeepSeek model architectures.
    """
    def __init__(self, model_config):
        super(DeepSeekMVWeightAdaptor, self).__init__(model_config)
        self.meta_info = {'replace': {'kv_a_proj_with_mqa': 'qkv_proj'},
                          'delete': ['q_a_proj']}
        self.params_mapping = [
            # (megatron core gpt model name, vllm model name)
            ("embedding.word_embeddings", "model.embed_tokens"),
            ("self_attention.linear_qkv", "self_attn.qkv_proj"),  # q_a_proj, kv_a_proj_with_mqa
            ("self_attention.linear_proj", "self_attn.o_proj"),
            ("input_layernorm", "input_layernorm"),
            ("pre_mlp_layernorm", "post_attention_layernorm"),
            ("mlp.linear_fc1", "mlp.gate_up_proj"),
            ("mlp.linear_fc2", "mlp.down_proj"),
            ("decoder.final_layernorm", "model.norm"),
            ("output_layer", "lm_head"),
            ("self_attention.linear_qb", "self_attn.q_b_proj"),
            ("self_attention.linear_kvb", "self_attn.kv_b_proj"),
            ("mlp.router.expert_bias", "mlp.gate.e_score_correction_bias"),
            ('mlp.router', 'mlp.gate'),  # weight, expert_bias
            ("mlp.shared_experts.linear_fc1", "mlp.shared_experts.gate_up_proj"),
            ("mlp.shared_experts.linear_fc2", "mlp.shared_experts.down_proj"),
            ("mlp.experts.weight1", "mlp.experts.w13_weight"),
            ("mlp.experts.weight2", "mlp.experts.w2_weight"),
            ("self_attention.q_layernorm", "self_attn.q_a_layernorm"),
            ("self_attention.k_layernorm", "self_attn.kv_a_layernorm"),
        ]

    def get_weight_buffer_meta(self, model, valid_names=None):
        weight_buffer_meta = {}
        for name, param in sorted(model.named_parameters()):
            if valid_names and name not in valid_names:
                continue
            if 'kv_a_proj_with_mqa' in name:
                # 将kv_a_proj_with_mqa和q_a_proj的tensor拼接，并用qkv_proj和拼接的结果替换掉原来kv_a_proj_with_mqa的对应部分
                q_param = dict(model.named_parameters()).get(name.replace('kv_a_proj_with_mqa', 'q_a_proj' if self.model_config.q_lora_rank else "q_proj"))
                qkv_param_shape = torch.cat([q_param, param], dim=0).shape
                qkv_name = name.replace('kv_a_proj_with_mqa', 'qkv_proj')
                weight_buffer_meta[qkv_name] = {'shape': qkv_param_shape, 'dtype': param.dtype}
            elif 'q_a_proj' in name or 'q_proj' in name:
                continue
            else:
                weight_buffer_meta[name] = {'shape': param.shape, 'dtype': param.dtype}
        return weight_buffer_meta

    def convert_weight_name_meta(self, weight_names):
        if not self.meta_info:
            return weight_names

        weight_names_meta = list()
        for elements in weight_names:
            if isinstance(elements, list):
                tmp_weight_names_meta = self.convert_weight_name_meta(elements)
                weight_names_meta.append(tmp_weight_names_meta)
            else:
                converted = False
                if not converted and 'replace' in self.meta_info:
                    for key, value in self.meta_info['replace'].items():
                        if key in elements:
                            qkv_name = elements.replace(key, value)
                            weight_names_meta.append(qkv_name)
                            converted = True
                            break

                if not converted and 'delete' in self.meta_info:
                    for key in self.meta_info['delete']:
                        if key in elements:
                            converted = True
                            break

                if not converted:
                    weight_names_meta.append(elements)

        return weight_names_meta



class QwenMVWeightAdaptor(MegatronVLLMWeightAdaptor):
    """
    Megatron-vLLM WeightAdaptor for Qwen model architectures.
    """
    def __init__(self, model_config):
        super(QwenMVWeightAdaptor, self).__init__(model_config)


class Qwen2_5_VLWeightAdaptor(MegatronVLLMWeightAdaptor):
    def __init__(self, model_config):
        super(Qwen2_5_VLWeightAdaptor, self).__init__(model_config)
        self.params_mapping = [
            ("text_decoder.embedding.word_embeddings", "language_model.model.embed_tokens"),
            ("text_decoder.decoder.layers.{layer_num}.self_attention.linear_qkv", "language_model.model.layers.{layer_num}.self_attn.qkv_proj"),
            ("text_decoder.decoder.layers.{layer_num}.self_attention.linear_proj", "language_model.model.layers.{layer_num}.self_attn.o_proj"),
            ("text_decoder.decoder.layers.{layer_num}.input_layernorm", "language_model.model.layers.{layer_num}.input_layernorm"),
            ("text_decoder.decoder.layers.{layer_num}.pre_mlp_layernorm", "language_model.model.layers.{layer_num}.post_attention_layernorm"),
            ("text_decoder.decoder.layers.{layer_num}.mlp.linear_fc1", "language_model.model.layers.{layer_num}.mlp.gate_up_proj"),
            ("text_decoder.decoder.layers.{layer_num}.mlp.linear_fc2", "language_model.model.layers.{layer_num}.mlp.down_proj"),
            ("text_decoder.decoder.final_layernorm", "language_model.model.norm"),
            ("text_decoder.output_layer", "language_model.lm_head"),
            ("image_encoder.encoder.patch_embed.proj", "visual.patch_embed.proj"),
            ("image_encoder.encoder.blocks.layers.{layer_num}.self_attention.linear_qkv", "visual.blocks.{layer_num}.attn.qkv"),
            ("image_encoder.encoder.blocks.layers.{layer_num}.self_attention.linear_proj", "visual.blocks.{layer_num}.attn.proj"),
            ("image_encoder.encoder.blocks.layers.{layer_num}.input_layernorm", "visual.blocks.{layer_num}.norm1"),
            ("image_encoder.encoder.blocks.layers.{layer_num}.pre_mlp_layernorm", "visual.blocks.{layer_num}.norm2"),
            ("image_encoder.encoder.blocks.layers.{layer_num}.mlp.linear_fc1", "visual.blocks.{layer_num}.mlp.gate_up_proj"),
            ("image_encoder.encoder.blocks.layers.{layer_num}.mlp.linear_fc2", "visual.blocks.{layer_num}.mlp.down_proj"),
            ("image_encoder.projector.layernorm", "visual.merger.ln_q"),
            ("image_encoder.projector.encoder.linear_fc1", "visual.merger.mlp.0"),
            ("image_encoder.projector.encoder.linear_fc2", "visual.merger.mlp.2"),
        ]

    def replace_name_i2t(self, inference_name):
        weight_suffix = ""
        if inference_name.endswith(".weight"):
            weight_suffix = ".weight"
            base_name = inference_name[:-7]
        elif inference_name.endswith(".bias"):
            weight_suffix = ".bias"
            base_name = inference_name[:-5]
        else:
            base_name = inference_name

        for megatron_pattern, vllm_pattern in self.params_mapping:
            vllm_regex = vllm_pattern.replace("{layer_num}", r"(\d+)")
            match = re.match(f"^{vllm_regex}(.*)$", base_name)
            if match:
                groups = match.groups()
                layer_nums = [g for g in groups[:-1] if g is not None and g.isdigit()]
                extra_suffix = groups[-1] if groups and groups[-1] is not None else ""

                megatron_result = megatron_pattern
                for layer_num in layer_nums:
                    megatron_result = megatron_result.replace("{layer_num}", layer_num, 1)

                return megatron_result + extra_suffix + weight_suffix

        return inference_name

    @staticmethod
    def _convert_global_to_local_index(name, layer_keyword, pp_layers):
        """
        Convert global layer index to local layer index for a given layer type.

        Args:
            name: Weight name containing layer information
            layer_keyword: Layer type keyword ('blocks' for visual, 'layers' for language model)
            pp_layers: List of layer counts per pipeline parallel rank

        Returns:
            Updated weight name with local layer index
        """
        split_name = name.split('.')

        # Find the position of layer keyword
        layer_keyword_idx = -1
        for i, name_part in enumerate(split_name):
            if name_part == layer_keyword:
                layer_keyword_idx = i
                break

        if layer_keyword_idx == -1:
            return name

        layer_num_idx = layer_keyword_idx + 1
        if len(split_name) < layer_num_idx + 1 or not split_name[layer_num_idx].isdigit():
            raise ValueError(f'Invalid {layer_keyword} name: {split_name}')

        global_idx = int(split_name[layer_num_idx])

        # Calculate local index
        cumulative_layers = 0
        for layers_in_pp_rank in pp_layers:
            if layers_in_pp_rank == 0:
                continue
            if cumulative_layers <= global_idx < cumulative_layers + layers_in_pp_rank:
                local_index = global_idx - cumulative_layers
                split_name[layer_num_idx] = str(local_index)
                return '.'.join(split_name)
            cumulative_layers += layers_in_pp_rank

        raise ValueError(f'Could not map {layer_keyword} {global_idx} to a local index with distribution {pp_layers}')

    @staticmethod
    def global2local_layer(name, num_layer_list, vpp_rank=0, global2local_map=None):
        """
        Transform layer names from global space to local space for Qwen2VL models.
        Supports both visual blocks and language model layers.

        Args:
            name: Weight name to transform
            num_layer_list: [img_pp_layers, llm_pp_layers] distribution

        Returns:
            Transformed weight name with local layer indices
        """
        if vpp_rank > 0:
            raise NotImplementedError("VPP is not supported in multimodal models.")

        img_pp_layers, llm_pp_layers = num_layer_list

        if name.startswith('visual') and 'blocks' in name:
            return Qwen2_5_VLWeightAdaptor._convert_global_to_local_index(name, 'blocks', img_pp_layers)
        elif name.startswith('language_model') and 'layers' in name:
            return Qwen2_5_VLWeightAdaptor._convert_global_to_local_index(name, 'layers', llm_pp_layers)

        return name

    @staticmethod
    def _categorize_weights(vllm_names):
        """
        Categorize weight names by their types for easier processing.

        Args:
            vllm_names: List of vLLM weight names

        Returns:
            Dictionary containing categorized weight names
        """
        visual_weights = []
        lang_weights = []
        visual_pre_layer_weights = []
        visual_post_layer_weights = []
        lang_pre_layer_weights = []
        lang_post_layer_weights = []

        for name in vllm_names:
            if name.startswith('visual'):
                if 'blocks' not in name:
                    if 'patch_embed' in name:
                        visual_pre_layer_weights.append(name)
                    elif 'merger' in name:
                        visual_post_layer_weights.append(name)
                else:
                    visual_weights.append(name)
            elif name.startswith('language_model'):
                if 'layers' not in name:
                    if 'embed_tokens' in name:
                        lang_pre_layer_weights.append(name)
                    else:
                        lang_post_layer_weights.append(name)
                else:
                    lang_weights.append(name)

        return {
            'visual_weights': visual_weights,
            'lang_weights': lang_weights,
            'visual_pre_layer_weights': visual_pre_layer_weights,
            'visual_post_layer_weights': visual_post_layer_weights,
            'lang_pre_layer_weights': lang_pre_layer_weights,
            'lang_post_layer_weights': lang_post_layer_weights
        }

    @staticmethod
    def _calculate_layer_ranges(pp_layers):
        """
        Calculate layer ranges for each pipeline parallel stage.

        Args:
            pp_layers: List of layer counts per pipeline parallel rank

        Returns:
            List of (start_layer, end_layer) tuples for each rank
        """
        layer_ranges = []
        start_layer = 0
        for layers_in_pp_rank in pp_layers:
            if layers_in_pp_rank > 0:
                layer_ranges.append((start_layer, start_layer + layers_in_pp_rank - 1))
                start_layer += layers_in_pp_rank
            else:
                layer_ranges.append((-1, -1))
        return layer_ranges

    @staticmethod
    def _find_last_rank(pp_layers):
        """
        Find the last pipeline parallel rank that has non-zero layers.
        """
        for i in range(len(pp_layers) - 1, -1, -1):
            if pp_layers[i] > 0:
                return i
        return -1

    @staticmethod
    def _find_first_rank(pp_layers):
        """
        Find the first pipeline parallel rank that has non-zero layers.
        """
        for i, layers in enumerate(pp_layers):
            if layers > 0:
                return i
        return -1

    @staticmethod
    def _assign_layer_weights(weight_names_per_pp, weights, layer_ranges, layer_keyword):
        """
        Assign layer weights to their corresponding pipeline parallel stages.
        """
        for pp_rank, (start_layer, end_layer) in enumerate(layer_ranges):
            if start_layer >= 0 and end_layer >= 0:
                for name in weights:
                    match = re.match(rf'.*\.{layer_keyword}\.(\d+)', name)
                    if match:
                        layer_num = int(match.group(1))
                        if start_layer <= layer_num <= end_layer:
                            weight_names_per_pp[pp_rank].append(name)

    @staticmethod
    def get_weight_names_per_pp(layer_list, vllm_names, layers_num=None, vpp_size=0, noop_layers=None):
        """
        Get weight names for each pipeline parallel stage optimized for Qwen2VL models.

        Args:
            layer_list: [img_pp_layers, llm_pp_layers] distribution
            vllm_names: List of vLLM weight names

        Returns:
            List of weight names for each pipeline parallel rank
        """
        if not layers_num:
            if vpp_size > 0:
                ValueError(f"layers_num is required with vpp_size = {vpp_size}")
            layers_num = [sum(sub_layer_list) for sub_layer_list in layer_list]
        if vpp_size > 1:
            raise NotImplementedError("VPP is not supported in multimodal models.")

        img_pp_layers, llm_pp_layers = layer_list
        pp_size = len(img_pp_layers)

        weight_categories = Qwen2_5_VLWeightAdaptor._categorize_weights(vllm_names)

        img_blocks_range = Qwen2_5_VLWeightAdaptor._calculate_layer_ranges(img_pp_layers)
        llm_layers_range = Qwen2_5_VLWeightAdaptor._calculate_layer_ranges(llm_pp_layers)

        weight_names_per_pp = [[] for _ in range(pp_size)]

        last_img_rank = Qwen2_5_VLWeightAdaptor._find_last_rank(img_pp_layers)
        first_llm_rank = Qwen2_5_VLWeightAdaptor._find_first_rank(llm_pp_layers)
        last_llm_rank = Qwen2_5_VLWeightAdaptor._find_last_rank(llm_pp_layers)

        # Process visual weights
        for pp_rank in range(pp_size):
            start_layer, end_layer = img_blocks_range[pp_rank]

            if start_layer == 0 and end_layer >= 0:
                weight_names_per_pp[pp_rank].extend(weight_categories['visual_pre_layer_weights'])

            if pp_rank == last_img_rank:
                weight_names_per_pp[pp_rank].extend(weight_categories['visual_post_layer_weights'])

        # Assign visual layer weights
        Qwen2_5_VLWeightAdaptor._assign_layer_weights(
            weight_names_per_pp, weight_categories['visual_weights'], img_blocks_range, 'blocks'
        )

        # Process language model weights
        for pp_rank in range(pp_size):
            if pp_rank == first_llm_rank:
                weight_names_per_pp[pp_rank].extend(weight_categories['lang_pre_layer_weights'])

            if pp_rank == last_llm_rank:
                weight_names_per_pp[pp_rank].extend(weight_categories['lang_post_layer_weights'])

        # Assign language model layer weights
        Qwen2_5_VLWeightAdaptor._assign_layer_weights(
            weight_names_per_pp, weight_categories['lang_weights'], llm_layers_range, 'layers'
        )

        # Align vpp format, only support vpp=1 in the current version
        for i, weight_name_list in enumerate(weight_names_per_pp):
            weight_names_per_pp[i] = [weight_name_list]
        return weight_names_per_pp


WEIGHT_ADAPTOR_REGISTRY = {
    "Qwen2ForCausalLM": QwenMVWeightAdaptor,
    "DeepseekV3ForCausalLM": DeepSeekMVWeightAdaptor,
    "DeepseekV2ForCausalLM": DeepSeekMVWeightAdaptor,
    "CustomDeepseekV3ForCausalLM": DeepSeekMVWeightAdaptor,
    "Qwen2_5_VLForConditionalGeneration": Qwen2_5_VLWeightAdaptor,
}


def get_weight_adaptor(arch: str):
    if arch in WEIGHT_ADAPTOR_REGISTRY:
        return WEIGHT_ADAPTOR_REGISTRY[arch]
    raise ValueError(f"Model architectures {arch} are not supported for now.")