from abc import ABC, abstractmethod
import re
import torch


class BaseWeightAdaptor(ABC):
    def __init__(self):
        """
        Base class for weight adaptors.
        A weight adaptor provide a set of tools to transfer from training weight to inference weight.
        Currently, we support MegatronVLLMWeightAdaptor only.
        Args:
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
        super(MegatronVLLMWeightAdaptor, self).__init__()
        self.model_config = model_config
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


    def get_weight_buffer_meta(self, model, valid_names=None):
        weight_buffer_meta = {}
        for name, param in sorted(model.named_parameters()):
            if valid_names and name not in valid_names:
                continue
            else:
                weight_buffer_meta[name] = {'shape': param.shape, 'dtype': param.dtype}
        return weight_buffer_meta

    @staticmethod
    def global2local_layer(name, num_layer_list):
        """
        Transform the model name in each model_chunk in global space to local space
        """
        layer_name = 'layers'

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
            global_idx = int(split_name[layer_num_idx])
            for layers_in_pp in num_layer_list:
                global_idx -= layers_in_pp
                if global_idx < 0:
                    local_index = global_idx + layers_in_pp
                    break

            split_name[layer_num_idx] = str(local_index)
            name = '.'.join(split_name)  # weight name in inference_tp_model

        return name

    @staticmethod
    def get_weight_names_per_pp(layer_list, vllm_names):

        end_layer = sum(layer_list) - 1

        def get_weight_names_in_range(layer_range, names: list, layer_name='layers') -> list:
            """
            Extract weights in a given range and also include the weights before and after the range as needed.
            """
            start, end = layer_range
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
                    if start <= layer_num <= end:
                        names_in_range.append(name)

            # add names after decode layers
            if end == last_layer_index:
                for name in reversed(names):
                    if layer_name not in name:
                        names_in_range.append(name)
                    else:
                        break
            return names_in_range

        pp_layers_range = []
        start_layer = 0
        for layers_in_pp_rank in layer_list:
            pp_layers_range.append((start_layer, start_layer + layers_in_pp_rank - 1))
            start_layer += layers_in_pp_rank
        weight_names_per_pp = [get_weight_names_in_range(layer_range, vllm_names) for layer_range in pp_layers_range]
        return weight_names_per_pp


class DeepSeekMVWeightAdaptor(MegatronVLLMWeightAdaptor):
    """
    Megatron-vLLM WeightAdaptor for DeepSeek model architectures.
    """
    def __init__(self, model_config):
        super(DeepSeekMVWeightAdaptor, self).__init__(model_config)
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
                q_param = dict(model.named_parameters()).get(name.replace('kv_a_proj_with_mqa', 'q_a_proj'))
                qkv_param_shape = torch.cat([q_param, param], dim=0).shape
                qkv_name = name.replace('kv_a_proj_with_mqa', 'qkv_proj')
                weight_buffer_meta[qkv_name] = {'shape': qkv_param_shape, 'dtype': param.dtype}
            elif 'q_a_proj' in name:
                continue
            else:
                weight_buffer_meta[name] = {'shape': param.shape, 'dtype': param.dtype}
        return weight_buffer_meta


class QwenMVWeightAdaptor(MegatronVLLMWeightAdaptor):
    """
    Megatron-vLLM WeightAdaptor for Qwen model architectures.
    """
    def __init__(self, model_config):
        super(QwenMVWeightAdaptor, self).__init__(model_config)


WEIGHT_ADAPTOR_REGISTRY = {
    "Qwen2ForCausalLM": QwenMVWeightAdaptor,
    "DeepseekV3ForCausalLM": DeepSeekMVWeightAdaptor,
    "DeepseekV2ForCausalLM": DeepSeekMVWeightAdaptor,
}


def get_weight_adaptor(arch: str):
    if arch in WEIGHT_ADAPTOR_REGISTRY:
        return WEIGHT_ADAPTOR_REGISTRY[arch]
    raise ValueError(f"Model architectures {arch} are not supported for now.")