# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.
import random
from abc import ABC
from typing import Callable, Dict, List
from functools import partial
import itertools

import torch
import torch.nn.functional as F

from mindspeed_rl.models.loss.base_loss_func import BaseLossFunc
from mindspeed_rl.models.loss.loss_func_factory import LossFuncFactory
from mindspeed_rl.utils.utils import (
    append_to_dict, generate_mask, generate_position_ids, get_tune_attention_mask, is_multimodal
)
from mindspeed_rl.utils.seqlen_balancing import rearrange_micro_batches, get_reverse_idx
from mindspeed_rl.utils.remove_padding import preprocess_packed_seqs, postprocess_packed_seqs
from mindspeed_rl.utils.compute import get_parallel_state
from mindspeed_rl.utils.utils import get_batch_on_this_cp_rank, is_multimodal


class BaseTrainingEngine(ABC):
    """
    Initialize the base trainning engine.

    Args:
        model: The network model to be trained.
        optimizer: The optimizer for updating model parameters (e.g., Adam).
        opt_param_scheduler: The scheduler for optimizer parameters (e.g., learning rate scheduler).
        beta: float = 0 The weight coefficient for KL divergence (used in algorithms like PPO).
        mini_batch_size_per_dp: int = 1  The size of the mini-batch for each data parallel stage.
        epochs: int = 1 The number of training epochs.
        shuffle_mini_batch: bool = False Whether to shuffle the mini-batch data at each epoch.
        stage(str): str = None The training stage identifier (e.g., ray_grpo).
        kl_ctrl: float = 0.1 Adaptive kl ctrl.
        clip_ratio: float = 0.1 The clipping ratio threshold for PPO (limits the policy update range).
        role: str The role of actor in the RLHF frameworker.
        micro_batch_size: int = 1 Micro batch size for actor rollout.
        forward_backward_func: Callable = None The forward-backward function for distributed training.
        **kwargs: Additional keyword arguments.
    """

    def __init__(
            self,
            model,
            optimizer=None,
            opt_param_scheduler=None,
            beta: float = 0,
            mini_batch_size_per_dp: int = 1,
            epochs: int = 1,
            shuffle_mini_batch: bool = False,
            stage: str = None,
            kl_ctrl: float = 0.0,
            clip_ratio: float = 0.1,
            temperature: float = 1.0,
            role: str = None,
            micro_batch_size: int = 1,
            use_dynamic_bsz: bool = False,
            max_packing_token_size: bool = 4096,
            use_remove_padding: bool = False,
            set_actual_seq_len: Callable = None,
            forward_backward_func: Callable = None,
            entropy_coeff: float = 0.0,
            context_parallel_algo: str = "ulysses_cp_algo",
            context_parallel_size: int = 1,
            kl_penalty: str = "low_var_kl",
            **kwargs):
        self.forward_backward_func = forward_backward_func
        self.micro_batch_size = micro_batch_size
        self.use_dynamic_bsz = use_dynamic_bsz
        self.max_packing_token_size = max_packing_token_size
        self.use_remove_padding = use_remove_padding
        self.set_actual_seq_len = set_actual_seq_len
        self.model = model
        self.optimizer = optimizer
        self.opt_param_scheduler = opt_param_scheduler
        self.beta = beta
        self.mini_batch_size_per_dp = mini_batch_size_per_dp
        self.epochs = epochs
        self.shuffle_mini_batch = shuffle_mini_batch
        self.stage = stage
        self.role = role
        self.kl_ctrl = kl_ctrl
        self.kl_penalty = kl_penalty
        self.clip_ratio = clip_ratio
        self.entropy_coeff = entropy_coeff
        self.context_parallel_algo = context_parallel_algo
        self.context_parallel_size = context_parallel_size
        self.temperature = temperature
        self.loss_func: BaseLossFunc = LossFuncFactory.get_instance(self.stage, self.role)
        self.kwargs = kwargs

    @staticmethod
    def _split_batches(batch: Dict, batch_size: int, shuffle_mini_batch: bool, dim: int = 0, keep_list: bool = False) -> List[Dict]:
        batches = []
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                split_values = torch.split(value, batch_size, dim)
            elif isinstance(value, List): # 多模态场景
                num_batches = (len(value) + batch_size - 1) // batch_size
                if keep_list: # 保持值为list类型
                    split_values = [value[i * batch_size: (i + 1) * batch_size] for i in range(num_batches)]
                else:
                    split_values = [torch.concat(value[i * batch_size: (i + 1) * batch_size]) for i in range(num_batches)]
            for index, split_value in enumerate(split_values):
                if index >= len(batches):
                    batches.append({})
                batches[index][key] = split_value

        if shuffle_mini_batch:
            random.shuffle(batches)
        return batches

    @staticmethod
    def _split_batches_with_dynamic_bsz(batch: Dict, max_packing_token: int) -> (List[Dict], List[List[int]]):
        seq_len_list = []
        for prompt_len, response_len in zip(batch['prompt_length'], batch['response_length']):
            seq_len_list.append(prompt_len.item() + response_len.item())
        partitions = rearrange_micro_batches(seq_len_list, max_packing_token)
        batches = []
        for key, tensors in batch.items():
            for batch_idx, partition in enumerate(partitions):
                if batch_idx >= len(batches):
                    batches.append({})
                batches[batch_idx][key] = tensors[partition]
        return batches, partitions

    def _forward_backward_batch(self, batch: Dict[str, torch.Tensor], forward_only: bool = False):
        if self.use_dynamic_bsz:
            batches, indices = self._split_batches_with_dynamic_bsz(batch, self.max_packing_token_size)
        else:
            batches = self._split_batches(batch, batch_size=self.micro_batch_size,
                                          shuffle_mini_batch=self.shuffle_mini_batch)
        n_micro_batch = len(batches)
        batch_size = batch['input_ids'].shape[0]
        seq_len = batches[0]['input_ids'].shape[1]
        data_iter = iter(batches)
        if len(self.model) > 1:
            data_iter = [iter(batches) for _ in self.model]
        self.loss_func.add_loss_meta_info(self.get_loss_meta_func())
        post_process = get_parallel_state().get_pipeline_model_parallel_world_size() == 1 or get_parallel_state().is_pipeline_last_stage()

        def forward_step(batch_iter, model):
            cp_size = get_parallel_state().get_context_parallel_world_size()
            if is_multimodal():
                process_batch, seqlens_in_batch, cu_seqlens_padded = self._get_mulitmodal_forward_batch_info(batch_iter)
                output = model(**process_batch)
                if post_process:
                    output = postprocess_packed_seqs(output=output['logits'],
                                                     seqlens_in_batch=seqlens_in_batch,
                                                     cu_seqlens_padded=cu_seqlens_padded,
                                                     seq_len=seq_len,
                                                     return_tensor=True)
                    output.div_(self.temperature)
            elif self.use_remove_padding:
                input_ids, position_ids, process_batch, seqlens_in_batch, cu_seqlens_padded = self._get_forward_batch_info(batch_iter)
                self.set_actual_seq_len(cu_seqlens_padded.tolist())
                output_orig = model(input_ids=input_ids, attention_mask=None, position_ids=position_ids)
                if not post_process:
                    output = output_orig
                    output.div_(self.temperature)
                else:
                    output = postprocess_packed_seqs(output=output_orig,
                                                     seqlens_in_batch=seqlens_in_batch,
                                                     cu_seqlens_padded=cu_seqlens_padded,
                                                     seq_len=seq_len)
                    for item in output:
                        item.div_(self.temperature)
            else:
                input_ids, attention_mask, position_ids, process_batch = self._get_forward_batch_info(batch_iter)
                output = model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids)
                if post_process:
                    if cp_size > 1:
                        output_list = [torch.empty_like(output) for _ in range(cp_size)]
                        torch.distributed.all_gather(output_list, output.detach(), group=get_parallel_state().get_context_parallel_group())
                        output_list[get_parallel_state().get_context_parallel_rank()] = output
                        output = torch.cat(output_list, dim=1)
                output.div_(self.temperature)
            return output, partial(self.loss_func.compute_loss,
                                   batch=process_batch,
                                   forward_only=forward_only,
                                   use_dynamic_bsz=self.use_dynamic_bsz,
                                   actual_micro_batch_size=batch_size / n_micro_batch)

        # batch should be a list of batches inside micro-batches
        losses_reduced = self.forward_backward_func(
            forward_step_func=forward_step,
            data_iterator=data_iter,
            model=self.model,
            num_microbatches=n_micro_batch,
            seq_length=self.micro_batch_size * seq_len if self.use_remove_padding else seq_len,
            micro_batch_size=1 if self.use_remove_padding else self.micro_batch_size,
            forward_only=forward_only,
            collect_non_loss_data=forward_only,
        )

        # Reverse the batch index to be the same outside
        if self.use_dynamic_bsz and forward_only and post_process:
            losses_reduced_list = torch.cat(losses_reduced, dim=0)
            indices = list(itertools.chain.from_iterable(indices))
            revert_indices = get_reverse_idx(indices)
            losses_reduced = [losses_reduced_list[[idx, ]] for idx in revert_indices]

        return losses_reduced

    def get_loss_meta_func(self) -> Dict:
        """
        获取具体的loss计算超参
        :return: loss计算超参。
        """
        return {}

    def _get_forward_batch_info(self, batch_iter):
        batch = next(batch_iter)
        input_ids = batch['input_ids']
        attention_mask_1d = generate_mask(input_ids, batch['prompt_length'] + batch['response_length']).to(
            input_ids.device)
        cp_size = get_parallel_state().get_context_parallel_world_size()
        if self.use_remove_padding:
            tp_size = get_parallel_state().get_tensor_model_parallel_world_size()
            input_ids, position_ids, seqlens_in_batch, cu_seqlens_padded = preprocess_packed_seqs(
                input_ids=input_ids, attention_mask_1d=attention_mask_1d, tp_size=tp_size)
            return input_ids, position_ids, batch, seqlens_in_batch, cu_seqlens_padded
        else:
            position_ids = torch.tensor(generate_position_ids(input_ids)).to(input_ids.device)
            attention_mask = get_tune_attention_mask(attention_mask_1d)
            if cp_size > 1:
                batch_for_cp = {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'position_ids': position_ids
                }
                batch_cp = get_batch_on_this_cp_rank(self.context_parallel_algo, self.context_parallel_size, batch_for_cp)
                input_ids = batch_cp['input_ids']
                attention_mask = None
                position_ids = batch_cp['position_ids']

        return input_ids, attention_mask, position_ids, batch

    def _get_mulitmodal_forward_batch_info(self, batch_iter):
        batch = next(batch_iter)
        input_ids = batch['input_ids']
        batch_size = input_ids.size(0)

        response_attention_mask = generate_mask(batch['responses'], batch['response_length']).to(input_ids.device)
        attention_mask = torch.cat((batch['attention_mask'], response_attention_mask), dim=-1).bool()

        delta_position_id = torch.tensor(generate_position_ids(batch['responses'])).to(input_ids.device) + 1
        delta_position_id = delta_position_id.unsqueeze(1).expand(batch_size, 1, -1)

        if batch['position_ids'].dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.expand(batch_size, 3, -1)
        else:
            batch['position_ids'] = batch['position_ids'].view(batch_size, 1, -1)
        response_position_ids = batch['position_ids'][..., -1:] + delta_position_id
        position_ids = torch.cat([batch['position_ids'], response_position_ids], dim=-1)

        input_ids_rmpad_list = []
        position_ids_rmpad_list = []
        for i in range(batch_size):
            input_ids_rmpad_list.append(input_ids[i].masked_select(attention_mask[i]))
            masked_position_ids = position_ids[i].masked_select(attention_mask[i].expand_as(position_ids[i]))
            position_ids_rmpad_list.append(masked_position_ids.view(delta_position_id.size(1), -1).t())
        input_ids_rmpad = torch.cat(input_ids_rmpad_list).unsqueeze(0)
        position_ids_rmpad = torch.cat(position_ids_rmpad_list).t().unsqueeze(1)

        seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
        cu_seqlens_padded = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))

        batch['input_ids'] = input_ids_rmpad
        batch['position_ids'] = position_ids_rmpad
        batch['attention_mask'] = None
        return batch, seqlens_in_batch, cu_seqlens_padded

    def post_process_forward_backward_output(self, output: [torch.Tensor],
                                             batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        模型前反向计算结果后处理
        :param output: 模型前反向计算结果
        :param batch: 参与计算的batch数据
        :return: 模型前向计算结果。
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def forward(self, data: Dict) -> torch.Tensor:
        """
        模型前向计算
        :param data: 前向计算数据
        :return: 模型前向计算结果。
        """
        for k, v in data.items():
            if isinstance(v, List):
                data[k] = [t.to(next(self.model[0].parameters()).device) for t in v]
            else:
                data[k] = v.to(next(self.model[0].parameters()).device)
        for model_module in self.model:
            model_module.eval()
        with torch.no_grad():
            output = self._forward_backward_batch(data, forward_only=True)
            return self.post_process_forward_backward_output(output=output, batch=data)

    def update(self, data: Dict, kl_ctrl=None) -> Dict:
        """
        模型反向更新
        :param data: 反向更新数据
        :param kl_ctrl：KL散度计算controller
        :return: 模型反向计算结果。
        """
        self.kl_ctrl = kl_ctrl
        metrics = {}
        grad_norm_list = []
        for k, v in data.items():
            if v is not None:
                if isinstance(v, List):
                    data[k] = [t.to(next(self.model[0].parameters()).device) for t in v]
                else:
                    data[k] = v.to(next(self.model[0].parameters()).device)
        mini_batches = self._split_batches(data, batch_size=self.mini_batch_size_per_dp,
                                           shuffle_mini_batch=self.shuffle_mini_batch, dim=0, keep_list=True)
        for model_module in self.model:
            model_module.train()
        for _ in range(self.epochs):
            for mini_batch in mini_batches:
                for model_chunk in self.model:
                    model_chunk.zero_grad_buffer()
                self.optimizer.zero_grad()
                metric_micro_batch = self._forward_backward_batch(mini_batch)
                update_successful, grad_norm, num_zeros_in_grad = self.optimizer.step()

                if update_successful:
                    data_parallel_world_size = get_parallel_state().get_data_parallel_world_size()
                    increment = self.mini_batch_size_per_dp * data_parallel_world_size
                    self.opt_param_scheduler.step(increment=increment)
                grad_norm_list.append(grad_norm)

                for metric in metric_micro_batch:
                    append_to_dict(metrics, metric)  # append the metric from this micro-batch to global metrics.

        grad_norm = sum(grad_norm_list) / len(grad_norm_list)
        metrics["grad_norm"] = grad_norm_list
        return metrics
