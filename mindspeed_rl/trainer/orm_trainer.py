# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.
from functools import partial
from abc import ABC

import torch
import torch.distributed as dist

from mindspeed_rl.utils import Loggers
from mindspeed_rl.utils import get_tune_attention_mask
from .utils.training import (get_finetune_data_on_this_tp_rank,
                             broadcast_data, average_losses_across_data_parallel_group)


logger = Loggers("ORMTrainer")


class ORMTrainer(ABC):
    """
    A trainer class for Outcome Reward Model (ORM).

    This class provides methods for model initialize, computing losses and metrics, and training.
    """

    def __init__(
            self,
            args,
            model,
            optimizer,
            train_data_iterator,
            valid_data_iterator,
            test_data_iterator_list,
            scheduler,
            process_non_loss_data_func=None,
            model_config=None,
            **kwargs
    ) -> None:
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.train_data_iterator = train_data_iterator
        self.valid_data_iterator = valid_data_iterator
        self.test_data_iterator_list = test_data_iterator_list
        self.scheduler = scheduler
        self.process_non_loss_data_func = process_non_loss_data_func
        self.model_config = model_config
        # Similar to dpo, set actual_micro_batch_size to change the recv/send shape when using PP
        self.args = args
        self.args.actual_micro_batch_size = self.args.micro_batch_size * 2
        self.train_args = (self.forward_step, self.model, self.optimizer, self.scheduler, self.train_data_iterator,
                           self.valid_data_iterator, self.process_non_loss_data_func, self.model_config)
        self.train_func = kwargs.get('train_func', None)
        self.parallel_state = kwargs.get('parallel_state', None)
        self.save_checkpoint_func = kwargs.get('save_checkpoint_func', None)
        self.evaluate_fun = kwargs.get('evaluate_fun', None)
        self.get_batch_on_this_cp_rank = kwargs.get('batch_cp_func', None)

    def get_batch(self, data_iterator):
        """Generate a batch."""

        if (not self.parallel_state.is_pipeline_first_stage()) and (not self.parallel_state.is_pipeline_last_stage()):
            tokens, attention_mask = get_finetune_data_on_this_tp_rank(data_iterator, self.parallel_state, self.args)
            if self.args.variable_seq_lengths and self.args.pipeline_model_parallel_size > 2:
                return tokens, None, None, attention_mask, None
            else:
                batch = {'attention_mask': attention_mask}
                batch = self.get_batch_on_this_cp_rank(batch)
                return None, None, None, batch['attention_mask'], None
        # Items and their type.
        keys = ['input_ids', 'attention_mask', 'labels']
        data_type = torch.int64

        # Broadcast data.
        data_b = broadcast_data(keys, next(data_iterator), data_type, self.parallel_state)

        # Unpack
        labels = data_b.get('labels').long()
        tokens = data_b.get('input_ids').long()
        attention_mask_1d = data_b.get('attention_mask').long()
        loss_mask = attention_mask_1d

        # ORM uses only the last token to compute loss, so also keeps only the last 1 in loss mask,
        # for example, [1, 1, 1, 1, 1, 1, 0, 0] -> [0, 0, 0, 0, 0, 1, 0, 0]
        last_token_position = torch.sum(loss_mask, dim=1) - 1
        loss_mask_new = torch.zeros_like(loss_mask)
        fill = torch.ones(loss_mask.size(0)).to(loss_mask.dtype).to(loss_mask.device)
        loss_mask_new.scatter_(1, last_token_position.unsqueeze(1), fill.unsqueeze(1))
        loss_mask = loss_mask_new

        attention_mask = get_tune_attention_mask(attention_mask_1d,
                                                 tokenizer_padding_side=self.args.tokenizer_padding_side,
                                                 reset_attention_mask=self.args.reset_attention_mask
                                                 )

        batch = {
            'tokens': tokens,
            'labels': labels,
            'loss_mask': loss_mask,
            'attention_mask': attention_mask,
            'position_ids': None
        }
        batch = self.get_batch_on_this_cp_rank(batch)
        return batch.values()

    def loss_func(self, input_ids: torch.Tensor, loss_mask: torch.Tensor, output_tensor: torch.Tensor):
        """RM Loss function.
        """
        batch_size = input_ids.size(0) // 2
        chosen_masks, rejected_masks = torch.split(loss_mask, batch_size, dim=0)
        chosen_rewards, rejected_rewards = torch.split(output_tensor, batch_size, dim=0)
        chosen_rewards, rejected_rewards = chosen_rewards.squeeze(-1), rejected_rewards.squeeze(-1)
        chosen_scores = (chosen_masks * chosen_rewards).sum(dim=1)
        rejected_scores = (rejected_masks * rejected_rewards).sum(dim=1)
        if self.args.context_parallel_size > 1:
            dist.all_reduce(chosen_scores, group=self.parallel_state.get_context_parallel_group())
            dist.all_reduce(rejected_scores, group=self.parallel_state.get_context_parallel_group())

        loss = -torch.log(torch.sigmoid(chosen_scores.float() - rejected_scores.float())).mean()
        with torch.no_grad():
            acc = (chosen_scores > rejected_scores).sum() / len(chosen_scores)
        averaged_loss = average_losses_across_data_parallel_group([loss], parallel_state=self.parallel_state)
        return loss * self.args.context_parallel_size, {'lm loss': averaged_loss[0], 'acc': acc}

    def forward_step(self, data_iterator, model):
        """RM Forward training step.

        Args:
            data_iterator : Input data iterator
            model (GPTRewardModel): The GPT Reward Model
        """
        # Get the batch.
        tokens, labels, loss_mask, attention_mask, position_ids = self.get_batch(
            data_iterator)

        output_tensor = model(tokens, position_ids, attention_mask)

        return output_tensor, partial(self.loss_func, tokens, loss_mask)

    def train(self):
        test_data_iterator = self.test_data_iterator_list[0]

        if not self.args.skip_train:
            logger.info('training ...')

            if self.args.dataloader_type == 'cyclic' and self.args.retro_project_dir:
                if self.args.retro_cyclic_train_iters is None:
                    raise ValueError("retro_cyclic_train_iters must be provided.")
                self.args.train_iters = self.args.retro_cyclic_train_iters
                logger.info("retro cyclic train iters : %d" % self.args.train_iters)

            iteration = 0
            if self.args.do_train and self.args.train_iters > 0:
                iteration, num_floating_point_operations_so_far = self.train_func(*self.train_args)

            logger.info('after training is done')

            if self.args.save and iteration != 0 and iteration % self.args.save_interval != 0:
                self.save_checkpoint_func(
                    iteration,
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    num_floating_point_operations_so_far
                )
        else:
            logger.info('skipping training (--skip-train is on) ...')

            iteration = self.args.iteration

        if self.args.do_valid:
            prefix = f'iteration {iteration} on validation set'
            self.evaluate_fun(prefix, self.forward_step,
                              self.valid_data_iterator, self.model,
                              iteration, self.process_non_loss_data_func, self.model_config,
                              verbose=True, write_to_tensorboard=not self.args.skip_train)

        if self.args.do_test:
            prefix = f'iteration {iteration} on test set'
            self.evaluate_fun(prefix, self.forward_step,
                              test_data_iterator, self.model,
                              iteration, self.process_non_loss_data_func, self.model_config,
                              verbose=True, write_to_tensorboard=not self.args.skip_train)
