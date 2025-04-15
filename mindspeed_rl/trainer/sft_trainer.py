# Copyright (c) 2023; NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.
import os
from functools import partial
from abc import ABC

import torch

from mindspeed_rl.utils import Loggers
from mindspeed_rl.utils.utils import get_tune_attention_mask
from .utils.training import (
    get_finetune_data_on_this_tp_rank, broadcast_data, average_losses_across_data_parallel_group
)

logger = Loggers('stf_trainer')


class SFTTrainer(ABC):
    """
        Trainer to use while training reward model.

    Args:
        model (torch.nn.Module): the model to train
        strategy (Strategy): the strategy to use for training
        optim(Optimizer): the optimizer to use for training
        train_dataset (RewardDataset): the dataset to use for training
        eval_dataset (RewardDataset): the dataset to use for evaluation
        batch_size (int, defaults to 1): the batch size while training
        max_epochs (int, defaults to 2): the number of epochs to train
        optim_kwargs (dict, defaults to {'lr':1e-4}): the kwargs to use while initializing optimizer
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
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.train_data_iterator = train_data_iterator
        self.valid_data_iterator = valid_data_iterator
        self.test_data_iterator_list = test_data_iterator_list
        self.scheduler = scheduler
        self.process_non_loss_data_func = process_non_loss_data_func
        self.model_config = model_config
        self.train_args = (self.forward_step, self.model, self.optimizer, self.scheduler, self.train_data_iterator,
                           self.valid_data_iterator, self.process_non_loss_data_func, self.model_config)
        self.train_func = kwargs.get('train_func', None)
        self.parallel_state = kwargs.get('parallel_state', None)
        self.save_checkpoint_func = kwargs.get('save_checkpoint_func', None)
        self.evaluate_fun = kwargs.get('evaluate_fun', None)
        self.generate_seq_len_fun = kwargs.get('generate_seq_len_fun', None)
        self.get_batch_on_this_cp_rank = kwargs.get('batch_cp_func', None)

    def get_batch(self, data_iterator):
        """Generate a batch."""
        # Items and their type.
        keys = ['input_ids', 'attention_mask', 'labels']
        if self.args.reset_position_ids:
            keys += ['position_ids']
        data_type = torch.int64

        if (not self.parallel_state.is_pipeline_first_stage()) and (not self.parallel_state.is_pipeline_last_stage()):
            if self.args.variable_seq_lengths and self.args.pipeline_model_parallel_size > 2:
                tokens, attention_mask = get_finetune_data_on_this_tp_rank(data_iterator,
                                                                           self.parallel_state,
                                                                           self.args.reset_position_ids,
                                                                           self.args.tokenizer_padding_side)
                return tokens, None, None, attention_mask, None
            else:
                # Broadcast data.
                data_b = broadcast_data(keys, next(data_iterator), data_type)
                if self.args.reset_position_ids:
                    self.generate_seq_len_fun(data_b)
                attention_mask_1d = data_b.get('attention_mask').long()
                attention_mask = get_tune_attention_mask(attention_mask_1d)
                batch = {'attention_mask': attention_mask}
                batch = self.get_batch_on_this_cp_rank(batch)
                return None, None, None, batch['attention_mask'], None

        # Broadcast data.
        data_b = broadcast_data(keys, next(data_iterator), data_type, self.parallel_state)

        # Unpack
        labels = data_b.get('labels').long()
        tokens = data_b.get('input_ids').long()
        attention_mask_1d = data_b.get('attention_mask').long()
        # ignored label -100
        loss_mask = torch.where(labels == -100, 0, 1)

        if self.args.reset_position_ids:
            position_ids = data_b.get('position_ids').long()
            self.generate_seq_len_fun(data_b)
            batch = {
                'tokens': tokens,
                'labels': labels,
                'loss_mask': loss_mask,
            }
            batch = self.get_batch_on_this_cp_rank(batch)
            batch['attention_mask'] = None
            batch['position_ids'] = position_ids
            return batch.values()

        attention_mask = get_tune_attention_mask(attention_mask_1d,
                                                 tokenizer_padding_side=self.args.tokenizer_padding_side,
                                                 reset_attention_mask=self.args.reset_attention_mask
                                                 )
        position_ids = None
        batch = {
            'tokens': tokens,
            'labels': labels,
            'loss_mask': loss_mask,
            'attention_mask': attention_mask,
            'position_ids': position_ids
        }
        batch = self.get_batch_on_this_cp_rank(batch)
        return batch.values()

    def loss_func(self, input_tensor: torch.Tensor, output_tensor: torch.Tensor):
        """Loss function.

        Args:
            input_tensor (torch.Tensor): Used to mask out some portions of the loss
            output_tensor (torch.Tensor): The tensor with the losses
        """
        loss_mask = input_tensor

        losses = output_tensor.float()
        loss_mask = loss_mask[..., 1:].view(-1).float()
        if self.args.context_parallel_size > 1:
            loss = torch.cat([torch.sum(losses.view(-1) * loss_mask).view(1), loss_mask.sum().view(1)])
            torch.distributed.all_reduce(loss,
                                         group=self.parallel_state.get_context_parallel_group())
            loss = loss[0] / loss[1]
        else:
            loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

        # Check individual rank losses are not NaN prior to DP all-reduce.
        if self.args.check_for_nan_in_loss_and_grad:
            global_rank = torch.distributed.get_rank()
            if loss.isnan():
                raise ValueError(f'Rank {global_rank}: found NaN in local forward loss calculation. '
                                 f'Device: {torch.cuda.current_device()}, node: {os.uname()[1]}')

        # Reduce loss for logging.
        averaged_loss = average_losses_across_data_parallel_group([loss], parallel_state=self.parallel_state)

        return loss * self.args.context_parallel_size, {'lm loss': averaged_loss[0]}

    def forward_step(self, data_iterator, model):
        """Forward training step.

        Args:
            data_iterator : Input data iterator
            model (GPTModel): The GPT Model
        """

        # Get the batch.
        tokens, labels, loss_mask, attention_mask, position_ids = self.get_batch(
            data_iterator)

        output_tensor = model(tokens, position_ids, attention_mask,
                              labels=labels)

        return output_tensor, partial(self.loss_func, loss_mask)

    def train(self):
        test_data_iterator = self.test_data_iterator_list[0]
        (forward_step_func, model, optimizer, opt_param_scheduler, train_data_iterator, valid_data_iterator,
         process_non_loss_data_func, config) = self.train_args

        if not self.args.skip_train:
            logger.info('training ...')

            if self.args.dataloader_type == 'cyclic' and self.args.retro_project_dir:
                if self.args.retro_cyclic_train_iters is None:
                    raise ValueError("retro_cyclic_train_iters must be provided.")
                self.args.train_iters = self.args.retro_cyclic_train_iters
                logger.info("retro cyclic train iters : %d" % self.args.train_iters)

            iteration = 0
            if self.args.do_train and self.args.train_iters > 0:
                iteration, num_floating_point_operations_so_far = self.train_func(
                    *self.train_args)

            logger.info('after training is done')

            if self.args.save and iteration != 0 and iteration % self.args.save_interval != 0:
                self.save_checkpoint_func(
                    iteration,
                    model,
                    optimizer,
                    opt_param_scheduler,
                    num_floating_point_operations_so_far
                )
        else:
            logger.info('skipping training (--skip-train is on) ...')

            iteration = self.args.iteration

        if self.args.do_valid:
            prefix = f'iteration {iteration} on validation set'
            self.evaluate_fun(prefix, forward_step_func,
                              valid_data_iterator, model,
                              iteration, process_non_loss_data_func, config,
                              verbose=True, write_to_tensorboard=not self.args.skip_train)

        if self.args.do_test:
            prefix = f'iteration {iteration} on test set'
            self.evaluate_fun(prefix, forward_step_func,
                              test_data_iterator, model,
                              iteration, process_non_loss_data_func, config,
                              verbose=True, write_to_tensorboard=not self.args.skip_train)
