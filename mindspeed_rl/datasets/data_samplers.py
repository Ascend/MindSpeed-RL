# coding=utf-8
# Copyright (c) 2020; NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.


class PretrainingSampler:
    '''
    PretrainingSampler class used for megatron engine.

    parameters:
    total_samples: total samples used for megatron training
    consumed_samples: consumed samples in megatron training
    micro_batch_size: micro batch size
    data_parallel_rank: rank in dp group
    data_parallel_size: dp value
    drop_last: weather to drop out the last batch (default True)
    '''
    def __init__(self,
                 total_samples: int,
                 consumed_samples: int,
                 micro_batch_size: int,
                 data_parallel_rank: int,
                 data_parallel_size: int,
                 drop_last: bool = True
                 ):

        # Keep a copy of input params for later use.
        self.total_samples = total_samples
        self.consumed_samples = consumed_samples
        self.micro_batch_size = micro_batch_size
        self.data_parallel_rank = data_parallel_rank
        self.micro_batch_times_data_parallel_size = \
            self.micro_batch_size * data_parallel_size
        self.drop_last = drop_last

        # Sanity checks.
        if self.total_samples <= 0:
            raise ValueError('no sample to consume: {}'.format(self.total_samples))
        if self.consumed_samples >= self.total_samples:
            raise ValueError('no samples left to consume: {}, {}'.format(
                self.consumed_samples, self.total_samples))

        if self.micro_batch_size <= 0:
            raise ValueError('micro batch size {} should be larger '
                             'than 0'.format(self.micro_batch_size))
        if data_parallel_size <= 0:
            raise ValueError('data_parallel_size {} should be larger'
                             ' than 0'.format(data_parallel_size))
        if self.data_parallel_rank >= data_parallel_size:
            raise ValueError('data_parallel_rank should be smaller than data'
                             ' size: {}, {}'.format(self.data_parallel_rank, data_parallel_size))

    def __len__(self):
        return self.total_samples

    def get_start_end_idx(self):
        start_idx = self.data_parallel_rank * self.micro_batch_size
        end_idx = start_idx + self.micro_batch_size
        return start_idx, end_idx

    def __iter__(self):
        batch = []
        # Last batch will be dropped if drop_last is not set False
        for idx in range(self.consumed_samples, self.total_samples):
            batch.append(idx)
            if len(batch) == self.micro_batch_times_data_parallel_size:
                start_idx, end_idx = self.get_start_end_idx()
                yield batch[start_idx:end_idx]
                batch = []

        # Check the last partial batch and see drop_last is set
        if len(batch) > 0 and not self.drop_last:
            start_idx, end_idx = self.get_start_end_idx()
            yield batch[start_idx:end_idx]


class PromptSampler:
    def __init__(self, total_samples, consumed_samples, batch_size, drop_last=True
                 ):
        # Keep a copy of input params for later use.
        self.total_samples = total_samples
        self.consumed_samples = consumed_samples
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __len__(self):
        return self.total_samples

    def __iter__(self):
        indices = list(range(self.consumed_samples, self.total_samples))
        for i in range(0, len(indices), self.batch_size):
            batch = indices[i:i + self.batch_size]
            if len(batch) == self.batch_size or not self.drop_last:
                yield batch
