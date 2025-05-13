# coding=utf-8
# Copyright (c) 2020; NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.


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
