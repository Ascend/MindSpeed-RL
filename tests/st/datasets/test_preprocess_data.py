# coding=utf-8
# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.
import hydra

from preprocess_data import preprocess


@hydra.main(config_path="./", config_name="test_orca_rlhf")
def main(config):
    preprocess(config)


if __name__ == '__main__':
    main()
