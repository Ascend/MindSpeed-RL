# coding=utf-8
# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.
import os
import sys
import glob
import shutil
import argparse

from mindspeed_rl.utils.loggers import Loggers

logger = Loggers('check_profiler_output')


def check_profiler_output(profiler_dir: str) -> bool:
    """Check if profiler deliverables are generated correctly"""
    logger.info('Starting profiler deliverables check...')

    # Check if profiler_data directory exists
    if not os.path.exists(profiler_dir):
        logger.error(f'Profiler data directory not found: {profiler_dir}')
        return False

    # Find all xxx_ascend_xxx directories
    dirs = glob.glob(os.path.join(profiler_dir, "actor_update", "*_ascend_*"), recursive=True)
    if len(dirs) != 2:
        logger.error(f'Expected 2 *_ascend_* directories, found {len(dirs)}')
        return False

    # Check each xxx_ascend_xxx directory
    for ascend_dir in dirs:
        profiler_output = os.path.join(ascend_dir, "ASCEND_PROFILER_OUTPUT")
        if not os.path.exists(profiler_output) or not os.path.isdir(profiler_output):
            logger.error(f'ASCEND_PROFILER_OUTPUT not found in {ascend_dir}')
            return False

    logger.info('All profiler deliverables check passed!')
    return True


def clean_profiler_output(profiler_dir: str):
    """Clean up profiler deliverables"""
    logger.info('Starting cleanup...')
    if os.path.exists(profiler_dir):
        try:
            shutil.rmtree(profiler_dir)
            logger.info(f'Successfully deleted directory: {profiler_dir}')
        except Exception as e:
            logger.error(f'Failed to delete directory: {e}')


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Check and clean Profiler deliverables")
    parser.add_argument("--profiler-dir", type=str, default="./profiler_data",
                      help="Path to profiler data directory")
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        if check_profiler_output(args.profiler_dir):
            sys.exit(0)
        else:
            sys.exit(1)
    finally:
        clean_profiler_output(args.profiler_dir)

if __name__ == "__main__":
    main()
