# coding=utf-8
# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.
import os
import sys
import glob
import shutil
import argparse

from mindspeed_rl.utils.loggers import Loggers

logger = Loggers('check_mindstudio_output')


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


def check_msprobe_output(msprobe_dir: str) -> bool:
    """Check if msprobe deliverables are generated correctly"""
    logger.info('Starting msprobe deliverables check...')

    # Check if msprobe_data directory exists
    if not os.path.exists(msprobe_dir):
        logger.error(f'Msprobe data directory not found: {msprobe_dir}')
        return False

    if not os.path.isfile(os.path.join(msprobe_dir, "configurations.json")):
        logger.error(f'Configurations directory not found: {os.path.join(msprobe_dir, "configurations.json")}')
        return False
    
    if not os.path.isfile(os.path.join(msprobe_dir, "data", "responses", "step0", "rank0", "responses.json")):
        logger.error(f'Msprobe key data response not found: '
                     f'{os.path.join(msprobe_dir, "data", "responses", "step0", "rank0", "responses.json")}')
        return False
    
    if not os.path.isdir(os.path.join(msprobe_dir, "actor_update")):
        logger.error(f'Msprobe actor update directory not found: {os.path.join(msprobe_dir, "actor_update")}')
        return False

    if not os.path.isdir(os.path.join(msprobe_dir, "reference_compute_log_prob")):
        logger.error(f'Msprobe reference compute_log_prob directory not found: {os.path.join(msprobe_dir, "reference_compute_log_prob")}')
        return False
    
    logger.info('All msprobe deliverables check passed!')
    return True


def clean_msprobe_output(msprobe_dir: str):
    """Clean up msprobe deliverables"""
    logger.info('Starting cleanup...')
    if os.path.exists(msprobe_dir):
        try:
            shutil.rmtree(msprobe_dir)
            logger.info(f'Successfully deleted directory: {msprobe_dir}')
        except Exception as e:
            logger.error(f'Failed to delete directory: {e}')


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Check and clean Profiler deliverables")
    parser.add_argument("--profiler-dir", type=str, default="./profiler_data",
                      help="Path to profiler data directory")
    parser.add_argument("--msprobe-dir", type=str, default="./msprobe_dump",
                      help="Path to msprobe data directory")
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        if check_profiler_output(args.profiler_dir) and check_msprobe_output(args.msprobe_dir):
            sys.exit(0)
        else:
            sys.exit(1)
    finally:
        clean_profiler_output(args.profiler_dir)
        clean_msprobe_output(args.msprobe_dir)

if __name__ == "__main__":
    main()
