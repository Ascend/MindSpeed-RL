#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
export PYTHONPATH=$SCRIPT_DIR/../../..:$PYTHONPATH

echo "start test_actor_hybrid_worker st"

python $SCRIPT_DIR/test_actor_hybrid_worker.py