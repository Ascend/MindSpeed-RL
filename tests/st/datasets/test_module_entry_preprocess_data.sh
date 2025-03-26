SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
export PYTHONPATH=$SCRIPT_DIR/../../..:$PYTHONPATH

echo "start test_preprocess_data st"

python $SCRIPT_DIR/test_preprocess_data.py test_orca_rlhf