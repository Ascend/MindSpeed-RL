RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
MODEL_PATH=${MODEL_PATH:-"${RAY_DATA_HOME}/models/Qwen2.5-7B"}
QUANT_PATH=${QUANT_PATH:-"${RAY_DATA_HOME}/models/Qwen2.5-7B-w8a8"}
PROFILE_PATH=${PROFILE_PATH:-"${RAY_DATA_HOME}/profile.7b.pt"}
CONFIG_PATH=${CONFIG_PATH:-"${RAY_DATA_HOME}/.flashrl_config.7b.yaml"}
DEFAULT_SH="./test_grpo_qwen25_7b_fsdp_int8_A2.sh"
LOG_PATH="./logs"

if ! command -v flashrl &> /dev/null
then
    pip install flash-llm-rl # need to be installed in all nodes in multi-node training
fi

# manually add 'import flash_rl' in 'verl/verl/__init__.py'
if ! grep -q "import flash_rl" "${RAY_DATA_HOME}/verl/__init__.py"; then
    echo "Adding 'import flash_rl' to verl/verl/__init__.py"
    sed -i '1i import flash_rl' "${RAY_DATA_HOME}/verl/__init__.py"
fi

flashrl profile -m ${MODEL_PATH} -q ${QUANT_PATH} -o ${PROFILE_PATH} --fn int8
flashrl setup -m ${QUANT_PATH} -p ${PROFILE_PATH} --fn int8 -o ${CONFIG_PATH}
# (Optional) conduct rollout generation in 16bits and 8bits in a hybrid manner across DP workers
# flashrl setup -a --fn bf16 -o ${CONFIG_PATH}

flashrl cleanup

mkdir -p ${LOG_PATH}
bash $DEFAULT_SH 2>&1 | tee ${LOG_PATH}/grpo_qwen25_7b_fsdp_int8_A2.log