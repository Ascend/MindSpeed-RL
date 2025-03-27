
DEFAULT_YAML="grpo_trainer_deepseekv3_671b"
YAML=${1:-$DEFAULT_YAML}

export CUDA_DEVICE_MAX_CONNECTIONS=1
export HYDRA_FULL_ERROR=1
export HCCL_CONNECT_TIMEOUT=3600
export HCCL_EXEC_TIMEOUT=3600
export HCCL_DETERMINISTIC=True
export HCCL_IF_BASE_PORT=48000
export GLOO_SOCKET_IFNAME=eth0
export TP_SOCKET_IFNAME=eth0

ulimit -n 32768
mkdir logs

# 共使用多少节点训练
NNODES=52
# 每个节点有多少张卡
NPUS_PER_NODE=8
MASTER_ADDR="主节点 IP 地址"

# 子节点尝试往主节点注册ray直到成功
while true; do
    # 尝试连接 Ray 集群
    ray start --address="$MASTER_ADDR:6344" --resources='{"NPU": '$NPUS_PER_NODE'}'

    # 检查连接是否成功
    ray status
    if [ $? -eq 0 ]; then
        echo "Successfully connected to the Ray cluster!"
        break
    else
        echo "Failed to connect to the Ray cluster. Retrying in 5 seconds..."
        sleep 5
    fi
done