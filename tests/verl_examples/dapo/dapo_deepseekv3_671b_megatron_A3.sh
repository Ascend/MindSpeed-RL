pkill -9 python
ray stop --force
rm -rf /tmp/ray/*

#! HCCL 相关配置
export HCCL_EXEC_TIMEOUT=7200
export HCCL_EVENT_TIMEOUT=7200
export HCCL_CONNECT_TIMEOUT=7200
export ACL_DEVICE_SYNC_TIMEOUT=7200
export HCCL_ASYNC_ERROR_HANDLING=0
export P2P_HCCL_BUFFSIZE=30
export HCCL_BUFFSIZE=300
export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050

ulimit -n 32768
export RAY_DEDUP_LOGS=1  # Ray 日志去重
export HYDRA_FULL_ERROR=1
export TASK_QUEUE_ENABLE=2  
export PYTORCH_NPU_ALLOC_CONF="max_split_size_mb:2048"
export PYTHONUNBUFFERED=1

# VLLM设置
export VLLM_USE_V1=1
export VLLM_VERSION=0.11.0
export VLLM_ASCEND_ENABLE_NZ=0 #影响精度

#修改为当前需要跑的用例路径
DEFAULT_SH="./test_dapo_deepseekv3_671b_megatron_A3.sh"
echo "Use $DEFAULT_SH"
# 日志
export JOB_LOG_DIR="PATH TO SAVE LOG"
export JOB_LOG_DIR_CURR=${JOB_LOG_DIR}/$(date +"%Y%m%d_%H%M%S")

ulimit -n 65536
mkdir logs

# 修改为当前节点的通信网卡
SOCKET_IFNAME="Your SOCKET IFNAME"
export HCCL_SOCKET_IFNAME=$SOCKET_IFNAME
export TP_SOCKET_IFNAME=$SOCKET_IFNAME
export GLOO_SOCKET_IFNAME=$SOCKET_IFNAME
export GLOO_SOCKET_TIMEOUT=7200000
# 修改集群配置
export NNODES=16
export NPU_PER_NODE=16
#修改为对应主节点IP
export MASTER_ADDR="IP FOR MASTER NODE"
#获取当前节点IP
CURRENT_IP=$(ifconfig $TP_SOCKET_IFNAME | grep -Eo 'inet (addr:)?([0-9]{1,3}\.){3}[0-9]{1,3}' | awk '{print $NF}')

# 删除lock，防止卡住
find /root/.cache/torch_extensions -type f -name lock -delete

# MindSpeed预编译
python3 -c "import mindspeed; from mindspeed.op_builder import RotaryPositionEmbeddingOpBuilder; RotaryPositionEmbeddingOpBuilder().load()" &
python3 -c "import mindspeed; from mindspeed.op_builder import MoeTokenPermuteOpBuilder; MoeTokenPermuteOpBuilder().load()" &
python3 -c "import mindspeed; from mindspeed.op_builder import GMMOpBuilder; GMMOpBuilder().load()" &
python3 -c "import mindspeed; from mindspeed.op_builder import GMMV2OpBuilder; GMMV2OpBuilder().load()" &
python3 -c "import mindspeed; from mindspeed.op_builder import MoeTokenUnpermuteOpBuilder; MoeTokenUnpermuteOpBuilder().load()" &
python3 -c "import mindspeed; from mindspeed.op_builder import MatmulAddOpBuilder; MatmulAddOpBuilder().load()" &
python3 -c "import mindspeed; from mindspeed.op_builder import GroupMatmulAddOpBuilder; GroupMatmulAddOpBuilder().load()" &


if [ "$MASTER_ADDR" = "$CURRENT_IP" ]; then
    mkdir -p $JOB_LOG_DIR_CURR
    # 主节点启动
    ray start --head --port 6766 --dashboard-host=$MASTER_ADDR --node-ip-address=$CURRENT_IP --dashboard-port=8260 --resources='{"NPU": '$NPU_PER_NODE'}'
    while true; do
        ray_status_output=$(ray status)
        npu_count=$(echo "$ray_status_output" | grep -oP '(?<=/)\d+\.\d+(?=\s*(NPU|GPU))' | head -n 1)
        npu_count_int=$(echo "$npu_count" | awk '{print int($1)}')
        device_count=$((npu_count_int / $NPU_PER_NODE))

        # 判断 device_count 是否与 NNODES 相等
        if [ "$device_count" -eq "$NNODES" ]; then
            echo "Ray cluster is ready with $device_count devices (from $npu_count NPU resources), starting Python script."
            ray status
            bash $DEFAULT_SH 2>&1 |tee $JOB_LOG_DIR_CURR/run.log 
            break
        else
            echo "Waiting for Ray to allocate $NNODES devices. Current device count: $device_count"
            sleep 5
        fi
    done
else
    # 子节点尝试往主节点注册ray直到成功
    while true; do
        # 尝试连接 Ray 集群
        ray start --address="$MASTER_ADDR:6766" --resources='{"NPU": '$NPU_PER_NODE'}' --node-ip-address=$CURRENT_IP

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
fi

sleep 600