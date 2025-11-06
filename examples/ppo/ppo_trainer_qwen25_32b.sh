pkill -9 python
ray stop --force
export RAY_DEDUP_LOGS=0
export HYDRA_FULL_ERROR=1

DEFAULT_YAML="ppo_qwen25_32b_A3"
YAML=${1:-$DEFAULT_YAML}
echo "Use $YAML"

ulimit -n 32768
mkdir logs
# 保证用户基础环境umask设置，避免用户本身环境umask设置导致生成过高权限的文件夹
umask 0027

export TASK_QUEUE_ENABLE=2
export HCCL_IF_BASE_PORT=24703
export LD_PRELOAD=/usr/local/lib/libjemalloc.so.2
export MALLOC_MMAP_THRESHOLD_=512768
export LCAL_COMM_ID=127.0.0.1:27001

NNODES=2
NPUS_PER_NODE=16
#修改为对应主节点IP
MASTER_ADDR="IP FOR MASTER NODE"
#修改为当前节点的通信网卡
SOCKET_IFNAME="SOCKET IFNAME FOR CURRENT NODE"
#获取当前节点IP
CURRENT_IP=$(ifconfig $SOCKET_IFNAME | grep -Eo 'inet (addr:)?([0-9]{1,3}\.){3}[0-9]{1,3}' | awk '{print $NF}')

if [ "$MASTER_ADDR" = "$CURRENT_IP" ]; then
  # 主节点启动
  ray start --head --port 6766 --dashboard-host=$MASTER_ADDR --node-ip-address=$CURRENT_IP --dashboard-port=8260 --resources='{"NPU": '$NPUS_PER_NODE'}'

  while true; do
      ray_status_output=$(ray status)
      npu_count=$(echo "$ray_status_output" | grep -oP '(?<=/)\d+\.\d+(?=\s*NPU)' | head -n 1)
      npu_count_int=$(echo "$npu_count" | awk '{print int($1)}')
      device_count=$((npu_count_int / $NPUS_PER_NODE))

      # 判断 device_count 是否与 NNODES 相等
      if [ "$device_count" -eq "$NNODES" ]; then
          echo "Ray cluster is ready with $device_count devices (from $npu_count NPU resources), starting Python script."
          ray status
          python cli/train_ppo.py --config-name $YAML 2>&1 | tee logs/training.log
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
      ray start --address="$MASTER_ADDR:6766" --resources='{"NPU": '$NPUS_PER_NODE'}' --node-ip-address=$CURRENT_IP

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