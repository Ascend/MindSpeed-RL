pkill -9 python
ray stop --force
rm -rf /tmp/ray

# ============================================================
# 其他修正与加速优化
# ============================================================

# 1. 修正 RayDAPOTrainer 源码（位于 /recipe/dapo/dapo_ray_trainer.py）
#
#    - 原始代码：
#        gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
#        new_batch = new_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
#    - 修改后：
#        gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=False)
#        new_batch = new_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=False)
#
#    将 interleave 设置为 False，可在 rollout.n 较大时，避免多个相同样本集中在同一实例上运行，从而降低推理开销。
# ------------------------------------------------------------
# 2. 安装 jemalloc 以优化内存管理
#
#   可通过源码编译安装，前往官方仓库获取最新稳定版本
#
#   安装步骤：
#       tar -xvf jemalloc-{version}.tar.bz2
#       cd jemalloc-{version}
#       ./configure --prefix=/usr/local
#       make
#       make install
#
#   设置环境变量(假设安装路径为 /usr/local/lib/libjemalloc.so.2): export LD_PRELOAD=/usr/local/lib/libjemalloc.so.2


#设置运行环境变量
export RAY_DEDUP_LOGS=0
export HCCL_EXEC_TIMEOUT=17340
export HCCL_CONNECT_TIMEOUT=7200
export VLLM_USE_V1=1
export VLLM_VERSION=0.9.1
export HCCL_IF_BASE_PORT=23999
export VLLM_ASCEND_ENABLE_MOE_ALL2ALL_SEQ=1
export HCCL_ASYNC_ERROR_HANDLING=0
export P2P_HCCL_BUFFSIZE=20
export HYDRA_FULL_ERROR=1

#修改为当前需要跑的用例路径
DEFAULT_SH="./test_dapo_qwen3_235b_megatron_A3.sh"
echo "Use $DEFAULT_SH"

ulimit -n 65536
mkdir logs

NNODES=16
NPUS_PER_NODE=16
#修改为对应主节点IP
MASTER_ADDR="IP FOR MASTER NODE"
#修改为当前节点的通信网卡
SOCKET_IFNAME="Your SOCKET IFNAME"
export TP_SOCKET_IFNAME="SOCKET IFNAME FOR CURRENT NODE"
export HCCL_SOCKET_IFNAME="SOCKET IFNAME FOR CURRENT NODE"
export GLOO_SOCKET_IFNAME="SOCKET IFNAME FOR CURRENT NODE"
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
          bash $DEFAULT_SH
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