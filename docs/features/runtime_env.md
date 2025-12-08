### runtime_env 环境变量
环境变量相关参数配置 **（ 注：位于 configs/envs/runtime_env.yaml 中 ）** 说明如下： 
| 参数名 | 说明 |
|--------|------|
| `RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES` | 是否禁用 Ray 对 ASCEND_RT_VISIBLE_DEVICES 的自动设置，'true'为禁用 |
| `TOKENIZERS_PARALLELISM` | 设置tokenizers是否支持并行，'true'为支持 |
| `NCCL_DEBUG` | NCCL Debug日志级别，VERSION、WARN、INFO、TRACE |
| `PYTORCH_NPU_ALLOC_CONF` | 设置缓存分配器行为 |
| `HCCL_CONNECT_TIMEOUT` | HCCL 连接超时时间 |
| `HCCL_EXEC_TIMEOUT` | HCCL 执行超时时间 |
| `HCCL_IF_BASE_PORT` | HCCL 通信端口 |
| `CUDA_DEVICE_MAX_CONNECTIONS` | 设备最大连接数 |
| `HYDRA_FULL_ERROR` | 设置 HYDRA 是否输出完整错误日志 |
| `VLLM_DP_SIZE` | vLLM数据并行度（Data Parallelism）大小，控制数据分片数量，稠密模型需要设置为1，MOE模型要求必须和EP一致 |
| `HCCL_BUFFSIZE` | HCCL通信层单次传输的最大缓冲区大小（单位MB），影响跨设备通信效率 |
| `VLLM_USE_V1` | 使用vLLM的V1 engine API（v1接口），当前只支持 v1 ，需设置为 '1' |
| `USING_LCCL_COM` | 指定不使用 LCCL 通信 |
| `VLLM_VERSION` | 指定使用的vLLM版本号 |
| `VLLM_ENABLE_TOPK_OPTIMZE` | 使能vLLM TOPK性能优化 |
| `VLLM_ASCEND_ACL_OP_INIT_MODE` | vLLM aclop 初始化模式: 0: default, normal init. |
| `TASK_QUEUE_ENABLE` | 控制开启task_queue算子下发队列优化的等级，推荐设置为 '2' 使能 Level 2 优化 |
| `CPU_AFFINITY_CONF` | 指定使用绑核优化，推荐设置为 '1' |
| `LCAL_COMM_ID` | 开启coc特性时配套启用，设置为'127.0.0.1:27001' |
| `GLOO_SOCKET_IFNAME` | 指定 GLOO 框架通信网卡 |
| `TP_SOCKET_IFNAME` | 指定 TP 相关通信网卡 |
| `HCCL_SOCKET_IFNAME` | 指定 HCCL 通信网卡 |