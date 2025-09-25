# 镜像构建

在包含dockerfile的目录下运行：

```shell
docker build -t your_image_name:tag .
```

# 创建镜像容器

在使用此dockerfile之前，请安装配套的昇腾[CANN](https://support.huawei.com/carrier/navi?coltype=software#col=software&detailId=PBI1-265642021&path=PBI1-262732867/PBI1-262735886/PBI1-22892969/PBI1-23710427/PBI1-251168373)。
CANN的安装请参考[CANN安装](../install_guide.md)：

```shell
# 挂载镜像
docker run -dit --name 'rl_test' -v /usr/local/Ascend/driver:/usr/local/Ascend/driver -v /usr/local/Ascend/ascend-toolkit:/usr/local/Ascend/ascend-toolkit -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware  -v /usr/local/sbin/:/usr/local/sbin/ -v /home/:/home/ your_image_name:tag bash
```

# 登录镜像并确认环境状态

```shell
# 登录容器
docker exec -it rl_test /bin/bash                           
# 确认npu是否可以正常使用，否则返回3.检查配置
npu-smi info
```

# 编译以及安装apex

```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh 
git clone -b master https://gitcode.com/Ascend/apex.git
cd apex/
bash scripts/build.sh --python=3.10

cd apex/dist/
pip uninstall apex
pip install --upgrade apex-0.1+ascend*.whl
```

# 单机以及多机模型的预训练任务运行
基于镜像和仓库代码，完成环境部署，可执行单机和多机的预训练任务。
