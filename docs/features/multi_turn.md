# 多轮迭代

## 背景介绍
我们知道，固定参数量情况下，大模型的能力是有限的。若想进一步提升LLM在具体问题上的表现，可以通过让LLM学会使用工具来提高解决问题的能力，并基于多轮迭代框架来提高LLM的探索能力。
为增强LLM调用工具的能力，本仓库提供了多轮迭代（Multi Turn）的解决方案。

## 方案介绍
### ReTool
[ReTool](https://github.com/ReTool-RL/ReTool)是一个通过强化学习算法来赋予大模型策略性使用工具进行复杂推理能力的创新框架，由字节 seed 团队 提出。其关键点在于：在rollout的过程中，LLM推理与代码执行交错执行，多轮迭代之后，根据最终输出的结果反馈指导模型学习何时以及如何调用工具。

## 使用介绍
### ReTool
目前，仓库已支持ReTool。
其中，训练脚本参考[dapo_trainer_qwen25_7b_multi_turn.sh](../../examples/multi_turn/dapo_trainer_qwen25_7b_multi_turn.sh)，
训练配置参考[dapo_qwen25_7b_A2_multi_turn.yaml](../../configs/dapo_qwen25_7b_A2_multi_turn.yaml)。
具体来说，多轮迭代涉及的配置如下：
```yaml
rl_config:
  verifier_function: ["retool_reward"]
    
  multi_turn_enable: true
  tool_config_path: ./configs/tools/retool_config.yaml
  max_tool_calls: 1
  max_parallel_calls: 1
  max_total_response_length: 2048
  max_tool_response_length: 256
  tool_response_truncate_side: 'middle'
  tool_parser_format: 'hermes'

  async_engine: true

```

其中：
`verifier_function` 表示使用的规则奖励函数，默认传入**retool_reward**

`multi_turn_enable` 表示是否使能多轮迭代

`tool_config_path` 需要传入工具相关配置的路径，需要传入工具相关配置的路径，默认传入**search_tool_config.yaml**

`max_tool_calls` 表示多轮迭代过程中，累积最多可以调用多少次工具

`max_parallel_calls` 表示每次调用工具过程中，最多可以同时调用多少次工具

`max_total_response_length` 表示多轮迭代过程中，拼接完每次推理生成的response和工具调用结果的完整response长度的最大值

`max_tool_response_length` 表示每次调用工具后，工具执行结果长度的最大值，若超过，需截断

`tool_response_truncate_side` 表示每次调用工具后，若工具执行结果长度超过最大值，截断的方式，目前支持**left**、**right**、**middle**

`tool_parser_format` 表示每次从推理生成的response中，提取工具调用内容的方式，目前仅支持**hermes**

`async_engine` 表示是否使能异步引擎，目前多轮迭代仅在异步引擎开启时可用

#### 数据预处理
特别地，在进行**数据预处理**时，需要选择对应的处理模板**qwen_retool**，即在对应数据集yaml中，prompt_type字段更改为如下设置：
```yaml
prompt_type: qwen_retool
```

#### 工具配置
除此之外，retool对应的工具配置参考[retool_config.yaml](../../configs/tools/retool_config.yaml)，具体的配置如下：
```yaml
tools:
  - class_name: "mindspeed_rl.tools.retool.ReTool"
    config:
      sandbox_fusion_url: "http://localhost:8080/run_code"
      num_workers: 128
      enable_global_rate_limit: true
      rate_limit: 128
      default_timeout: 30
      default_language: "python"
      memory_limit_mb: 1024
      type: native

    tool_schema:
      type: "function"
      function:
        name: "code_interpreter"
        description: "A tool for executing code."
        parameters:
          type: "object"
          properties:
            code:
              type: "string"
              description: "The code to execute."
          required: ["code"]
```
其中：

`class_name` 需要传入准备使用的工具子类

`config` 包含了初始化对应工具子类的所有参数

`sandbox_fusion_url` 需要传入部署好的代码沙箱URL

`num_workers` 表示线程池（多线程并发调用代码沙箱）默认组的并发度

`enable_global_rate_limit` 表示是否初始化对应线程池worker

`rate_limit` 控制线程池同时获取资源的线程数量

`default_timeout` 表示调用代码沙箱默认的代码运行/编译超时时间

`default_language` 表示调用代码沙箱默认传入的代码语言

`memory_limit_mb` 表示调用代码沙箱默认能使用的最大内存量

`type` 控制是否是通过function call方式调用，目前仅支持**native**

`tool_schema` 表示此工具对应的schema，参考OpenAIFunctionToolSchema的格式

#### OpenAI Function calling
[Function calling](https://platform.openai.com/docs/guides/function-calling)也叫tool calling，用户输入基于统一的tool schema格式的schema配置，模型自行判断何时需要调用哪些工具，并且可以根据目标工具的schema生成符合要求的请求参数。下面是tool schema定义的一个样例：
```yaml
{
    "type": "function",
    "name": "get_weather",
    "description": "Retrieves current weather for the given location.",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "City and country e.g. Bogotá, Colombia"
            },
            "units": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "Units the temperature will be returned in."
            }
        },
        "required": ["location", "units"],
        "additionalProperties": false
    },
    "strict": true
}
```
其中：

`type` 一般设置为function

`name` 表示函数名

`description` 用于描述函数功能，模型会根据这段描述决定函数调用方式

`parameters` 需要传入一个Json Schema对象，以准确地定义函数所接受的参数。若调用函数时不需要传入参数，省略该参数即可

`strict` 表示函数调用是否要强制匹配当前格式

#### 工具部署
在整个多轮迭代过程中，ReTool涉及到的交互工具是代码解释器，此工具可参考字节开源的代码沙箱工具[SandboxFusion](https://github.com/bytedance/SandboxFusion)README文档中Installation来部署实现。 具体来说，代码沙箱的运行环境可以通过docker或者conda的方式来安装。以conda方式为例，其安装步骤如下：
```shell
conda create -n sandbox -y python=3.12
conda activate sandbox
pip install poetry
poetry install
pip install -r runtime/python/requirements.txt --ignore-requires-python
# to build the real docs, run `cd docs && npm ci && npm run build`
mkdir -p docs/build
```
安装完对应环境后，通过如下命令来启动代码沙箱服务：
```shell
make run-online
```


### Search Tool
目前，仓库已支持Search Tool。
其中，训练脚本参考[dapo_trainer_qwen25_7b_multi_turn.sh](../../examples/multi_turn/dapo_trainer_qwen25_7b_multi_turn.sh)，
训练配置参考[dapo_qwen25_7b_A2_multi_turn.yaml](../../configs/dapo_qwen25_7b_A2_multi_turn.yaml)。
具体来说，多轮迭代涉及的配置如下：
```yaml
rl_config:
  verifier_function: ["retool_reward"]
    
  multi_turn_enable: true
  tool_config_path: ./configs/tools/search_tool_config.yaml
  max_tool_calls: 1
  max_parallel_calls: 1
  max_total_response_length: 2048
  max_tool_response_length: 256
  tool_response_truncate_side: 'middle'
  tool_parser_format: 'hermes'

  async_engine: true
```

其中：
`tool_config_path` 需要传入工具相关配置的路径，使用Search Tool需要传入**search_tool_config.yaml**

其余参数说明可查阅上一节ReTool。


#### 自定义规则奖励
训练配置[dapo_qwen25_7b_A2_multi_turn.yaml](../../configs/dapo_qwen25_7b_A2_multi_turn.yaml)默认传入的是**retool_reward**，用户可通过[rule_verifier.py](../../mindspeed_rl/models/rule_verifier.py)来增加自定义奖励函数，同时修改文件中的rule_verifier_function变量增加对应的函数映射，最后将训练配置中的**retool_reward**换成自定义奖励函数名即可。


#### 数据预处理
特别地，在进行**数据预处理**时，可以选择对应的处理模板**qwen_search_tool**，即在对应数据集yaml中，prompt_type字段更改为如下设置：
```yaml
prompt_type: qwen_search_tool
```
也可以根据需要对模板进行自定义修改。


#### 工具配置
除此之外，search tool对应的工具配置参考[search_tool_config.yaml](../../configs/tools/search_tool_config.yaml)，具体的配置如下：
```yaml
tools:
  - class_name: "mindspeed_rl.tools.search_tool.SearchTool"
    config:
      retrieval_service_url: "http://localhost:8080/retrieve"
      num_workers: 120
      enable_global_rate_limit: true
      rate_limit: 120
      timeout: 30
      topk: 3
      type: native

    tool_schema:
      type: "function"
      function:
        name: "search"
        description: "Searches the web for relevant information based on the given query."
        parameters:
          type: "object"
          properties:
            query_list:
              type: "array"
              item:
                type: "string"
              description: "A list of fully-formed semantic queries. The tool will return search results for each query."
          required: ["query_list"]
```
其中：

`class_name` 需要传入准备使用的工具子类

`config` 包含了初始化对应工具子类的所有参数

`retrieval_service_url` 需要传入部署好的搜索工具URL

`num_workers` 表示线程池（多线程并发调用搜索工具）默认组的并发度

`enable_global_rate_limit` 表示是否初始化对应线程池worker

`rate_limit` 控制线程池同时获取资源的线程数量

`timeout` 表示调用搜索工具默认的请求超时时间

`topk` 表示调用搜索工具返回结果数量的最大值

`type` 控制是否是通过function call方式调用，目前仅支持**native**

`tool_schema` 表示此工具对应的schema，参考OpenAIFunctionToolSchema的格式


#### 工具部署
在整个多轮迭代过程中，search tool涉及到的交互工具是搜索引擎。用户可以在本地搭建自定义的搜索引擎，也可以参考[search-R1](https://github.com/PeterGriffinJin/Search-R1/blob/main/docs/retriever.md)中的说明，使用合适的集成搜索引擎。
