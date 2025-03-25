# coding=utf-8
# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.
import os
import logging
from datetime import datetime, timezone

import tensordict
import torch
import torch_npu
import torch.distributed as dist


class Loggers(object):
    def __init__(self,
                 name='root',
                 logger_level=logging.INFO,
                 ):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logger_level)
        
        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logger_level)
            console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)

        self.logger.propagate = False

    def handle_msg(self, msg, level, iteration, steps):
        current_time = str(datetime.now(tz=timezone.utc)).split(".")[0]

        if iteration is not None and steps is not None:
            fmt_msg = f"[{current_time}] "
            fmt_msg += f"iteration: {iteration} / {steps} | "
            if isinstance(msg, dict):
                for key in msg:
                    if key == "grpo/lr":
                        lr = "{:e}".format(msg[key])
                        fmt_msg += f"{key} : {lr} | "
                    else:
                        fmt_msg += f"{key} : {format(msg[key], '.4f')} | "
                fmt_msg = fmt_msg[:-2]
            else:
                fmt_msg = f"{fmt_msg} {str(msg)}"
        else:
            fmt_msg = f"[{current_time}] {level} " + str(msg)
        return fmt_msg

    def info(self, msg, iteration=None, steps=None):
        if torch.distributed.is_initialized():
            if int(os.getenv("RANK", dist.get_rank())) == int(os.getenv("WORLD_SIZE", dist.get_world_size())) - 1:
                format_msg = self.handle_msg(msg, "INFO", iteration, steps)
                self.logger.info(format_msg)
        else:
            format_msg = self.handle_msg(msg, "INFO", iteration, steps)
            self.logger.info(format_msg)

    def warning(self, msg, iteration=None, steps=None):
        format_msg = self.handle_msg(msg, "WARNING", iteration, steps)
        self.logger.warning(format_msg)

    def debug(self, msg, iteration=None, steps=None):
        format_msg = self.handle_msg(msg, "DEBUG", iteration, steps)
        self.logger.debug(format_msg)

    def error(self, msg, iteration=None, steps=None):
        format_msg = self.handle_msg(msg, "ERROR", iteration, steps)
        self.logger.error(format_msg)


class WandbLogger(Loggers):
    """
    一般在trainer中初始化WandbLogger, 记录待可视化的训练指标

    """
    def __init__(self, kwargs):
        super(WandbLogger, self).__init__()

        self.wandb = None
        if kwargs.get("wandb_project", ""):
            self._init_wandb(kwargs)

    def _import_wandb(self):
        try:
            import wandb
        except ImportError as e:
            raise ImportError('Please run "pip install wandb" to install wandb') from e
        self.wandb = wandb

    def _init_wandb(self, kwargs):
        if self.wandb is None:
            self._import_wandb()

        if kwargs.get("wandb_exp_name", "") == "":
            raise ValueError("Please specify the wandb experiment name!")
        if kwargs.get("wandb_save_dir", ""):
            save_dir = kwargs["wandb_save_dir"]
        else:
            # Defaults to the save dir.
            save_dir = os.path.join("./", 'wandb')
        wandb_kwargs = {
            'dir': save_dir,
            'name': kwargs["wandb_exp_name"],
            'project': kwargs["wandb_project"]}
        os.makedirs(wandb_kwargs['dir'], exist_ok=True)

        if not self.wandb.api.api_key:
            if not os.getenv("WANDB_API_KEY"):
                raise ValueError(
                    "Please set your wandb api key in the environment variable, you can set WANDB_API_KEY=$your_wandb_api_key ")
            self.wandb.login(key=os.getenv("WANDB_API_KEY"))

        # 初始化 wandb
        try:
            self.wandb.init(**wandb_kwargs)
        except Exception as e:
            logging.warning(f"Failed to initialize wandb as {e}, switch to offline mode")
            os.environ["WANDB_MODE"] = "offline"
            self.wandb.init(**wandb_kwargs)

    def log_metrics(self, metrics, step=None):
        """
        记录指标，x轴默认是step。

        :param metrics: dict[str, Any]. 指标字典，例如 {"accuracy": 0.95, "loss": 0.1}
        :param step: (int| None). 当前step（可选）

        for example:
        wandb_logger = WandbLogger()
        wandb_logger.log_metrics({"train-loss": 0.4}, step=step)

        """
        if self.wandb:
            self.wandb.log(metrics, step=step)

    def log_define_metrics(self, name, step_metric=None, step_sync=None):
        """
        自定义使用 wandb.log() 记录的指标。

        :param name: str. The name of the metric to customize.
        :param step_metric: str. The name of another metric to serve as the X-axis
                                    for this metric in automatically generated charts.
        :param step_sync: bool. Automatically insert the last value of step_metric into run.log()
                                    if it is not provided explicitly.
                                    Defaults to True if step_metric is specified.
        """
        if self.wandb:
            self.wandb.define_metric(name, step_metric=step_metric, step_sync=step_sync)

    def log_config(self, config):
        """
        记录配置（超参数）。
        :param config: 配置字典
        """
        if self.wandb:
            self.wandb.config.update(config)

    def finish(self):
        """
        结束 wandb 运行。
        """
        if self.wandb:
            self.wandb.finish()
