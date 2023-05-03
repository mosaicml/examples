# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

import math
from typing import Union

from composer.utils import dist
from omegaconf import DictConfig
from omegaconf import OmegaConf as om


def update_batch_size_info(cfg: DictConfig):
    global_batch_size, device_microbatch_size = cfg.global_train_batch_size, cfg.device_train_microbatch_size
    if global_batch_size % dist.get_world_size() != 0:
        raise ValueError(
            f'Global batch size {global_batch_size} is not divisible by {dist.get_world_size()} '
            'as a result, the batch size would be truncated, please adjust `global_batch_size` '
            f'to be divisible by world size, {dist.get_world_size()}.')
    device_train_batch_size = global_batch_size// dist.get_world_size()
    if isinstance(device_microbatch_size, int):
        if device_microbatch_size > device_train_batch_size:
            print(
                f'WARNING: device_train_microbatch_size > device_train_batch_size, '
                f'will be reduced from {device_microbatch_size} -> {device_train_batch_size}.'
            )
            device_microbatch_size = device_train_batch_size
    cfg.n_gpus = dist.get_world_size()
    cfg.device_train_batch_size = device_train_batch_size
    cfg.device_train_microbatch_size = device_microbatch_size


def log_config(cfg: DictConfig):
    print(om.to_yaml(cfg))
    if 'wandb' in cfg.get('loggers', {}):
        try:
            import wandb
        except ImportError as e:
            raise e
        if wandb.run:
            wandb.config.update(om.to_container(cfg, resolve=True))
