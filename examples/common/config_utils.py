# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

import math
from typing import Union

from composer.utils import dist
from omegaconf import DictConfig
from omegaconf import OmegaConf as om


def update_batch_size_info(cfg: DictConfig):
    cfg.n_gpus = dist.get_world_size()
    # Safely set `device_eval_batch_size` if not provided by user
    if 'device_eval_batch_size' not in cfg:
        if cfg.device_train_microbatch_size == 'auto':
            cfg.device_eval_batch_size = 1  # TODO debug auto eval microbatching
        else:
            cfg.device_eval_batch_size = cfg.device_train_microbatch_size
    return cfg


def log_config(cfg: DictConfig):
    print(om.to_yaml(cfg))
    if 'wandb' in cfg.get('loggers', {}):
        try:
            import wandb
        except ImportError as e:
            raise e
        if wandb.run:
            wandb.config.update(om.to_container(cfg, resolve=True))
