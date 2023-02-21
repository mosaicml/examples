# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn
from composer.utils import dist, get_device
from torch.distributed import new_group


class BaseMoE(nn.Module):
    # attach moe process group to class so that all ranks have one process group
    # instead of needing to init as many process grups as moe layers
    _moe_pg = None

    def __init__(self):
        super().__init__()
        if not dist.is_initialized():
            # initialize dist so we can access world size and global rank
            dist.initialize_dist(get_device(None))

        if BaseMoE._moe_pg is None:
            # MOE Class init current rank process group instead of initializing a pg
            # for every moe layer
            # can be overridden in GPT model if needed
            BaseMoE._moe_pg = new_group(ranks=[dist.get_global_rank()])

        self.moe = None

    @classmethod
    def param_init_fn(cls, module, cfg):
        raise NotImplementedError

    @classmethod
    def param_count(cls, parent_module):
        raise NotImplementedError

    @classmethod
    def active_param_count(cls, parent_module, use_capacity_fac=False):
        raise NotImplementedError(f'')

    def forward(self, x):
        return self.moe(x)
