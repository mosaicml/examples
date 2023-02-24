# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

from abc import ABC

import torch.nn as nn
from composer.utils import dist
from torch.distributed import new_group
from composer.utils.dist import get_global_rank, get_local_rank, get_local_world_size, get_node_rank, get_world_size


class BaseMoE(nn.Module, ABC):
    """Abstract class for different MoE implementations.

    MoE experts are only synced within their own process group but the dist
    process group can be shared across MoE layers. _moe_pg being a class attribute
    enables a single dist process group to be shared for all MoE layers.

    param_init_fn enables MosaicGPT to be initialized using `meta` parameters

    param_count allows MosaicGPT to count parameters across experts

    active_param_count enables MosaicGPT to count per-token FLOPs by counting
    parameters activated in the forward pass
    """
    _moe_pg = None

    def __init__(self):
        super().__init__()
        # if BaseMoE._moe_pg is None:
        #     # MOE Class init current rank process group instead of initializing a pg
        #     # for every moe layer
        #     # can be overridden in GPT model if needed

        #     pg = 'local_rank_across_nodes'  # 'node'  # 'self'  # [dist.get_global_rank()]

        #     if pg == 'self':
        #         ranks = [get_global_rank()]
        #     elif pg == 'node':
        #         node_rank = get_node_rank()
        #         local_world_size = get_local_world_size()
        #         ranks = list(range(node_rank * local_world_size, (node_rank + 1) * local_world_size))
        #     elif pg == 'local_rank_across_nodes':
        #         local_rank = get_local_rank()
        #         local_world_size = get_local_world_size()
        #         num_nodes = get_world_size() // get_local_world_size()
        #         ranks = [local_rank + local_world_size * n for n in range(num_nodes)]
        #     elif isinstance(pg, list):
        #         ranks = pg
            
        #     BaseMoE._moe_pg = new_group(ranks=ranks)
        # #     # BaseMoE._moe_pg = [dist.get_global_rank()]
        # #     BaseMoE._moe_pg = 'self'

        # self.moe = None

    @classmethod
    def param_init_fn(cls, module, cfg):
        raise NotImplementedError

    @classmethod
    def param_count(cls, parent_module):
        raise NotImplementedError

    @classmethod
    def active_param_count(cls, parent_module, use_capacity_factor=False):
        raise NotImplementedError
