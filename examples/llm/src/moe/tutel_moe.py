import math
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from composer.utils import get_device, dist
from omegaconf import DictConfig
from omegaconf.listconfig import ListConfig

from tutel import moe as tutel_moe
from tutel.impls.moe_layer import MOELayer
from tutel.experts.ffn import FusedExpertsNetwork
from tutel.gates.top import LinearTopKGate
from tutel.gates.cosine_top import CosineTopKGate

from examples.common.seed_ctx_manager import SeedContextManager


class TutelMOE(nn.Module):
    # attach moe process group to class so that all ranks have one process group
    # instead of needing to init as many process grups as moe layers
    _moe_pg = None

    def __init__(self, cfg: DictConfig, block_idx: int, device: Optional[str] = None):
        super().__init__()
        if not dist.is_initialized():
            # initialize dist so we can access world size and global rank
            dist.initialize_dist(get_device(None))

        if TutelMOE._moe_pg is None:
            # MOE Class init current rank process group instead of initializing a pg
            # for every moe layer
            # can be overridden in GPT model if needed
            TutelMOE._moe_pg = torch.distributed.new_group(ranks=[dist.get_global_rank()])

        world_size = dist.get_world_size()

        num_experts = cfg.moe.get('num_experts')
        if isinstance(num_experts, ListConfig):
            # enables pyramid moe
            num_experts = num_experts[block_idx]

        if num_experts >= world_size:
            assert num_experts % world_size == 0
            num_local_experts = num_experts // world_size
        else:
            assert world_size % num_experts == 0
            num_local_experts = - world_size // num_experts

        gate_type = {
            'type': cfg.moe.get('gate_type', 'top'),
            'k': cfg.moe.get('gate_k', 1),
            'fp32_gate': cfg.moe.get('fp32_gate', True),
            'device': device
        }
        if cfg.moe.get('capacity_factor', None) is not None:
            gate_type['capacity_factor'] = cfg.moe.get('capacity_factor')
        if cfg.moe.get('gate_noise', None) is not None:
            gate_type['gate_noise'] = cfg.moe.get('gate_noise')

        experts = {
            'count_per_node': num_local_experts,
            'type': cfg.moe.get('experts_type', 'ffn'),
            'hidden_size_per_expert': cfg.mlp_ratio * cfg.d_model,
            'activation_fn': lambda x: F.gelu(x, approximate='none')}

        rank_seed = torch.initial_seed() + dist.get_global_rank()
        gpu_devices = [torch.cuda.current_device()]
        # guarentee init seeds are different for all devices
        # SeedContextManager sets the given seed, then restore local seed on ctx mgr exit
        with SeedContextManager(gpu_devices=gpu_devices, seed=rank_seed) as s_ctx_mgr:
            self.moe = tutel_moe.moe_layer(
                gate_type=gate_type,
                model_dim=cfg.d_model,
                experts=experts,
                scan_expert_func=lambda name, param: setattr(param, 'moe_expert', True),
                result_func=cfg.moe.get('result_func', None),
                group=cfg.moe.get('group', None),
                seeds=(rank_seed, rank_seed + 1, rank_seed + 2),
                a2a_ffn_overlap_degree=cfg.moe.get('a2a_ffn_overlap_degree', 1),
                is_postscore=cfg.moe.get('is_postscore', True),
                batch_prioritized_routing=cfg.moe.get('batch_prioritized_routing', False),
                normalize_gate=cfg.moe.get('normalize_gate', True),
                is_gshard_loss=cfg.moe.get('is_gshard_loss', True),
                parallel_type=cfg.moe.get('parallel_type', 'auto'),
                use_2dh=cfg.moe.get('use_2dh', False),
                device=device,
            )

        self.moe.experts.batched_fc2_w._is_residual = True  # type: ignore

    @classmethod
    def param_init_fn(cls, module, cfg):
        # Param Initialization, needed for device='meta' fast initialization
        if isinstance(module, MOELayer):
            # set buffer
            local_expert = -module.sharded_count if module.sharded_count > 1 else module.num_local_experts
            module._num_global_experts = torch.tensor(module.global_expert_count(local_expert, module.group))

        if isinstance(module, FusedExpertsNetwork):
            # guarentee init seeds are different for all devices
            # SeedContextManager sets the given seed, then restore local when finished
            rank_seed = torch.initial_seed() + dist.get_global_rank()
            tensors = module.batched_fc1_w, module.batched_fc2_w
            with SeedContextManager(*tensors, seed=rank_seed) as s_ctx_mgr:
                if cfg.get('init_experts', False):
                    # init fc1
                    module.batched_fc1_w.normal_(
                        mean=0.0,
                        std=cfg.init_std
                    )
                    torch.nn.init.zeros_(module.batched_fc1_bias)

                    # init fc2
                    module.batched_fc2_w.normal_(
                        mean=0.0,
                        std=(cfg.init_std / math.sqrt(2 * cfg.n_layers)))
                    torch.nn.init.zeros_(module.batched_fc2_bias)
                else:
                    module.reset_parameters()

        if isinstance(module, LinearTopKGate):
            # Linear init handeled by MosaicLLMs nn.Linear
            pass

        if isinstance(module, CosineTopKGate):
            # Linear handeled by nn.Linear
            raise NotImplementedError('Implement init of buffers')

    def forward(self, x):
        return self.moe(x)
