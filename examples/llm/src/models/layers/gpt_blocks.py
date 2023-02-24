# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

"""GPT Blocks used for the GPT Model."""

from typing import Optional

import torch
import torch.nn as nn
from composer.utils.dist import (get_local_rank, get_local_world_size,
                                 get_world_size)
from omegaconf import DictConfig


class GPTMLP(nn.Module):

    def __init__(self, cfg: DictConfig, device: Optional[str] = None):
        super().__init__()
        self.mlp_up = nn.Linear(cfg.d_model,
                                cfg.mlp_ratio * cfg.d_model,
                                device=device)
        self.mlp_act = nn.GELU(approximate='none')
        self.mlp_down = nn.Linear(cfg.mlp_ratio * cfg.d_model,
                                  cfg.d_model,
                                  device=device)
        self.mlp_down._is_residual = True  # type: ignore

    def forward(self, x):
        return self.mlp_down(self.mlp_act(self.mlp_up(x)))


class GPTFakeTPMLP(nn.Module):

    def __init__(self, cfg: DictConfig, device: Optional[str] = None):
        super().__init__()
        local_rank = get_local_rank()
        local_world_size = get_local_world_size()
        world_size = get_world_size()
        # self.tp_world_size = cfg.get('tp_mlp_group_size', local_world_size)
        # if world_size % self.tp_world_size != 0:
        #     raise RuntimeError
        # fsdp_process_group = [
        #     local_rank + self.tp_world_size * i
        #     for i in range(0, world_size // self.tp_world_size)
        # ]

        # self._fsdp_wrap = {'process_group': fsdp_process_group}

        self.tp_world_size = local_world_size
        self._fsdp_wrap = {'process_group': 'local_rank_across_nodes'}
        if cfg.get('tp_mlp_sharding_strategy') is not None:
            self._fsdp_wrap['sharding_strategy'] = cfg.tp_mlp_sharding_strategy

        mlp_inner = cfg.mlp_ratio * cfg.d_model // self.tp_world_size
        self.mlp_up = nn.Linear(cfg.d_model, mlp_inner, device=device)
        self.mlp_act = nn.GELU(approximate='none')
        self.mlp_down = nn.Linear(mlp_inner, cfg.d_model, device=device)
        self.mlp_down._is_residual = True  # type: ignore

    def forward(self, x):
        sizes = [self.tp_world_size] + [-1] * x.ndim
        fake_all_gather_x = x.expand(sizes)
        out = self.mlp_down(self.mlp_act(self.mlp_up(fake_all_gather_x)))
        fake_reduce_scatter_out = out.sum(axis=0)
        return fake_reduce_scatter_out


class GPTBlock(nn.Module):

    def __init__(self,
                 cfg: DictConfig,
                 causal_attn_cls,
                 device: Optional[str] = None):
        super().__init__()
        if cfg.get('alibi', False):
            assert cfg.attn_impl == 'triton' or cfg.attn_impl == 'torch', 'Only triton kernel or torch supports alibi'
        self.ln_1 = nn.LayerNorm(cfg.d_model, device=device)
        self.causal_attn = causal_attn_cls(cfg, device)
        self.ln_2 = nn.LayerNorm(cfg.d_model, device=device)
        if cfg.get('tp_mlp', False):
            self.mlp = GPTFakeTPMLP(cfg, device=device)
        else:
            self.mlp = GPTMLP(cfg, device=device)
        self.resid_attn_dropout = nn.Dropout(cfg.resid_pdrop)
        self.resid_mlp_dropout = nn.Dropout(cfg.resid_pdrop)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.ByteTensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        a = self.ln_1(x)
        b, _ = self.causal_attn(a, key_padding_mask, attn_mask)
        x = x + self.resid_attn_dropout(b)
        m = self.ln_2(x)
        n = self.mlp(m)
        x = x + self.resid_mlp_dropout(n)
        return x
