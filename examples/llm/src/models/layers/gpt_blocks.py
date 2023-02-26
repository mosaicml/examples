# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

"""GPT Blocks used for the GPT Model."""

from collections.abc import Sequence
from typing import Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig

from examples.llm.src.models.layers.moe.tutel_moe import TutelMOE


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


class GPTBlock(nn.Module):

    def __init__(self,
                 cfg: DictConfig,
                 causal_attn_cls,
                 block_idx: int,
                 device: Optional[str] = None):
        super().__init__()
        if cfg.get('alibi', False):
            assert cfg.attn_impl == 'triton' or cfg.attn_impl == 'torch', 'Only triton kernel or torch supports alibi'
        self.ln_1 = nn.LayerNorm(cfg.d_model, device=device)
        self.causal_attn = causal_attn_cls(cfg, device)
        self.ln_2 = nn.LayerNorm(cfg.d_model, device=device)

        use_moe = False
        if cfg.get('moe', None):
            num_experts = cfg.moe.get('num_experts')
            if isinstance(num_experts, Sequence):
                if len(num_experts) != cfg.n_layers:
                    raise ValueError(
                        f'If using PyramidMoEs, each layer ({cfg.n_layers=}) should have an associated expert count ({len(num_experts)=})'
                    )
                num_experts = num_experts[block_idx]
            use_moe = True if num_experts > 1 else False

        if use_moe:
            moe_type = cfg.moe.get('moe_type')
            moe_cls = None
            if moe_type == 'tutel':
                moe_cls = TutelMOE
            else:
                raise NotImplementedError(f'{moe_type=} not integrated')
            self.mlp = moe_cls(cfg, block_idx, device=device)
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
