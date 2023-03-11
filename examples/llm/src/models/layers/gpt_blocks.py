# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

"""GPT Blocks used for the GPT Model."""

from typing import Optional

import torch
import torch.nn as nn
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

    def forward(self, x: torch.Tensor):
        return self.mlp_down(self.mlp_act(self.mlp_up(x)))


class GPTBlock(nn.Module):

    def __init__(self,
                 cfg: DictConfig,
                 causal_attn_cls: nn.Module,
                 device: Optional[str] = None):
        super().__init__()
        if cfg.get('alibi', False):
            assert cfg.attn_impl == 'triton' or cfg.attn_impl == 'torch', 'Only triton kernel or torch supports alibi'
        self.ln_1 = nn.LayerNorm(cfg.d_model, device=device)
        self.causal_attn = causal_attn_cls(cfg, device)
        self.ln_2 = nn.LayerNorm(cfg.d_model, device=device)
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


class GPTBlockPostLN(nn.Module):

    def __init__(self,
                 cfg: DictConfig,
                 causal_attn_cls: nn.Module,
                 device: Optional[str] = None):
        super().__init__()
        if cfg.get('alibi', False):
            assert cfg.attn_impl == 'triton' or cfg.attn_impl == 'torch', 'Only triton kernel or torch supports alibi'
        self.ln_1 = nn.LayerNorm(cfg.d_model, device=device)
        self.causal_attn = causal_attn_cls(cfg, device)
        self.ln_2 = nn.LayerNorm(cfg.d_model, device=device)
        self.mlp = GPTMLP(cfg, device=device)
        self.resid_attn_dropout = nn.Dropout(cfg.resid_pdrop)
        self.resid_mlp_dropout = nn.Dropout(cfg.resid_pdrop)
        if cfg.get('param_init_fn', 'default_') == 'deepnorm_':
            # constant for decoder-only models
            # see the paper DeepNet, https://arxiv.org/abs/2203.00555
            self.residual_scale = (2 * cfg.n_layers)**0.25
        else:
            self.residual_scale = 1.0

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.ByteTensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        a, _ = self.causal_attn(x, key_padding_mask, attn_mask)
        residue = x * self.residual_scale
        x = self.ln_1(residue + self.resid_attn_dropout(a))
        residue = x * self.residual_scale
        x = self.ln_2(residue + self.resid_mlp_dropout(self.mlp(x)))
        return x
