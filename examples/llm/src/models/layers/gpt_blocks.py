# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

"""GPT Blocks used for the GPT Model."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from composer.algorithms.low_precision_layernorm.low_precision_layernorm import \
    LPLayerNorm
from omegaconf import DictConfig

from examples.llm.src.models.ops import (check_if_dropout_layer_norm_installed,
                                         check_if_fused_mlp_installed)

try:
    from flash_attn.ops.fused_dense import FusedMLP
    from flash_attn.ops.layer_norm import DropoutAddLayerNorm
    CUDA_OPTIMIZATIONS_INSTALLED = True
except ImportError as e:
    CUDA_OPTIMIZATIONS_INSTALLED = False


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
                 device: Optional[str] = None):
        super().__init__()
        if cfg.get('alibi', False):
            if not (cfg.attn_impl == 'triton' or cfg.attn_impl == 'torch'):
                raise ValueError(f'Only triton kernel or torch supports alibi')

        layernorm_class = LPLayerNorm if cfg.get('low_precision_layernorm',
                                                 False) else nn.LayerNorm
        self.ln_1 = layernorm_class(cfg.d_model, device=device)
        self.causal_attn = causal_attn_cls(cfg, device)
        self.ln_2 = layernorm_class(cfg.d_model, device=device)
        self.mlp = GPTMLP(cfg, device=device)
        self.resid_attn_dropout = nn.Dropout(cfg.resid_pdrop)
        self.resid_mlp_dropout = nn.Dropout(cfg.resid_pdrop)

    def forward(
        self,
        a: torch.Tensor,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.ByteTensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b, _ = self.causal_attn(a, key_padding_mask, attn_mask)
        x = x + self.resid_attn_dropout(b)
        m = self.ln_1(x)
        n = self.mlp(m)
        x = x + self.resid_mlp_dropout(n)
        a = self.ln_2(x)
        return a, x


class OptimizedGPTBlock(nn.Module):

    def __init__(self,
                 cfg: DictConfig,
                 causal_attn_cls,
                 device: Optional[str] = None):
        super().__init__()
        if cfg.get('alibi', False):
            if not (cfg.attn_impl == 'triton' or cfg.attn_impl == 'torch'):
                raise ValueError(f'Only triton kernel or torch supports alibi')
        if not CUDA_OPTIMIZATIONS_INSTALLED:
            raise ImportError(
                'The CUDA optimizations required for the Optimized GPT Block were not installed. Please install them from examples/llm/requirements_optimized_perf.txt and the examples/llm/csrc folder.'
            )
        self.causal_attn = causal_attn_cls(cfg, device)
        self.dropout_add_ln_1 = DropoutAddLayerNorm(cfg.d_model,
                                                    prenorm=True,
                                                    p=cfg.resid_pdrop,
                                                    device=device)
        self.mlp = FusedMLP(in_features=cfg.d_model,
                            hidden_features=cfg.mlp_ratio * cfg.d_model,
                            out_features=cfg.d_model,
                            device=device)
        self.mlp.fc2._is_residual = True  # type: ignore
        self.dropout_add_ln_2 = DropoutAddLayerNorm(cfg.d_model,
                                                    prenorm=True,
                                                    p=cfg.resid_pdrop,
                                                    device=device)

    def forward(
        self,
        a: torch.Tensor,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.ByteTensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b, _ = self.causal_attn(a, key_padding_mask, attn_mask)
        m, x = self.dropout_add_ln_1(b, x)
        n = self.mlp(m)
        a, x = self.dropout_add_ln_2(n, x)
        return a, x
