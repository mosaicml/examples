# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

"""GPT Blocks used for the GPT Model."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from composer.algorithms.low_precision_layernorm.low_precision_layernorm import \
    LPLayerNorm
from omegaconf import DictConfig

from examples.llm.src.models.layers.attention import MultiheadAttention

try:
    from flash_attn.ops.layer_norm import DropoutAddLayerNorm  # type: ignore
    DROPOUT_ADD_LAYERNORM_INSTALLED = True
except ImportError as e:
    DROPOUT_ADD_LAYERNORM_INSTALLED = False


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

    def __init__(self, cfg: DictConfig, device: Optional[str] = None):
        super().__init__()
        layernorm_class = LPLayerNorm if cfg.get('low_precision_layernorm',
                                                 False) else nn.LayerNorm
        self.ln_1 = layernorm_class(cfg.d_model, device=device)
        self.attn = MultiheadAttention(cfg, device)
        self.ln_2 = layernorm_class(cfg.d_model, device=device)
        self.mlp = GPTMLP(cfg, device=device)
        self.resid_attn_dropout = nn.Dropout(cfg.resid_pdrop)
        self.resid_mlp_dropout = nn.Dropout(cfg.resid_pdrop)

    def forward(
        self,
        a: torch.Tensor,
        x: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attn_bias: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.ByteTensor] = None,
        is_causal: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        b, _, past_key_value = self.attn(a,
                                         past_key_value=past_key_value,
                                         attn_bias=attn_bias,
                                         key_padding_mask=key_padding_mask,
                                         is_causal=is_causal)
        x = x + self.resid_attn_dropout(b)
        m = self.ln_1(x)
        n = self.mlp(m)
        x = x + self.resid_mlp_dropout(n)
        a = self.ln_2(x)
        return a, x, past_key_value


class OptimizedGPTBlock(nn.Module):

    def __init__(self, cfg: DictConfig, device: Optional[str] = None):
        super().__init__()

        if not DROPOUT_ADD_LAYERNORM_INSTALLED:
            raise ImportError(
                'A CUDA optimization required for the Optimized GPT Block were not installed. Please install it from the examples/llm/csrc folder.'
            )

        self.attn = MultiheadAttention(cfg, device)
        self.dropout_add_ln_1 = DropoutAddLayerNorm(cfg.d_model,
                                                    prenorm=True,
                                                    p=cfg.resid_pdrop,
                                                    residual_in_fp32=False,
                                                    device=device)
        self.mlp = GPTMLP(cfg, device=device)
        self.dropout_add_ln_2 = DropoutAddLayerNorm(cfg.d_model,
                                                    prenorm=True,
                                                    p=cfg.resid_pdrop,
                                                    residual_in_fp32=False,
                                                    device=device)

    def forward(
        self,
        a: torch.Tensor,
        x: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attn_bias: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.ByteTensor] = None,
        is_causal: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        b, _, past_key_value = self.attn(a,
                                         past_key_value=past_key_value,
                                         attn_bias=attn_bias,
                                         key_padding_mask=key_padding_mask,
                                         is_causal=is_causal)
        m, x = self.dropout_add_ln_1(b, x)
        n = self.mlp(m)
        a, x = self.dropout_add_ln_2(n, x)
        return a, x, past_key_value
