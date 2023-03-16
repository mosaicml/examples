# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

"""Attention layers for the GPT models."""

import math
import warnings
from typing import Optional

import torch
from einops import rearrange
from omegaconf import DictConfig
from torch import nn


def scaled_multihead_dot_product_attention(
    query,
    key,
    value,
    n_heads,
    softmax_scale=None,
    attn_bias=None,
    key_padding_mask=None,
    is_causal=False,
    dropout_p=0.0,
    training=False,
    needs_weights=False,
):
    q = rearrange(query, 'b s (h d) -> b h s d', h=n_heads)
    k = rearrange(key, 'b s (h d) -> b h d s', h=n_heads)  # includes key.t()
    v = rearrange(value, 'b s (h d) -> b h s d', h=n_heads)

    b, _, s_q, d = q.shape
    s_k = k.size(-1)

    if softmax_scale is None:
        softmax_scale = 1 / math.sqrt(d)

    attn_weight = q.matmul(k)
    attn_weight *= softmax_scale

    if attn_bias is not None:
        if attn_bias.size(-1) != s_k or attn_bias.size(-2) != s_q:
            raise RuntimeError(
                f'attn_bias (shape: {attn_bias.shape}) is expected to broadcast to shape: {attn_weight.shape}.'
            )
        attn_weight = attn_weight + attn_bias

    if key_padding_mask is not None:
        attn_weight.masked_fill_(~key_padding_mask.view((b, 1, 1, s_k)),
                                 -float('inf'))

    if is_causal:
        s = max(s_q, s_k)
        causal_mask = attn_weight.new_ones(s, s, dtype=torch.bool)
        causal_mask.tril_()
        causal_mask.logical_not_()
        causal_mask = causal_mask[:s_q, :s_k]
        attn_weight.masked_fill_(causal_mask.view(1, 1, s_q, s_k),
                                 -float('inf'))

    attn_weight = torch.softmax(attn_weight, dim=-1)

    if dropout_p:
        attn_weight = torch.nn.functional.dropout(attn_weight,
                                                  p=dropout_p,
                                                  training=training,
                                                  inplace=True)

    out = attn_weight.matmul(v)
    out = rearrange(out, 'b h s d -> b s (h d)', h=n_heads)

    if needs_weights:
        return out, attn_weight
    return out, None


def scaled_multihead_dot_product_self_attention(
    qkv,
    n_heads,
    softmax_scale=None,
    attn_bias=None,
    key_padding_mask=None,
    is_causal=False,
    dropout_p=0.0,
    training=False,
    needs_weights=False,
):
    qkv = rearrange(qkv, 'b s (t hd) -> b s t hd', t=3)
    return scaled_multihead_dot_product_attention(
        qkv[:, :, 0],
        qkv[:, :, 1],
        qkv[:, :, 2],
        n_heads,
        softmax_scale=softmax_scale,
        attn_bias=attn_bias,
        key_padding_mask=key_padding_mask,
        is_causal=is_causal,
        dropout_p=dropout_p,
        training=training,
        needs_weights=needs_weights,
    )


class MultiheadAttention(nn.Module):
    """Multi-head self attention.

    Using torch or triton attention implemetation enables user to also use
    additive bias.
    """

    def __init__(self, cfg: DictConfig, device: Optional[str] = None):
        super().__init__()
        self.attn_impl = cfg.get('attn_impl')

        self.clip_qkv = cfg.get('attn_clip_qkv')
        self.attn_qk_ln = cfg.get('attn_qk_ln')

        self.d_model = cfg.d_model
        self.n_heads = cfg.n_heads
        self.softmax_scale = cfg.get('softmax_scale')
        if self.softmax_scale is None:
            self.softmax_scale = 1 / math.sqrt(self.d_model / self.n_heads)
        self.attn_dropout_p = cfg.get('attn_pdrop')

        self.Wqkv = nn.Linear(self.d_model,
                              3 * self.d_model,
                              bias=True,
                              device=device)
        # for param init fn; enables shape based init of fused layers
        fuse_splits = (cfg.d_model, 2 * cfg.d_model)
        self.Wqkv._fused = (0, fuse_splits)  # type: ignore

        if self.attn_qk_ln:
            self.q_ln = nn.LayerNorm(self.d_model, device=device)
            self.k_ln = nn.LayerNorm(self.d_model, device=device)

        self.is_causal = True  # causal attn impl
        if self.attn_impl == 'flash':
            try:
                from flash_attn import flash_attention  # type: ignore
            except ImportError as e:
                raise e

            self.inner_attn = flash_attention.FlashAttention(  # type: ignore
                attention_dropout=self.attn_dropout_p,
                device=device,
                softmax_scale=self.softmax_scale)
        elif self.attn_impl == 'triton':
            if self.attn_dropout_p:
                raise ValueError(
                    'Triton kernel does not support attention dropout.')

            try:
                from examples.llm.src.models import layers  # type: ignore
            except ImportError as e:
                raise e

            warnings.warn(
                'While `attn_impl: triton` can be faster than `attn_impl: flash` '
                'it uses more memory. When training larger models this can trigger '
                'alloc retries which hurts performance. If encountered, we recommend '
                'using `attn_impl: flash`.')

            self.inner_attn = layers.flash_attention.FlashAttention(  # type: ignore
                num_heads=cfg.n_heads,
                softmax_scale=self.softmax_scale,
                device=device)

        elif self.attn_impl == 'torch':
            pass

        else:
            raise ValueError(f"{cfg.get('attn_impl')=} is an invalid setting.")

        self.out_proj = nn.Linear(self.d_model,
                                  self.d_model,
                                  bias=True,
                                  device=device)
        self.out_proj._is_residual = True  # type: ignore

    def forward(self,
                x,
                attn_bias=None,
                key_padding_mask=None,
                needs_weights=False):
        qkv = self.Wqkv(x)
        if self.clip_qkv:
            qkv.clamp_(min=-self.clip_qkv, max=self.clip_qkv)
        if self.attn_qk_ln:
            # Applying layernorm to qk
            dtype = qkv.dtype
            q, k, v = qkv.split(self.d_model, dim=-1)
            q = self.q_ln(q).to(dtype)
            k = self.k_ln(k).to(dtype)
            qkv = torch.cat([q, k, v], dim=-1)

        # attention
        if self.attn_impl == 'flash':
            if needs_weights:
                raise NotImplementedError(
                    'FlashAttn cannot return attention weights.')
            if attn_bias is not None:
                raise RuntimeError(
                    'attn_impl: flash does not support an attn bias.')
            qkv = rearrange(qkv,
                            'b s (t h d) -> b s t h d',
                            t=3,
                            h=self.n_heads)

            context, attn_weights = self.inner_attn(
                qkv,
                key_padding_mask=key_padding_mask,
                causal=self.is_causal,
                need_weights=False)
            context = rearrange(context, 'b s h d -> b s (h d)')

        elif self.attn_impl == 'triton':
            if needs_weights:
                raise NotImplementedError(
                    'Triton variant of FlashAttn cannot return attention weigths.'
                )
            context, attn_weights = self.inner_attn(
                qkv,
                key_padding_mask=key_padding_mask,
                attn_bias=attn_bias,
                is_causal=self.is_causal)

        elif self.attn_impl == 'torch':
            context, attn_weights = scaled_multihead_dot_product_self_attention(
                qkv,
                self.n_heads,
                softmax_scale=self.softmax_scale,
                attn_bias=attn_bias,
                key_padding_mask=key_padding_mask,
                is_causal=self.is_causal,
                dropout_p=self.attn_dropout_p,
                training=self.training,
                needs_weights=needs_weights,
            )
        else:
            raise RuntimeError('Internal Logic Error')

        return self.out_proj(context), attn_weights


def attn_bias_shape(attn_impl, n_heads, seq_len, alibi):
    if attn_impl == 'flash':
        return None
    elif attn_impl == 'triton':
        # in triton, key_padding_mask must be integrated into attn_bias
        return (1, n_heads, 1, seq_len) if alibi else (1, 1, 1, seq_len)
    elif attn_impl == 'torch':
        return (1, n_heads, 1, seq_len) if alibi else None
    else:
        raise ValueError(f'{attn_impl=} is an invalid setting.')


def attn_bias_(attn_impl,
               attn_bias,
               n_heads,
               seq_len,
               alibi=False,
               alibi_bias_max=8):
    if attn_impl == 'flash':
        return None
    elif attn_impl == 'triton':
        attn_bias.zero_()
        if alibi:
            # in place add alibi to attn bias
            device, dtype = attn_bias.device, attn_bias.dtype
            attn_bias.add_(
                alibi_bias(n_heads,
                           seq_len,
                           full=False,
                           alibi_bias_max=alibi_bias_max,
                           device=device,
                           dtype=dtype))
        return attn_bias
    elif attn_impl == 'torch':
        if attn_bias is not None:
            attn_bias.zero_()
            if alibi:
                # in place add alibi to attn bias
                device, dtype = attn_bias.device, attn_bias.dtype
                attn_bias.add_(
                    alibi_bias(n_heads,
                               seq_len,
                               full=False,
                               alibi_bias_max=alibi_bias_max,
                               device=device,
                               dtype=dtype))
            return attn_bias
    else:
        raise ValueError(f'{attn_impl=} is an invalid setting.')


def generate_attn_bias(attn_impl,
                       attn_bias,
                       seq_len,
                       batch_size,
                       key_padding_mask=None):
    if attn_bias is not None:
        # select seq_len subset of attn mask
        attn_bias = attn_bias[..., :seq_len, :seq_len]

    if attn_impl == 'triton' and key_padding_mask is not None:
        attn_bias = attn_bias.expand(batch_size, -1, -1, -1)
        attn_bias.masked_fill(
            ~key_padding_mask.view((batch_size, 1, 1, seq_len)), -float('inf'))

    return attn_bias


def alibi_bias(n_heads,
               seq_len,
               full=False,
               alibi_bias_max=8,
               device=None,
               dtype=None):
    alibi_bias = torch.arange(1 - seq_len, 1, dtype=dtype,
                              device=device).view(1, 1, 1, seq_len)
    if full:
        # generate 1 x Heads x SeqLen x SeqLen alibi bias mask
        # otherwise the mask is 1 x Heads x 1 x SeqLen (which is braodcasted up to the approproate size)
        alibi_bias = alibi_bias - torch.arange(
            1 - seq_len, 1, dtype=dtype, device=device).view(1, 1, seq_len, 1)
        alibi_bias.abs_().mul_(
            -1
        )  # since we're using causal flag, this isn't really needed, but why not include it

    m = torch.arange(1, n_heads + 1, dtype=dtype, device=device)
    m.mul_(alibi_bias_max / n_heads)
    alibi_bias = alibi_bias * (1. / (2**m.view(1, n_heads, 1, 1)))
    return alibi_bias
