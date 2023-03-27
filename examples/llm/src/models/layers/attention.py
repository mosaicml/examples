# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

"""Attention layers for the GPT models."""

import warnings
from typing import Optional

import torch
import torch.nn as nn
from composer.algorithms.low_precision_layernorm.low_precision_layernorm import \
    LPLayerNorm
from einops import rearrange
from omegaconf import DictConfig


class TorchCausalAttention(nn.Module):

    def __init__(self, cfg: DictConfig, device: Optional[str] = None):
        super().__init__()
        self.mhsa = nn.MultiheadAttention(
            embed_dim=cfg.d_model,
            num_heads=cfg.n_heads,
            dropout=cfg.attn_pdrop,
            bias=True,
            batch_first=True,
            device=device,
        )
        self.mhsa.out_proj._is_residual = True  # type: ignore

        warnings.warn(
            DeprecationWarning(
                'Using `attn_impl: torch` is deprecated; recommened using `attn_impl: flash`.'
            ))

    def forward(self, x, key_padding_mask, attn_mask=None):
        if key_padding_mask is not None:
            key_padding_mask = ~key_padding_mask
        return self.mhsa(x,
                         x,
                         x,
                         attn_mask=attn_mask,
                         key_padding_mask=key_padding_mask,
                         need_weights=True)

    @staticmethod
    def mask_shape(n_heads, seq_len, alibi):
        if alibi:
            return (n_heads, seq_len, seq_len)
        return (seq_len, seq_len)

    @staticmethod
    def attn_mask_(attn_mask, n_heads, seq_len, alibi=False, alibi_bias_max=8):
        # in-place fill causal attn mask
        #
        # Two important disclaimers
        # 1. Torch uses additive attention. If your attn_mask/key_padding mask is a float tensor, it will add the floats
        #   directly to your attention matrix. If they are boolean masks, True will be converted to -inf before adding the
        #   mask to your attentions. See https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html#torch.nn.MultiheadAttention.forward
        #   Basically True/-inf indicates tokens we do not want to attend to.
        #
        # 2. This is is the exact opposite behavior of Huggingface's tokenizers, which use the convention that True denotes tokens
        #   we do want to attend to. See https://huggingface.co/docs/transformers/glossary#attention-mask
        attn_mask.fill_(float('-inf'))
        # attn_mask.triu_(diagonal=1)  # triu_ is not implemented for cuda bf16
        # TODO: revert back to triu_ when torch supports triu_ for cuda bf16
        attn_mask.masked_fill_(attn_mask.to(bool).fill_(1).tril_(), 0.)

        if alibi:
            device, dtype = attn_mask.device, attn_mask.dtype
            a_bias = alibi_bias(n_heads,
                                seq_len,
                                full=True,
                                alibi_bias_max=alibi_bias_max,
                                device=device,
                                dtype=dtype)
            attn_mask.add_(a_bias.squeeze())

        return attn_mask

    @staticmethod
    def generate_attn_mask(
        attn_mask,
        batch_size,
        heads,
        seq_len,
        prefix_mask=None,
        sequence_id=None,
        key_padding_mask=None,
        alibi=False,
        dtype=None,
    ):

        if prefix_mask is not None:
            raise NotImplementedError(
                'This attention implementation does not support use of `prefix_lm`.'
            )
        if sequence_id is not None:
            raise NotImplementedError(
                'This attention implementation not support use of `sequence_id`... yet.'
            )

        # select seq_len subset of attn mask
        attn_mask = attn_mask[..., :seq_len, :seq_len]

        if key_padding_mask is not None and _check_apply_key_padding_mask(
                key_padding_mask):
            attn_mask = attn_mask.expand(batch_size, heads, seq_len,
                                         seq_len).clone()

            kpm_fill_value = -1e4  # numerically stable -inf
            attn_mask.masked_fill_(
                ~key_padding_mask.view(batch_size, 1, 1, seq_len),
                kpm_fill_value)
            attn_mask.masked_fill_(
                ~key_padding_mask.view(batch_size, 1, seq_len, 1),
                kpm_fill_value)
            attn_mask = attn_mask.reshape(-1, seq_len, seq_len)
        elif alibi:
            # WARNING: Alibi with torch attn is not thoroughly tested
            # torch mask is supposed to be of shape nzz x SeqLen x SeqLen
            # we must braodcast to batch size then flatten batchsize * n_heads dim
            # Note: if key_padding_mask is triggered, the needed expansion is already done.
            attn_mask = attn_mask.expand(batch_size, heads, seq_len,
                                         seq_len).reshape(-1, seq_len, seq_len)

        return attn_mask


class FlashCausalAttention(nn.Module):

    def __init__(self, cfg: DictConfig, device: Optional[str] = None):
        super().__init__()
        try:
            from flash_attn.flash_attention import (  # type: ignore
                FlashAttention, FlashMHA)
        except ImportError as e:
            raise e

        self.clip_qkv = cfg.get('attn_clip_qkv')
        self.attn_qk_ln = cfg.get('attn_qk_ln')
        self.softmax_scale = cfg.get('softmax_scale')
        self.d_model = cfg.d_model
        self.n_heads = cfg.n_heads

        if self.attn_qk_ln or self.clip_qkv or self.softmax_scale:
            self.W_qkv = nn.Linear(self.d_model,
                                   3 * self.d_model,
                                   bias=True,
                                   device=device)
            self.inner_attn = FlashAttention(attention_dropout=cfg.attn_pdrop,
                                             device=device,
                                             softmax_scale=self.softmax_scale)
            self.out_proj = nn.Linear(self.d_model,
                                      self.d_model,
                                      bias=True,
                                      device=device)
            # for param init fn
            fuse_splits = (cfg.d_model, 2 * cfg.d_model)
            self.W_qkv._fused = (0, fuse_splits)  # type: ignore
            self.out_proj._is_residual = True  # type: ignore

            if self.attn_qk_ln:
                layernorm_class = nn.LayerNorm if not cfg.get(
                    'low_precision_layernorm', False) else LPLayerNorm
                self.q_ln = layernorm_class(self.d_model, device=device)
                self.k_ln = layernorm_class(self.d_model, device=device)
        else:
            self.mhsa = FlashMHA(
                embed_dim=cfg.d_model,
                num_heads=cfg.n_heads,
                attention_dropout=cfg.attn_pdrop,
                bias=True,
                batch_first=True,
                causal=True,
                device=device,
            )
            # for param init fn
            fuse_splits = (cfg.d_model, 2 * cfg.d_model)
            self.mhsa.Wqkv._fused = (0, fuse_splits)  # type: ignore
            self.mhsa.out_proj._is_residual = True

    def forward(self, x, key_padding_mask, attn_mask=None):
        assert attn_mask is None

        if self.attn_qk_ln or self.clip_qkv or self.softmax_scale:
            qkv = self.W_qkv(x)
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
            qkv = rearrange(qkv,
                            'b s (three h d) -> b s three h d',
                            three=3,
                            h=self.n_heads)

            context, attn_weights = self.inner_attn(
                qkv,
                key_padding_mask=key_padding_mask,
                causal=True,
                need_weights=False)
            return self.out_proj(rearrange(
                context, 'b s h d -> b s (h d)')), attn_weights

        else:
            return self.mhsa(x,
                             key_padding_mask=key_padding_mask,
                             need_weights=False)

    @staticmethod
    def mask_shape(*args, **kwargs):
        return None

    @staticmethod
    def attn_mask_(*args, **kwargs):
        return None

    @staticmethod
    def generate_attn_mask(
        attn_mask,
        batch_size,
        heads,
        seq_len,
        prefix_mask=None,
        sequence_id=None,
        key_padding_mask=None,
        alibi=False,
        dtype=None,
    ):
        if prefix_mask is not None:
            raise NotImplementedError(
                'This attention implementation does not support use of `prefix_lm`.'
            )

        return attn_mask  # None


class TritonFlashAttention(nn.Module):
    """Multi-headed self attention using triton FlashAttn kernel.

    This also includes bias for Alibi integration or Prefix LM attention
    (see :class:`TritonFlashPrefixAttention` for the latter).

    Don't invoke this class directly! Use `TritonFlashCausalAttention`
    or `TritonFlashPrefixAttention`.
    """
    causal = None

    def __init__(self, cfg: DictConfig, device: Optional[str] = None):
        super().__init__()
        try:
            from examples.llm.src.models.layers.flash_attention import (  # type: ignore
                FlashAttention, FlashMHA)
        except ImportError as e:
            raise e

        assert cfg.attn_pdrop == 0, 'triton kernel does not support attn_dropout'

        self.clip_qkv = cfg.get('attn_clip_qkv')
        self.attn_qk_ln = cfg.get('attn_qk_ln')
        self.d_model = cfg.d_model
        self.n_heads = cfg.n_heads

        if self.attn_qk_ln or self.clip_qkv:
            self.Wqkv = nn.Linear(self.d_model,
                                  3 * self.d_model,
                                  bias=True,
                                  device=device)
            self.inner_attn = FlashAttention(
                num_heads=cfg.n_heads,
                softmax_scale=cfg.get('softmax_scale'),
                device=device)
            self.out_proj = nn.Linear(self.d_model,
                                      self.d_model,
                                      bias=True,
                                      device=device)
            # for param init fn
            fuse_splits = (cfg.d_model, 2 * cfg.d_model)
            self.Wqkv._fused = (0, fuse_splits)  # type: ignore
            self.out_proj._is_residual = True  # type: ignore

            if self.attn_qk_ln:
                layernorm_class = nn.LayerNorm if not cfg.get(
                    'low_precision_layernorm', False) else LPLayerNorm
                self.q_ln = layernorm_class(self.d_model, device=device)
                self.k_ln = layernorm_class(self.d_model, device=device)
        else:
            self.mhsa = FlashMHA(
                embed_dim=cfg.d_model,
                num_heads=cfg.n_heads,
                bias=True,
                batch_first=True,
                causal=self.causal,  # type: ignore
                softmax_scale=cfg.get('softmax_scale'),
                device=device,
            )
            # for param init fn
            fuse_splits = (cfg.d_model, 2 * cfg.d_model)
            self.mhsa.Wqkv._fused = (0, fuse_splits)  # type: ignore
            self.mhsa.out_proj._is_residual = True  # type: ignore

        warnings.warn(
            'While `attn_impl: triton` can be faster than `attn_impl: flash` '
            'it uses more memory. When training larger models this can trigger '
            'alloc retries which hurts performance. If encountered, we recommend '
            'using `attn_impl: flash`.')

    def forward(self, x, key_padding_mask=None, attn_mask=None):
        if self.attn_qk_ln or self.clip_qkv:
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
            context, attn_weights = self.inner_attn(
                qkv,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
                is_causal=self.causal)

            return self.out_proj(context), attn_weights

        else:
            return self.mhsa(x,
                             key_padding_mask=None,
                             attn_mask=attn_mask,
                             need_weights=False)

    @classmethod
    def mask_shape(cls, n_heads, seq_len, alibi):
        if cls.causal:
            return (1, n_heads, 1, seq_len) if alibi else None
        return (1, n_heads, seq_len, seq_len) if alibi else None

    @classmethod
    def attn_mask_(cls,
                   attn_mask,
                   n_heads,
                   seq_len,
                   alibi=False,
                   alibi_bias_max=8):
        if attn_mask is not None:
            # in-place fill causal attn mask
            attn_mask.zero_()

            if alibi:
                device, dtype = attn_mask.device, attn_mask.dtype
                attn_mask.add_(
                    alibi_bias(n_heads,
                               seq_len,
                               full=not cls.causal,
                               alibi_bias_max=alibi_bias_max,
                               device=device,
                               dtype=dtype))

        return attn_mask

    @staticmethod
    def generate_attn_mask(*args, **kwargs):
        # To be implemented by child classes
        raise NotImplementedError


class TritonFlashCausalAttention(TritonFlashAttention):
    """Triton FlashAttn kernel for a Causal LM.

    See parent class :class:`TritonFlashAttention` for details.
    """
    causal = True

    @staticmethod
    def generate_attn_mask(
        attn_mask,
        batch_size,
        heads,
        seq_len,
        prefix_mask=None,
        sequence_id=None,
        key_padding_mask=None,
        alibi=False,
        dtype=None,
    ):
        if prefix_mask is not None:
            raise NotImplementedError(
                'This attention implementation does not support use of `prefix_lm`.'
            )

        if attn_mask is not None:
            # select seq_len subset of attn mask
            attn_mask = attn_mask[..., :seq_len, :seq_len]

        if key_padding_mask is not None and _check_apply_key_padding_mask(
                key_padding_mask):
            if attn_mask is None:
                attn_mask = key_padding_mask.new_zeros(
                    ((batch_size, 1, seq_len, seq_len)), dtype=dtype)

            kpm_fill_value = -1e4  # numerically stable -inf
            attn_mask = attn_mask.masked_fill(
                ~key_padding_mask.view((batch_size, 1, 1, seq_len)),
                kpm_fill_value)
            attn_mask = attn_mask.masked_fill(
                ~key_padding_mask.view((batch_size, 1, seq_len, 1)),
                kpm_fill_value)

        if sequence_id is not None:
            different_sequence = torch.logical_not(
                torch.eq(sequence_id.view(batch_size, 1, 1, seq_len),
                         sequence_id.view(batch_size, 1, seq_len, 1)))
            attn_mask = attn_mask.masked_fill(different_sequence, -1e4)

        return attn_mask


class TritonFlashPrefixAttention(TritonFlashAttention):
    """Triton FlashAttn kernel for a Prefix LM.

    See parent class :class:`TritonFlashAttention` for details.
    """
    causal = False

    @staticmethod
    def generate_attn_mask(
        attn_mask,
        batch_size,
        heads,
        seq_len,
        prefix_mask,
        sequence_id=None,
        key_padding_mask=None,
        alibi=False,
        dtype=None,
    ):
        if attn_mask is not None:
            # select seq_len subset of attn mask
            attn_mask = attn_mask[..., :seq_len, :seq_len]

        # Mix the causal max and the bidirectional mask to get the full
        # allowable attention (i.e. full = not accounting for padding yet)
        causal = torch.tril(
            torch.ones((seq_len, seq_len),
                       dtype=torch.uint8,
                       device=prefix_mask.device)).view(1, 1, seq_len, seq_len)
        prefix = prefix_mask.view(batch_size, 1, 1, seq_len)
        can_attend = torch.logical_or(causal, prefix)

        if sequence_id is not None:
            same_sequence = torch.eq(
                sequence_id.view(batch_size, 1, 1, seq_len),
                sequence_id.view(batch_size, 1, seq_len, 1))
            can_attend = torch.logical_and(can_attend, same_sequence)

        kpm_fill_value = -1e4  # numerically stable -inf

        # Account for padding
        if key_padding_mask is not None:
            not_padding = key_padding_mask.view(batch_size, 1, 1, seq_len)
            can_attend = torch.logical_and(can_attend, not_padding)

        # attn_mask is added to the attention logits. Convert accordingly.
        if attn_mask is None:
            attn_mask = torch.where(can_attend, 0, kpm_fill_value)
        else:
            attn_mask = torch.where(can_attend, attn_mask, kpm_fill_value)

        if dtype is not None:
            attn_mask = attn_mask.to(dtype)

        return attn_mask  # [batch_size, 1, seq_len, seq_len]


def _check_apply_key_padding_mask(key_padding_mask):
    if key_padding_mask.bool().logical_not().any():
        # check to verify all tokens after the first invalid tokens are invalid.
        # if there are no valid tokens after the first invalid token,
        # key_padding_mask isn't required given causal mask will eliminate
        # unwanted token interaction.
        # WARNING: this approach only works for right padded causal attn
        # NOTE: I chose this algorithm given its vectorized; there is room for improvement...
        c_sum = key_padding_mask.cumsum(1)
        num_valid_tokens = c_sum[:, -1].long()
        vals = c_sum[range(key_padding_mask.size(0)), num_valid_tokens - 1]
        return any(vals != num_valid_tokens)
    return False


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
