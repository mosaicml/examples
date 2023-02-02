# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

"""A simple, flexible implementation of a GPT model.

Inspired by https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
"""

import math
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from composer.metrics.nlp import LanguageCrossEntropy, Perplexity
from composer.models.base import ComposerModel
from omegaconf import DictConfig

from examples.llm.src.mosaic_gpt.layers import (FlashCausalAttention, GPTBlock,
                                                TorchCausalAttention,
                                                TritonFlashCausalAttention)


class MosaicGPT(nn.Module):

    def __init__(self, cfg: DictConfig):
        super().__init__()
        assert cfg.name == 'mosaic_gpt', f'Tried to build MosaicGPT model with cfg.name={cfg.name}'
        self.cfg = cfg

        # Choose the attention implementation
        if cfg.attn_impl == 'torch':
            self.causal_attn_cls = TorchCausalAttention
        elif cfg.attn_impl == 'flash':
            self.causal_attn_cls = FlashCausalAttention
        elif cfg.attn_impl == 'triton':
            self.causal_attn_cls = TritonFlashCausalAttention
        else:
            raise ValueError(f'Unknown attn_impl={cfg.attn_impl}')

        # TODO: comment and reference about ALIBI
        self.cfg.alibi = cfg.get('alibi', False)
        self.cfg.alibi_bias_max = cfg.get('alibi_bias_max',
                                          8 if self.cfg.alibi else None)

        # CogView (https://arxiv.org/abs/2105.13290) and GLM-130B (https://arxiv.org/abs/2210.02414)
        # both report that embedding_fraction helps stabilize training
        self.cfg.embedding_fraction = cfg.get('embedding_fraction', 1)
        assert 0 < self.cfg.embedding_fraction <= 1, 'embedding_fraction must be between 0 (exclusive) and 1 (inclusive)!'

        # Build the GPT architecture
        self.transformer = nn.ModuleDict({
            'wte':
                nn.Embedding(cfg.vocab_size,
                             cfg.d_model,
                             device=cfg.init_device)
        })
        if not self.cfg.alibi:
            self.transformer.update({
                'wpe':
                    nn.Embedding(cfg.max_seq_len,
                                 cfg.d_model,
                                 device=cfg.init_device)
            })
        self.transformer.update({'emb_drop': nn.Dropout(cfg.emb_pdrop)})
        self.transformer.update({
            'blocks':
                nn.ModuleList([
                    GPTBlock(cfg,
                             causal_attn_cls=self.causal_attn_cls,
                             device=cfg.init_device)
                    for _ in range(cfg.n_layers)
                ])
        })
        self.transformer.update(
            {'ln_f': nn.LayerNorm(cfg.d_model, device=cfg.init_device)})

        # Initialize parameters unless using 'meta' init_device
        if cfg.init_device != 'meta':
            self.apply(self.param_init_fn)

        # Define attention mask shape
        # Mask will be initialized on first forward pass
        # To work safely with 'meta' tensors
        self._attn_mask_initialized = False
        mask_shape = self.causal_attn_cls.mask_shape(cfg.n_heads,
                                                     cfg.max_seq_len,
                                                     self.cfg.alibi)
        if mask_shape:
            self.register_buffer(
                'attn_mask', torch.empty(mask_shape, device=cfg.init_device))
        else:
            self.attn_mask = None

    def _attn_mask(self, batch_size=None, seq_len=None, key_padding_mask=None):
        if not self._attn_mask_initialized:
            self.causal_attn_cls.attn_mask_(
                self.attn_mask,
                self.cfg.n_heads,
                self.cfg.max_seq_len,
                alibi=self.cfg.alibi,
                alibi_bias_max=self.cfg.alibi_bias_max)
            self._attn_mask_initialized = True

        if self.cfg.attn_impl == 'flash':
            return self.attn_mask  # None

        # select seq_len subset of attn mask
        assert self.attn_mask is not None, 'Internal logic error'
        attn_mask = self.attn_mask[..., :seq_len, :seq_len]

        if self.cfg.attn_impl == 'triton' and key_padding_mask is not None and key_padding_mask.bool(
        ).logical_not().any():
            attn_mask = attn_mask.masked_fill(
                ~key_padding_mask.view(batch_size, 1, 1, seq_len),
                float('-inf'))

        if self.cfg.attn_impl == 'torch':
            if key_padding_mask is not None and key_padding_mask.bool(
            ).logical_not().any():
                attn_mask = attn_mask.expand(batch_size, self.cfg.n_heads,
                                             seq_len, seq_len).clone()
                attn_mask.masked_fill_(
                    ~key_padding_mask.view(batch_size, 1, 1, seq_len),
                    float('-inf'))
                attn_mask = attn_mask.reshape(-1, seq_len, seq_len)
            elif self.cfg.alibi:
                # WARNING: Alibi with torch attn is not thoroughly tested
                # torch mask is supposed to be of shape nzz x SeqLen x SeqLen
                # we must braodcast to batch size then flatten batchsize * n_heads dim
                # Note: if key_padding_mask is triggered, the needed expansion is already done.
                attn_mask = attn_mask.expand(batch_size, self.cfg.n_heads,
                                             seq_len, seq_len).reshape(
                                                 -1, seq_len, seq_len)

        return attn_mask

    def forward(self,
                input_ids: torch.LongTensor,
                key_padding_mask: Optional[torch.ByteTensor] = None):
        B, S = input_ids.size()
        assert (
            S <= self.cfg.max_seq_len
        ), f'Cannot forward input with seq_len={S}, this model only supports seq_len<={self.cfg.max_seq_len}'

        tok_emb = self.transformer.wte(input_ids)  # type: ignore
        if self.cfg.alibi:
            x = tok_emb
        else:
            pos = torch.arange(0, S, dtype=torch.long,
                               device=input_ids.device).unsqueeze(0)
            pos_emb = self.transformer.wpe(pos)  # type: ignore
            x = tok_emb + pos_emb

        if self.cfg.embedding_fraction == 1:
            x = self.transformer.emb_drop(x)  # type: ignore
        else:
            # this implementation is proposed on page 7 of the GLM-130B paper https://arxiv.org/abs/2210.02414
            x_shrunk = (x * self.cfg.embedding_fraction) + (
                x.detach() * (1 - self.cfg.embedding_fraction))
            assert isinstance(self.transformer.emb_drop, nn.Module)  # pyright
            x = self.transformer.emb_drop(x_shrunk)

        attn_mask = self._attn_mask(batch_size=B,
                                    seq_len=S,
                                    key_padding_mask=key_padding_mask)
        for block in self.transformer.blocks:  # type: ignore
            x = block(
                x, None if self.cfg.attn_impl == 'triton' else key_padding_mask,
                attn_mask)
        x = self.transformer.ln_f(x)  # type: ignore
        # output embedding weight tied to input embedding
        assert isinstance(self.transformer.wte, nn.Module)  # pyright
        assert isinstance(self.transformer.wte.weight, torch.Tensor)  # pyright
        logits = F.linear(x, self.transformer.wte.weight, None)
        return logits

    # Param Initialization, needed for device='meta' fast initialization
    def param_init_fn(self, module):
        init_fn = partial(torch.nn.init.normal_,
                          mean=0.0,
                          std=self.cfg.init_std)
        # Linear
        if isinstance(module, nn.Linear):
            init_fn(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

            if getattr(module, '_is_residual', False):
                module.weight.data.normal_(
                    mean=0.0,
                    std=(self.cfg.init_std / math.sqrt(2 * self.cfg.n_layers)))

        # Embedding
        if isinstance(module, nn.Embedding):
            init_fn(module.weight)

        # LayerNorm
        if isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

        # torch's MultiheadAttention
        if isinstance(module, nn.MultiheadAttention):
            if module._qkv_same_embed_dim:
                assert module.in_proj_weight is not None
                assert module.q_proj_weight is None and module.k_proj_weight is None and module.v_proj_weight is None
                init_fn(module.in_proj_weight)
            else:
                assert module.q_proj_weight is not None and module.k_proj_weight is not None and module.v_proj_weight is not None
                assert module.in_proj_weight is None
                init_fn(module.q_proj_weight)
                init_fn(module.k_proj_weight)
                init_fn(module.v_proj_weight)

            # bias
            if module.in_proj_bias is not None:
                torch.nn.init.zeros_(module.in_proj_bias)
            if module.bias_k is not None:
                torch.nn.init.zeros_(module.bias_k)
            if module.bias_v is not None:
                torch.nn.init.zeros_(module.bias_v)

            # out proj
            if module.out_proj._is_residual:
                module.out_proj.weight.data.normal_(
                    mean=0.0,
                    std=(self.cfg.init_std / math.sqrt(2 * self.cfg.n_layers)))
            else:
                init_fn(module.out_proj.weight)
            if module.out_proj.bias is not None:
                torch.nn.init.zeros_(module.out_proj.bias)

    # FSDP Wrap function
    def fsdp_wrap_fn(self, module):
        return isinstance(module, GPTBlock)

    # Activation Checkpointing
    def activation_checkpointing_fn(self, module):
        return isinstance(module, GPTBlock)


class ComposerMosaicGPT(ComposerModel):

    def __init__(self, cfg):
        super().__init__()
        self.model = MosaicGPT(cfg)
        self.__num_fwd_flops = None
        self.train_metrics = {
            'LanguageCrossEntropy': LanguageCrossEntropy(cfg.vocab_size),
            'Perplexity': Perplexity(),
        }
        self.eval_metrics = {
            'LanguageCrossEntropy': LanguageCrossEntropy(cfg.vocab_size),
            'Perplexity': Perplexity(),
        }

    def get_targets(self, batch):
        targets = torch.roll(batch['labels'], shifts=-1)
        targets[:, -1] = -100
        return targets

    def forward(self, batch):
        return self.model(batch['input_ids'],
                          key_padding_mask=batch['attention_mask'].bool())

    def eval_forward(self, batch, outputs=None):
        return outputs if outputs is not None else self.forward(batch)

    def loss(self, outputs, batch):
        targets = self.get_targets(batch)
        return F.cross_entropy(outputs.view(-1, outputs.size(-1)),
                               targets.view(-1),
                               ignore_index=-100)

    def get_metrics(self, is_train=False):
        return self.train_metrics if is_train else self.eval_metrics

    def update_metric(self, batch, outputs, metric):
        outputs = outputs.view(-1, outputs.size(-1))
        targets = self.get_targets(batch).view(-1)
        metric.update(outputs, targets)

    @property
    def num_fwd_flops(self):
        if self.__num_fwd_flops:
            return self.__num_fwd_flops
        n_params = sum(p.numel() for p in self.parameters())
        # the number of paramters is approximately the number of multiply-accumulates (MAC) in the network
        # each MAC has 2 FLOPs - we multiply by 2 ie 2 * n_param
        # this gets us FLOPs / token
        params_flops_per_token = 2 * n_params
        params_flops_per_seq = params_flops_per_token * self.model.cfg.max_seq_len
        # there are 2 FLOPS per mac; there is A=Q*K^T and out=A*V ops (ie mult by 2)
        attn_flops_per_seq = self.model.cfg.n_layers * 2 * 2 * (
            self.model.cfg.d_model * (self.model.cfg.max_seq_len**2))
        self.__num_fwd_flops = params_flops_per_seq + attn_flops_per_seq
        return self.__num_fwd_flops
