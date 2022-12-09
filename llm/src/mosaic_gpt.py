# Copyright 2022 MosaicML Benchmarks authors
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

        self.register_buffer(
            'mask',
            torch.empty((cfg.max_seq_len, cfg.max_seq_len), device=device))
        self.mask_initialized = False
        self.mhsa.out_proj._is_residual = True  # type: ignore

    def _fill_causal_attn_mask(self):
        assert isinstance(self.mask, torch.Tensor)  # for type checking
        torch.full(size=self.mask.shape,
                   fill_value=float('-inf'),
                   out=self.mask)
        torch.triu(input=self.mask, diagonal=1, out=self.mask)

    def forward(self, x, key_padding_mask):
        # Two important disclaimers
        # 1. Torch uses additive attention. If your attn_mask/key_padding mask is a float tensor, it will add the floats
        #   directly to your attention matrix. If they are boolean masks, True will be converted to -inf before adding the
        #   mask to your attentions. See https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html#torch.nn.MultiheadAttention.forward
        #   Basically True/-inf indicates tokens we do not want to attend to.
        #
        # 2. This is is the exact opposite behavior of Huggingface's tokenizers, which use the convention that True denotes tokens
        #   we do want to attend to. See https://huggingface.co/docs/transformers/glossary#attention-mask
        #
        if not self.mask_initialized:
            self._fill_causal_attn_mask()
            self.mask_initialized = True

        return self.mhsa(x,
                         x,
                         x,
                         attn_mask=self.mask,
                         key_padding_mask=~key_padding_mask,
                         need_weights=True)


class FlashCausalAttention(nn.Module):

    def __init__(self, cfg: DictConfig, device: Optional[str] = None):
        super().__init__()
        try:
            from flash_attn.flash_attention import FlashMHA
        except ImportError as e:
            raise e

        self.mhsa = FlashMHA(
            embed_dim=cfg.d_model,
            num_heads=cfg.n_heads,
            attention_dropout=cfg.attn_pdrop,
            bias=True,
            batch_first=True,
            causal=True,
            device=device,
        )
        self.mhsa.out_proj._is_residual = True

    def forward(self, x, key_padding_mask):
        return self.mhsa(x,
                         key_padding_mask=key_padding_mask,
                         need_weights=False)

class SparseSelfAttention(nn.Module):
    def __init__(self, cfg: Mapping[str, Any], device: str = None):
        super().__init__()
        from deepspeed.ops.sparse_attention import SparseSelfAttention
        from deepspeed.ops.sparse_attention import FixedSparsityConfig

        # Arguably we should really be using BertSparseSelfAttention b/c we're copying their code,
        # but the underlying SparseSelfAttention class seems to only support
        # fp16 training since they rewrote the kernels in Triton and BertSparseSelfAttention no longer works.

        # copied from https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/ops/sparse_attention/bert_sparse_self_attention.py#L36
        self.attention_head_size = int(cfg.d_model / cfg.n_heads)
        self.num_attention_heads = cfg.n_heads
        self.all_head_size = cfg.n_heads * self.attention_head_size
        self.query = nn.Linear(cfg.d_model, self.all_head_size)
        self.key = nn.Linear(cfg.d_model, self.all_head_size)
        self.value = nn.Linear(cfg.d_model, self.all_head_size)
        print(f"at init time: {self.query.weight.dtype=} {self.key.weight.dtype=} {self.value.weight.dtype=}")
        print(f"at init time: {self.query.bias.dtype=} {self.key.bias.dtype=} {self.value.bias.dtype=}")

        config = FixedSparsityConfig(attention="unidirectional", num_heads=cfg.n_heads)
        self.mha = SparseSelfAttention(config, max_seq_length=cfg.max_seq_len)
        print(self.mha)
        # self.mha.out_proj._is_residual = True

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads,
                                       self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, x, key_padding_mask):
        # copied from https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/ops/sparse_attention/bert_sparse_self_attention.py#LL62-L73C84
        # print(f"{x.dtype=}")
        # print(f"at forward time: {self.query.weight.dtype=} {self.key.weight.dtype=} {self.value.weight.dtype=}")
        # print(f"at forward time: {self.query.bias.dtype=} {self.key.bias.dtype=} {self.value.bias.dtype=}")
        
        mixed_query_layer = self.query(x)
        mixed_key_layer = self.key(x)
        mixed_value_layer = self.value(x)

        # print("mixed_query_layer layer", mixed_query_layer.dtype, "mixed_key_layer", mixed_key_layer.dtype, "mixed_value_layer layer", mixed_value_layer.dtype)


        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # print("query layer", query_layer.dtype, "key_layer", key_layer.dtype, "value layer", value_layer.dtype)

        context_layer = self.mha(query_layer,
                                                   key_layer,
                                                   value_layer,
                                                   key_padding_mask=key_padding_mask)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size, )
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


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
        self.ln_1 = nn.LayerNorm(cfg.d_model, device=device)
        if cfg.attn_impl == 'torch':
            self.causal_attn = TorchCausalAttention(cfg, device)
        elif cfg.attn_impl == 'flash':
            self.causal_attn = FlashCausalAttention(cfg, device)
        elif cfg.attn_impl == 'sparse':
            self.causal_attn = SparseSelfAttention(cfg, device)
            self.causal_attn._fsdp_wrap = True
        else:
            raise ValueError(f'Unknown attn_impl={cfg.attn_impl}')
        self.ln_2 = nn.LayerNorm(cfg.d_model, device=device)
        self.mlp = GPTMLP(cfg, device=device)
        self.resid_attn_dropout = nn.Dropout(cfg.resid_pdrop)
        self.resid_mlp_dropout = nn.Dropout(cfg.resid_pdrop)

    def forward(
            self,
            x: torch.Tensor,
            key_padding_mask: Optional[torch.ByteTensor] = None
    ) -> torch.Tensor:
        a = self.ln_1(x)
        b = self.causal_attn(a, key_padding_mask)
        if type(b) == tuple:
            b = b[0]
        x = x + self.resid_attn_dropout(b)
        m = self.ln_2(x)
        n = self.mlp(m)
        x = x + self.resid_mlp_dropout(n)
        return x


class MosaicGPT(nn.Module):

    def __init__(self, cfg: DictConfig):
        super().__init__()
        assert cfg.name == 'mosaic_gpt', f'Tried to build MosaicGPT model with cfg.name={cfg.name}'
        self.cfg = cfg
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(cfg.vocab_size, cfg.d_model,
                                 device=cfg.device),
                wpe=nn.Embedding(cfg.max_seq_len,
                                 cfg.d_model,
                                 device=cfg.device),
                emb_drop=nn.Dropout(cfg.emb_pdrop),
                blocks=nn.ModuleList([
                    GPTBlock(cfg, device=cfg.device)
                    for _ in range(cfg.n_layers)
                ]),
                ln_f=nn.LayerNorm(cfg.d_model, device=cfg.device),
            ))
        self.lm_head = nn.Linear(cfg.d_model,
                                 cfg.vocab_size,
                                 bias=False,
                                 device=cfg.device)

        # Apply weight tying
        # Ensures that wte and lm_head are in the same FSDP block
        self.transformer._fsdp_wrap = False  # type: ignore
        self.transformer.wte._fsdp_wrap = False  # type: ignore
        self.lm_head._fsdp_wrap = False  # type: ignore
        self.lm_head.weight = self.transformer.wte.weight  # type: ignore

        if cfg.device != 'meta':
            self.apply(self.param_init_fn)

    def forward(self,
                input_ids: torch.LongTensor,
                key_padding_mask: Optional[torch.ByteTensor] = None):
        _, S = input_ids.size()
        assert (
            S <= self.cfg.max_seq_len
        ), f'Cannot forward input with seq_len={S}, this model only supports seq_len<={self.cfg.max_seq_len}'
        pos = torch.arange(0, S, dtype=torch.long,
                           device=input_ids.device).unsqueeze(0)

        tok_emb = self.transformer.wte(input_ids)  # type: ignore
        pos_emb = self.transformer.wpe(pos)  # type: ignore
        x = self.transformer.emb_drop(tok_emb + pos_emb)  # type: ignore
        for block in self.transformer.blocks:  # type: ignore
            x = block(x, key_padding_mask)
        x = self.transformer.ln_f(x)  # type: ignore
        logits = self.lm_head(x)
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

