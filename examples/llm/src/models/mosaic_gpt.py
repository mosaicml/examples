# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

"""A simple, flexible implementation of a GPT model.

Inspired by https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
"""

import math
import warnings
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from composer.algorithms.low_precision_layernorm.low_precision_layernorm import \
    LPLayerNorm
from composer.metrics import METRIC_DEFAULT_CTORS, InContextLearningMetric
from composer.metrics.nlp import LanguageCrossEntropy, Perplexity
from composer.models.base import ComposerModel
from omegaconf import DictConfig
from omegaconf import OmegaConf as om
from transformers import PreTrainedModel

import examples.llm.src.models.layers.attention as attention
import examples.llm.src.models.layers.gpt_blocks as gpt_blocks
from examples.llm.src.models.configuration_mosaic_gpt import MosaicGPTConfig
from examples.llm.src.models.param_init_fns import MODEL_INIT_REGISTRY


class MosaicGPT(PreTrainedModel):
    config_class = MosaicGPTConfig
    base_model_prefix = 'mosaic_gpt'

    def __init__(self, config: MosaicGPTConfig):
        super().__init__(config)
        if config.attn_impl == 'torch':
            self.causal_attn_cls = attention.TorchCausalAttention
        elif config.attn_impl == 'flash':
            self.causal_attn_cls = attention.FlashCausalAttention
        elif config.attn_impl == 'triton':
            self.causal_attn_cls = attention.TritonFlashCausalAttention

        self.alibi = config.alibi
        self.alibi_bias_max = config.alibi_bias_max

        layernorm_class = LPLayerNorm if config.low_precision_layer_norm else nn.LayerNorm

        # CogView (https://arxiv.org/abs/2105.13290) and GLM-130B (https://arxiv.org/abs/2210.02414)
        # both report this helping with stabilizing training
        self.embedding_fraction = config.embedding_fraction

        self.transformer = nn.ModuleDict({
            'wte':
                nn.Embedding(config.vocab_size,
                             config.d_model,
                             device=config.init_device)
        })
        if not self.alibi:
            self.transformer.update({
                'wpe':
                    nn.Embedding(config.max_seq_len,
                                 config.d_model,
                                 device=config.init_device)
            })
        self.transformer.update({'emb_drop': nn.Dropout(config.emb_pdrop)})
        self.transformer.update({
            'blocks':
                nn.ModuleList([
                    gpt_blocks.GPTBlock(causal_attn_cls=self.causal_attn_cls,
                                        device=config.init_device,
                                        **config.to_dict())
                    for _ in range(config.n_layers)
                ])
        })
        self.transformer.update({
            'ln_f': layernorm_class(config.d_model, device=config.init_device)
        })

        # enables scaling output logits; similar to a softmax "temperature"
        # PaLM paper uses scale 1/sqrt(config.d_model)
        self.logit_scale = None
        if config.logit_scale is not None:
            logit_scale = config.logit_scale
            if isinstance(logit_scale, str):
                if logit_scale == 'inv_sqrt_d_model':
                    logit_scale = 1 / math.sqrt(config.d_model)
                else:
                    raise ValueError(
                        f"{logit_scale=} is not recognized as an option; use numeric value or 'inv_sqrt_d_model'."
                    )
            self.logit_scale = logit_scale

        if config.init_device != 'meta':
            print(
                f'You are using {config.init_device=}, but you can also use config.init_device="meta" with Composer + FSDP for fast initialization.'
            )
            self.apply(self.param_init_fn)

        # define attn mask
        self._attn_mask_initialized = False
        mask_shape = self.causal_attn_cls.mask_shape(config.n_heads,
                                                     config.max_seq_len,
                                                     self.alibi)
        if mask_shape is not None:
            self.register_buffer(
                'attn_mask', torch.empty(mask_shape, device=config.init_device))
        else:
            self.attn_mask = None

        if config.no_bias:
            for module in self.modules():
                if hasattr(module, 'bias') and isinstance(
                        module.bias, nn.Parameter):
                    if config.verbose:
                        print(f'Removing bias ({module.bias}) from {module}.')
                    module.register_parameter('bias', None)

        if config.verbose and config.verbose > 2:
            print(self)

    def _attn_mask(self,
                   batch_size=None,
                   seq_len=None,
                   key_padding_mask=None,
                   dtype=None):
        if not self._attn_mask_initialized:
            self.causal_attn_cls.attn_mask_(self.attn_mask,
                                            self.config.n_heads,
                                            self.config.max_seq_len,
                                            alibi=self.alibi,
                                            alibi_bias_max=self.alibi_bias_max)
            self._attn_mask_initialized = True

        return self.causal_attn_cls.generate_attn_mask(
            self.attn_mask,
            batch_size,
            self.config.n_heads,
            seq_len,
            key_padding_mask=key_padding_mask,
            alibi=self.alibi,
            dtype=dtype)

    def forward(self,
                input_ids: torch.LongTensor,
                key_padding_mask: Optional[torch.ByteTensor] = None):
        B, S = input_ids.size()
        assert (
            S <= self.config.max_seq_len
        ), f'Cannot forward input with seq_len={S}, this model only supports seq_len<={self.config.max_seq_len}'

        tok_emb = self.transformer.wte(input_ids)  # type: ignore
        if self.alibi:
            x = tok_emb
        else:
            pos = torch.arange(0, S, dtype=torch.long,
                               device=input_ids.device).unsqueeze(0)
            pos_emb = self.transformer.wpe(pos)  # type: ignore
            x = tok_emb + pos_emb

        if self.embedding_fraction == 1:
            x = self.transformer.emb_drop(x)  # type: ignore
        else:
            # this implementation is proposed on page 7 of the GLM-130B paper https://arxiv.org/abs/2210.02414
            x_shrunk = (x * self.embedding_fraction) + (
                x.detach() * (1 - self.embedding_fraction))
            assert isinstance(self.transformer.emb_drop, nn.Module)  # pyright
            x = self.transformer.emb_drop(x_shrunk)

        attn_mask = self._attn_mask(batch_size=B,
                                    seq_len=S,
                                    key_padding_mask=key_padding_mask,
                                    dtype=x.dtype)
        if self.config.attn_impl == 'flash' and key_padding_mask is None:
            # HazyResearch FlashMHA appears to use more memory when `key_padding_mask=None`
            # in certain settings like MosaicGPT-7B. So we always provide a tensor.
            # See https://github.com/mosaicml/examples/pull/163 for more details.
            mod_key_padding_mask = torch.ones_like(input_ids, dtype=torch.bool)
        elif self.config.attn_impl == 'triton':
            mod_key_padding_mask = None
        else:
            mod_key_padding_mask = key_padding_mask
        for block in self.transformer.blocks:  # type: ignore
            x = block(x, mod_key_padding_mask, attn_mask)
        x = self.transformer.ln_f(x)  # type: ignore
        # output embedding weight tied to input embedding
        assert isinstance(self.transformer.wte, nn.Module)  # pyright
        assert isinstance(self.transformer.wte.weight, torch.Tensor)  # pyright
        logits = F.linear(x, self.transformer.wte.weight, None)

        if self.logit_scale is not None:
            if self.logit_scale == 0:
                warnings.warn(
                    f'Multiplying logits by {self.logit_scale=}. This will produce uniform (uninformative) outputs.'
                )
            logits *= self.logit_scale

        return logits

    # Param Initialization, needed for device='meta' fast initialization
    def param_init_fn(self, module):
        init_fn_name = self.config.param_init_fn
        if self.config.verbose > 1:
            warnings.warn(f'Using {init_fn_name} initialization.')
        MODEL_INIT_REGISTRY[init_fn_name](module=module,
                                          **self.config.to_dict())

    # FSDP Wrap function
    def fsdp_wrap_fn(self, module):
        return isinstance(module, gpt_blocks.GPTBlock)

    # Activation Checkpointing
    def activation_checkpointing_fn(self, module):
        return isinstance(module, gpt_blocks.GPTBlock)


class ComposerMosaicGPT(ComposerModel):

    def __init__(self, om_model_config: DictConfig):
        super().__init__()

        resolved_om_config = om.to_container(om_model_config, resolve=True)
        self.hf_config = MosaicGPTConfig.from_dict(resolved_om_config)
        self.model = MosaicGPT(self.hf_config)
        self.num_fwd_flops = self._compute_num_fwd_flops()
        self.train_metrics = {
            'LanguageCrossEntropy':
                LanguageCrossEntropy(self.hf_config.vocab_size),
            'Perplexity':
                Perplexity(),
        }
        self.eval_metrics = {
            'LanguageCrossEntropy':
                LanguageCrossEntropy(self.hf_config.vocab_size),
            'Perplexity':
                Perplexity(),
        }

    def get_targets(self, batch):
        targets = torch.roll(batch['labels'], shifts=-1)
        targets[:, -1] = -100
        return targets

    def forward(self, batch):
        input_ids = batch['input_ids']
        key_padding_mask = batch['attention_mask'].bool(
        ) if 'attention_mask' in batch else None
        return self.model(input_ids=input_ids,
                          key_padding_mask=key_padding_mask)

    def eval_forward(self, batch, outputs=None):
        return outputs if outputs is not None else self.forward(batch)

    def loss(self, outputs, batch):
        targets = self.get_targets(batch)
        return F.cross_entropy(outputs.view(-1, outputs.size(-1)),
                               targets.view(-1),
                               ignore_index=-100)

    def get_metrics(self, is_train=False):
        return self.train_metrics if is_train else self.eval_metrics

    def update_metric(self, batch, outputs, metric) -> None:
        if isinstance(metric, InContextLearningMetric):
            if batch.get('mode', None) == 'icl_task':
                # only apply ICL metrics to specially constructed
                # icl_task batches
                targets = self.get_targets(batch)
                metric.update(batch, outputs, targets)
        else:
            outputs = outputs.view(-1, outputs.size(-1))
            targets = self.get_targets(batch).view(-1)
            metric.update(outputs, targets)

    def add_eval_metrics(self, evaluator):
        evaluator_metrics = {
            m: METRIC_DEFAULT_CTORS[m]() for m in evaluator.metric_names
        }
        if self.eval_metrics is not None:
            self.eval_metrics.update(evaluator_metrics)
        else:
            self.eval_metrics = evaluator_metrics

    def _compute_num_fwd_flops(self):
        n_params = sum(p.numel() for p in self.parameters())
        # the number of paramters is approximately the number of multiply-accumulates (MAC) in the network
        # each MAC has 2 FLOPs - we multiply by 2 ie 2 * n_param
        # this gets us FLOPs / token
        params_flops_per_token = 2 * n_params
        params_flops_per_seq = params_flops_per_token * self.model.config.max_seq_len
        # there are 2 FLOPS per mac; there is A=Q*K^T and out=A*V ops (ie mult by 2)
        attn_flops_per_seq = self.model.config.n_layers * 2 * 2 * (
            self.model.config.d_model * (self.model.config.max_seq_len**2))
        return params_flops_per_seq + attn_flops_per_seq

    def flops_per_batch(self, batch):
        # Note: this computation does not take into account padding, and assumes
        # that the dataset has been constructed without padding. Additionally, we
        # assume the backward pass is approximately 2x the forward pass
        return self.num_fwd_flops * 3 * batch['input_ids'].shape[0]
