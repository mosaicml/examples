# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

"""A HuggingFace-style model configuration."""

from typing import Optional, Tuple, Union

from transformers import PretrainedConfig


class MosaicGPTConfig(PretrainedConfig):
    model_type = 'mosaic_gpt'

    def __init__(
        self,
        d_model: int = 2048,
        n_heads: int = 16,
        n_layers: int = 24,
        mlp_ratio: int = 4,
        max_seq_len: int = 2048,
        vocab_size: int = 50257,
        init_std: float = 0.02,
        attn_pdrop: float = 0.0,
        resid_pdrop: float = 0.0,
        emb_pdrop: float = 0.0,
        attn_impl: str = 'triton',
        attn_qk_ln: bool = False,
        attn_clip_qkv: Optional[float] = None,
        softmax_scale: Optional[float] = None,
        alibi: bool = False,
        alibi_bias_max: int = 8,
        init_device: str = 'cpu',
        logit_scale: Optional[Union[float, str]] = None,
        no_bias: bool = False,
        verbose: int = 0,
        param_init_fn: str = 'baseline_',
        init_div_is_residual: bool = True,
        emb_init_std: Optional[float] = None,
        emb_init_uniform_lim: Optional[Union[Tuple[float, float],
                                             float]] = None,
        init_gain: float = 0,
        fan_mode: str = 'fan_in',
        init_nonlinearity: str = 'leaky_relu',
        embedding_fraction: float = 1.0,
        low_precision_layernorm: bool = False,
        **kwargs,
    ):
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.mlp_ratio = mlp_ratio
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.init_std = init_std
        self.attn_pdrop = attn_pdrop
        self.resid_pdrop = resid_pdrop
        self.emb_pdrop = emb_pdrop
        self.attn_impl = attn_impl
        self.attn_qk_ln = attn_qk_ln
        self.attn_clip_qkv = attn_clip_qkv
        self.softmax_scale = softmax_scale
        self.alibi = alibi
        self.alibi_bias_max = alibi_bias_max
        self.init_device = init_device
        self.logit_scale = logit_scale
        self.no_bias = no_bias
        self.verbose = verbose
        self.param_init_fn = param_init_fn
        self.init_div_is_residual = init_div_is_residual
        self.emb_init_std = emb_init_std
        self.emb_init_uniform_lim = emb_init_uniform_lim
        self.init_std = init_std
        self.init_gain = init_gain
        self.fan_mode = fan_mode
        self.init_nonlinearity = init_nonlinearity
        self.embedding_fraction = embedding_fraction
        self.low_precision_layernorm = low_precision_layernorm
        if 'name' in kwargs:
            del kwargs['name']
        super().__init__(**kwargs)

        self._validate_config()

    def _validate_config(self):
        if self.d_model % self.n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads')
        if any(prob < 0 or prob > 1
               for prob in [self.attn_pdrop, self.resid_pdrop, self.emb_pdrop]):
            raise ValueError(
                'attn_pdrop, resid_pdrop, emb_pdrop are probabilities and must be between 0 and 1'
            )
        if self.attn_impl not in ['torch', 'flash', 'triton']:
            raise ValueError(f'Unknown attn_impl={self.attn_impl}')
        if self.attn_qk_ln and self.attn_impl not in ['flash', 'triton']:
            raise NotImplementedError(
                'LayerNorm over queries and keys in attention is only implemented with flash and triton attention.'
            )
        if self.attn_clip_qkv and self.attn_impl not in ['flash', 'triton']:
            raise NotImplementedError(
                'QKV clipping only implemented with flash and triton attention.'
            )
        if self.softmax_scale is not None and self.attn_impl not in [
                'flash', 'triton'
        ]:
            raise NotImplementedError(
                'softmax_scale only implemented with flash and triton attention.'
            )
        if self.alibi and self.attn_impl not in ['torch', 'triton']:
            raise NotImplementedError(
                'alibi only implemented with torch and triton attention.')
        if self.embedding_fraction > 1 or self.embedding_fraction <= 0:
            raise ValueError(
                'model.embedding_fraction must be between 0 (exclusive) and 1 (inclusive)!'
            )
        if isinstance(self.logit_scale,
                      str) and self.logit_scale != 'inv_sqrt_d_model':
            raise ValueError(
                f"{self.logit_scale=} is not recognized as an option; use numeric value or 'inv_sqrt_d_model'."
            )