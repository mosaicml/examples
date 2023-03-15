# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

"""A HuggingFace-style model configuration."""

from transformers import PretrainedConfig


class MosaicGPTConfig(PretrainedConfig):
    model_type = 'mosaic_gpt'

    def __init__(
        self,
        d_model: int = 2048,
        n_heads: int = 16,
        n_layers: int = 24,
        mlp_ratio: int = 4,
        max_seq_len: int = 8096,
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
        **kwargs,
    ):
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads')
        if any(prob < 0 or prob > 1
               for prob in [attn_pdrop, resid_pdrop, emb_pdrop]):
            raise ValueError(
                'attn_pdrop, resid_pdrop, emb_pdrop are probabilities and must be between 0 and 1'
            )
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
        if 'name' in kwargs:
            del kwargs['name']
        super().__init__(**kwargs)
