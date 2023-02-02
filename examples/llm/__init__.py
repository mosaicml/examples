# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

try:
    from examples.llm.src.flash_attention import FlashAttention, FlashMHA
    from examples.llm.src.flash_attn_triton import \
        flash_attn_func as flash_attn_func_llm
    from examples.llm.src.flash_attn_triton import \
        flash_attn_qkvpacked_func as flash_attn_qkvpacked_func_llm
    from examples.llm.src.hf_causal_lm import (
        ComposerHFCausalLM, hf_get_causal_base_model,
        hf_get_causal_hidden_layers, hf_get_lm_head,
        hf_get_tied_embedding_weights, prepare_hf_causal_lm_model_for_fsdp)
    from examples.llm.src.model_registry import COMPOSER_MODEL_REGISTRY
    from examples.llm.src.mosaic_gpt import (GPTMLP, ComposerMosaicGPT,
                                             FlashCausalAttention, GPTBlock,
                                             MosaicGPT, TorchCausalAttention,
                                             TritonFlashCausalAttention,
                                             alibi_bias)
    from examples.llm.src.tokenizer import (TOKENIZER_REGISTRY, HFTokenizer,
                                            LLMTokenizer)
except ImportError as e:
    raise Exception(
        'Please make sure to pip install .[llm] to get the requirements for the LLM example.'
    ) from e

__all__ = [
    'FlashAttention',
    'FlashMHA',
    'flash_attn_func_llm',
    'flash_attn_qkvpacked_func_llm',
    'hf_get_causal_base_model',
    'hf_get_lm_head',
    'hf_get_causal_hidden_layers',
    'hf_get_tied_embedding_weights',
    'prepare_hf_causal_lm_model_for_fsdp',
    'ComposerHFCausalLM',
    'COMPOSER_MODEL_REGISTRY',
    'TorchCausalAttention',
    'FlashCausalAttention',
    'TritonFlashCausalAttention',
    'alibi_bias',
    'GPTMLP',
    'GPTBlock',
    'MosaicGPT',
    'ComposerMosaicGPT',
    'LLMTokenizer',
    'HFTokenizer',
    'TOKENIZER_REGISTRY',
]
