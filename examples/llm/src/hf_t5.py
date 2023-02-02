# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import transformers
from composer.metrics.nlp import (LanguageCrossEntropy, MaskedAccuracy)
from composer.models.huggingface import HuggingFaceModel
from transformers import T5Config, T5ForConditionalGeneration
from transformers.models.t5.modeling_t5 import T5Block
from examples.common.hf_fsdp import prepare_hf_model_for_fsdp


def create_hf_t5(pretrained_model_name: str = 't5-base',
                 use_pretrained: Optional[bool] = False,
                 model_config: Optional[dict] = None,
                 tokenizer_name: Optional[str] = None):
    """T5 model based on |:hugging_face:| Transformers.

    For more information, see `Transformers <https://huggingface.co/transformers/>`_.

    Args:
        pretrained_model_name (str): Name of the Hugging Face model to instantiate. Default: ``'t5-base'``.
        use_pretrained (bool, optional): Whether to initialize the model with the pretrained weights. Default: ``False``.
        model_config (dict): The settings used to create a Hugging Face T5Config. T5Config is used to specify the
            architecture of a Hugging Face model.
        tokenizer_name (str, optional): Tokenizer name used to preprocess the dataset and validate the models inputs.

    To create a |:hugging_face:| T5 model for Masked Language Model pretraining:

    .. testcode::

        from examples.llm.src.hf_t5 import create_hf_t5
        model = create_hf_t5('t5-base')
    """

    if not pretrained_model_name:
        pretrained_model_name = 't5-base'

    # setup the tokenizer
    if tokenizer_name:
        tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained_model_name)

    if not model_config:
        model_config = {}

    if use_pretrained:
        model = T5ForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name, **model_config)
    else:
        config = T5Config.from_pretrained(pretrained_model_name, **model_config)
        model = T5ForConditionalGeneration(config)

    # Make this available for FSDP
    # prepare_hf_t5_model_for_fsdp(model)
    prepare_hf_model_for_fsdp(model)

    # We use `len(tokenizer) instead of `model.config.vocab_size` because 
    # `HuggingFaceModel` will set the latter to the former
    metrics = [
        # Note: The HF code is hardcoded to use -100 as the ignore index
        LanguageCrossEntropy(
            ignore_index=-100, vocab_size=len(tokenizer)
        ),  
        MaskedAccuracy(ignore_index=-100)
    ]
    return HuggingFaceModel(model=model,
                            tokenizer=tokenizer,
                            metrics=metrics,
                            use_logits=True)


def prepare_hf_t5_model_for_fsdp(model: T5ForConditionalGeneration):
    """FSDP wrap a HuggingFace T5 model.

    Wrap any model for FSDP which follows one of the 3 existing conventions from
    HuggingFace for decoder-only LLMs.
    """
    # When using the HF LM models,
    # the weights of the self.shared, self.encoder.embed_tokens,
    # self.decoder.embed_tokens, and self.lm_head are tied.
    # This tying occurs inside the `self.post_init()` function.
    # This is a hurdle for FSDP because they need to be in the same FSDP block
    # These lines ensures that both modules stay together in the top-most block when
    # the model has this tying enabled (almost all do; this property defaults to True)
    if model.config.tie_word_embeddings:
        # Tied embeddings & output head
        model.shared._fsdp_wrap = False # type: ignore
        model.encoder.embed_tokens._fsdp_wrap = False # type: ignore
        model.decoder.embed_tokens._fsdp_wrap = False # type: ignore
        model.lm_head._fsdp_wrap = False # type: ignore
        # Model backbones
        model.encoder._fsdp_warp = False # type: ignore
        model.decoder._fsdp_warp = False # type: ignore

    # FSDP Wrap and Activation Checkpoint every model block
    model.fsdp_wrap_fn = lambda module: isinstance(module, T5Block)
    model.activation_checkpointing_fn = lambda module: isinstance(
        module, T5Block)
