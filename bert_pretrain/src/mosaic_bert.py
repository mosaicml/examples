# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Implements a BERT wrapper around a :class:`.ComposerTransformer`."""

from __future__ import annotations

from typing import Optional

import transformers
from composer.metrics.nlp import LanguageCrossEntropy, MaskedAccuracy
from composer.models.huggingface import HuggingFaceModel

from src.bert_layers import BertForMaskedLM

all = ['create_mosaic_bert_mlm']


def create_mosaic_bert_mlm(pretrained_model_name: str = 'bert-base-uncased',
                           use_pretrained: Optional[bool] = False,
                           model_config: Optional[dict] = None,
                           tokenizer_name: Optional[str] = None,
                           gradient_checkpointing: Optional[bool] = False):
    """BERT model based on HazyResearch's MLPerf 2.0 submission and HuggingFace transformer"""
    if not model_config:
        model_config = {}

    if not pretrained_model_name:
        pretrained_model_name = 'bert-base-uncased'

    config = transformers.AutoConfig.from_pretrained(pretrained_model_name, **model_config)
    assert transformers.AutoModelForMaskedLM.from_config is not None, 'AutoModelForMaskedLM has from_config method'
    config.return_dict = False
    config.dense_seq_output = True  # Required BertForMaskedLM
    config.last_layer_subset = True
    # Padding for divisibility by 8
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)

    if use_pretrained:
        raise NotImplementedError('The `use_pretrained` option is not currently supported for mosaic_bert. Please set `use_pretrained=False`.')
    else:
        model = BertForMaskedLM(config)

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()  # type: ignore

    # setup the tokenizer
    if tokenizer_name:
        tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
    else:
        tokenizer = None

    metrics = [
        LanguageCrossEntropy(ignore_index=-100, vocab_size=model.config.vocab_size),
        MaskedAccuracy(ignore_index=-100)
    ]

    hf_model = HuggingFaceModel(model=model, tokenizer=tokenizer, use_logits=True, metrics=metrics)

    # Padding for divisibility by 8
    # We have to do it again here because wrapping by HuggingFaceModel changes it
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)
    hf_model.model.resize_token_embeddings(config.vocab_size)

    return hf_model


# from transformers import AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# tokens = tokenizer(["yooooooooo", f"friendship ended with cuda, now triton is my {tokenizer.mask_token} friend"], return_tensors="pt", padding=True).to("cuda")

# model = create_mosaic_bert_mlm()
# model = model.to("cuda")
# print(model.model(**tokens))