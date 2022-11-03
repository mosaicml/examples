# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Implements a Hugging Face BERT wrapped inside a :class:`.ComposerModel`."""

from __future__ import annotations

from typing import Optional

from composer.metrics.nlp import LanguageCrossEntropy, MaskedAccuracy
from composer.models.huggingface import HuggingFaceModel
from composer.utils.import_helpers import MissingConditionalImportError

__all__ = ['create_hf_bert_mlm']

def create_hf_bert_mlm(pretrained_model_name: str = 'bert-base-uncased',
                    use_pretrained: Optional[bool] = False,
                    model_config: Optional[dict] = None,
                    tokenizer_name: Optional[str] = None,
                    gradient_checkpointing: Optional[bool] = False):
    """BERT model based on |:hugging_face:| Transformers.

    For more information, see `Transformers <https://huggingface.co/transformers/>`_.

    Args:
        pretrained_model_name (str): Name of the Hugging Face model to instantiate. Default: ``'bert-base-uncased'``.
        use_pretrained (bool, optional): Whether to initialize the model with the pretrained weights. Default: ``False``.
        model_config (dict): The settings used to create a Hugging Face BertConfig. BertConfig is used to specify the
        architecture of a Hugging Face model.
        tokenizer_name (str, optional): Tokenizer name used to preprocess the dataset and validate the models inputs.
        gradient_checkpointing (bool, optional): Use gradient checkpointing. Default: ``False``.

        .. code-block::

            {
              "_name_or_path": "bert-base-uncased",
              "architectures": ["BertForMaskedLM"],
              "attention_probs_dropout_prob": 0.1,
              "classifier_dropout": null,
              "gradient_checkpointing": false,
              "hidden_act": "gelu",
              "hidden_dropout_prob": 0.1,
              "hidden_size": 768,
              "initializer_range": 0.02,
              "intermediate_size": 3072,
              "layer_norm_eps": 1e-12,
              "max_position_embeddings": 512,
              "model_type": "bert",
              "num_attention_heads": 12,
              "num_hidden_layers": 12,
              "pad_token_id": 0,
              "position_embedding_type": "absolute",
              "transformers_version": "4.16.0",
              "type_vocab_size": 2,
              "use_cache": true,
              "vocab_size": 30522
            }

   To create a BERT model for Masked Language Model pretraining:

    .. testcode::

        from src.hf_bert import create_hf_bert_mlm
        model = create_bert_mlm()

    """
    try:
        import transformers
    except ImportError as e:
        raise MissingConditionalImportError(extra_deps_group='nlp', conda_package='transformers') from e

    if not model_config:
        model_config = {}

    if not pretrained_model_name:
        pretrained_model_name = 'bert-base-uncased'

    if use_pretrained:
        assert transformers.AutoModelForMaskedLM.from_pretrained is not None, 'AutoModelForMaskedLM has from_pretrained method'
        model = transformers.AutoModelForMaskedLM.from_pretrained(pretrained_model_name_or_path=pretrained_model_name,
                                                                  **model_config)
    else:
        config = transformers.AutoConfig.from_pretrained(pretrained_model_name, **model_config)
        assert transformers.AutoModelForMaskedLM.from_config is not None, 'AutoModelForMaskedLM has from_config method'
        model = transformers.AutoModelForMaskedLM.from_config(config)

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
    return HuggingFaceModel(model=model, tokenizer=tokenizer, use_logits=True, metrics=metrics)

