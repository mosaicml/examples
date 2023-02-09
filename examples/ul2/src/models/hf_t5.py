# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Implements a Hugging Face BERT wrapped inside a :class:`.ComposerModel`."""

from __future__ import annotations

import inspect
from collections import UserDict
from typing import Optional

import torch
from composer.metrics.nlp import LanguageCrossEntropy, MaskedAccuracy
from composer.models.huggingface import HuggingFaceModel
from composer.utils.import_helpers import MissingConditionalImportError

from examples.common.hf_fsdp import prepare_hf_enc_dec_model_for_fsdp
from examples.ul2.src.super_glue.metrics import ExactMatch
from examples.ul2.src.utils import AutoTokenizerForMOD

__all__ = ['create_hf_t5']

# HuggingFace hardcodes the ignore index to -100
_HF_IGNORE_INDEX = -100


class HuggingFaceModelWithZLoss(HuggingFaceModel):

    def __init__(self, model, tokenizer, metrics, z_loss=0.0):
        super().__init__(model=model,
                         tokenizer=tokenizer,
                         metrics=metrics,
                         use_logits=True)
        self.z_loss = float(z_loss)
        assert self.z_loss >= 0.0

        self.model_forward_args = inspect.getfullargspec(
            self.model.forward).args

    def forward(self, batch):
        if isinstance(batch, dict) or isinstance(batch, UserDict):
            # Further input validation is left to the huggingface forward call
            batch = {
                k: v for k, v in batch.items() if k in self.model_forward_args
            }
            output = self.model(**batch)  # type: ignore (thirdparty)
        else:
            raise ValueError(
                'Unexpected batch type. Expected a dictionary with keys corresponding to the inputs to the forward function of the Huggingface model'
            )
        return output

    def loss(self, outputs, batch):
        if self.config.use_return_dict:
            loss, logits = outputs['loss'], outputs['logits']
        else:
            # loss is at index 0 in the output tuple, logits are at index 1
            loss, logits = outputs[:2]
        if self.z_loss == 0.0:
            return loss

        # Add a z_loss to the standard loss
        logits_flat = logits.view(-1, logits.size(-1))
        labels_flat = batch['labels'].view(-1)
        log_z = torch.logsumexp(logits_flat[labels_flat != _HF_IGNORE_INDEX],
                                dim=1)
        log_z2 = log_z**2
        z_loss = log_z2.mean() * self.z_loss
        if self.config.use_return_dict:
            outputs['loss'] += z_loss
            return outputs['loss']
        else:
            outputs[0] += z_loss
            return outputs[0]


def create_hf_t5(pretrained_model_name: str = 't5-base',
                 use_pretrained: Optional[bool] = False,
                 model_config: Optional[dict] = None,
                 tokenizer_name: Optional[str] = None,
                 z_loss: float = 0.0,
                 task_finetuning: Optional[bool] = False):
    """T5 model based on |:hugging_face:| Transformers.

    For more information, see `Transformers <https://huggingface.co/transformers/>`_.

    Args:
        pretrained_model_name (str): Name of the Hugging Face model to instantiate. Default: ``'t5-base'``.
        use_pretrained (bool, optional): Whether to initialize the model with the pretrained weights. Default: ``False``.
        model_config (dict): The settings used to create a Hugging Face T5Config. T5Config is used to specify the
            architecture of a Hugging Face model.
        tokenizer_name (str, optional): Tokenizer name used to preprocess the dataset and validate the models inputs.
        z_loss (float, optional): Coefficient of `z-loss` term to use during training. Default: ``0.0``.
        task_finetuning (bool, optional): Whether to add ExactMatch metric used in fine-tuning. Default: ``False``.

    To create a |:hugging_face:| T5 model for Masked Language Model pretraining:

    .. testcode::

        from examples.ul2.src.hf_t5 import create_hf_t5
        model = create_hf_t5('t5-base')
    """
    try:
        from transformers import T5Config, T5ForConditionalGeneration
    except ImportError as e:
        raise MissingConditionalImportError(extra_deps_group='nlp',
                                            conda_package='transformers') from e

    if not pretrained_model_name:
        pretrained_model_name = 't5-base'

    # setup the tokenizer
    tokenizer_name = tokenizer_name or pretrained_model_name
    tokenizer = AutoTokenizerForMOD.from_pretrained(tokenizer_name)

    if not model_config:
        model_config = {}

    if use_pretrained:
        model = T5ForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name, **model_config)
    else:
        config = T5Config.from_pretrained(pretrained_model_name, **model_config)
        model = T5ForConditionalGeneration(config)

    # Super charge
    prepare_hf_enc_dec_model_for_fsdp(model)

    # We use `len(tokenizer) instead of `model.config.vocab_size` because
    # `HuggingFaceModel` will set the latter to the former
    metrics = [
        # Note: The HF code is hardcoded to use -100 as the ignore index
        LanguageCrossEntropy(ignore_index=_HF_IGNORE_INDEX,
                             vocab_size=len(tokenizer)),
        MaskedAccuracy(ignore_index=_HF_IGNORE_INDEX)
    ]
    if task_finetuning:
        metrics.append(ExactMatch(ignore_index=_HF_IGNORE_INDEX))
    return HuggingFaceModelWithZLoss(model=model,
                                     tokenizer=tokenizer,
                                     metrics=metrics,
                                     z_loss=z_loss)
