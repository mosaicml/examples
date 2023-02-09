# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
from collections import UserDict
from typing import Optional

import torch
from composer.metrics.nlp import LanguageCrossEntropy, MaskedAccuracy
from composer.models.huggingface import HuggingFaceModel
from composer.utils.import_helpers import MissingConditionalImportError

from examples.common.hf_fsdp import prepare_hf_causal_lm_model_for_fsdp
from examples.ul2.src import utils
from examples.ul2.src.super_glue.metrics import ExactMatch

__all__ = ['create_hf_prefix_lm']

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

        # Shift the labels since this is for decoder-only models
        labels_shifted = torch.full_like(batch['labels'], _HF_IGNORE_INDEX)
        labels_shifted[..., :-1] = batch['labels'][..., 1:]
        labels_flat = labels_shifted.contiguous().view(-1)
        # Add a z_loss to the standard loss
        logits_flat = logits.view(-1, logits.size(-1))
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


def create_hf_prefix_lm(pretrained_model_name: str = 'gpt2',
                        tokenizer_name: str = 'gpt2',
                        use_pretrained: Optional[bool] = False,
                        model_config: Optional[dict] = None,
                        gradient_checkpointing: Optional[bool] = False,
                        z_loss: Optional[float] = 0.0,
                        task_finetuning: Optional[bool] = False):
    """Prefix-LM model based on |:hugging_face:| Transformers.

    For more information, see `Transformers <https://huggingface.co/transformers/>`_.

    Note: HuggingFace does not natively support Prefix-LM-style models. This function uses
    `transformers.AutoModelForCausalLM` to instantiate a Causal-LM, then uses a conversion utility
    to turn the model into a Prefix-LM. Currently, that conversion utility only supports the
    following HuggingFace Causal-LM types:
        - `GPT2LMHeadModel`
        - `GPTNeoForCausalLM`
        - `GPTNeoXForCausalLM`
        - `GPTJForCausalLM`
        - `BloomForCausalLM`
        - `OPTForCausalLM`

    Args:
        pretrained_model_name (str): Name of the Hugging Face model to instantiate. Default: ``'gpt2'``.
        tokenizer_name (str): Tokenizer name used to preprocess the dataset and validate the models inputs. To be compatible
            with denoising tasks, special sentinel tokens ("<extra_id_0>", "<extra_id_1>", etc.) will be added. The model's vocab_size
            will be hardcoded to match the resulting vocab_size of the tokenizer.
        use_pretrained (bool, optional): Whether to initialize the model with the pretrained weights. Default: ``False``.
        model_config (dict, optional): The settings used to create a Hugging Face CausalLM. Used to specify/modify
            the architecture of the Hugging Face model.
        gradient_checkpointing (bool, optional): Use gradient checkpointing. Default: ``False``.
        z_loss (float, optional): Coefficient of `z-loss` term to use during training. Default: ``0.0``.
        task_finetuning (bool, optional): Whether to add ExactMatch metric used in fine-tuning. Default: ``False``.

    To create a |:hugging_face:| Prefix-LM model for UL2 pretraining:

    .. testcode::

        from examples.ul2.src.hf_prefix_lm import create_prefix_lm
        model = create_prefix_lm('gpt2')
    """
    try:
        import transformers
    except ImportError as e:
        raise MissingConditionalImportError(extra_deps_group='nlp',
                                            conda_package='transformers') from e

    # Set up the tokenizer (add tokens for denoising sentinels if needed)
    tokenizer = utils.AutoTokenizerForMOD.from_pretrained(tokenizer_name)
    vocab_size = len(tokenizer)

    if not model_config:
        model_config = {}

    if not pretrained_model_name:
        pretrained_model_name = 'gpt2'

    if use_pretrained:
        assert transformers.AutoModelForCausalLM.from_pretrained is not None, 'AutoModelForCausalLM has from_pretrained method'
        model = transformers.AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name, **model_config)
    else:
        config = transformers.AutoConfig.from_pretrained(
            pretrained_model_name, **model_config)
        assert transformers.AutoModelForCausalLM.from_config is not None, 'AutoModelForCausalLM has from_config method'
        model = transformers.AutoModelForCausalLM.from_config(config)

    # Super charge
    prepare_hf_causal_lm_model_for_fsdp(model)

    # Convert the Causal LM into a Prefix LM via our custom wrapper
    model = utils.convert_hf_causal_lm_to_prefix_lm(model)

    # Expand the embeddings/vocab size to match the tokenizer
    if model.config.vocab_size != vocab_size:
        model.resize_token_embeddings(new_num_tokens=vocab_size)

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()  # type: ignore

    metrics = [
        LanguageCrossEntropy(ignore_index=_HF_IGNORE_INDEX,
                             vocab_size=model.config.vocab_size),
        MaskedAccuracy(ignore_index=_HF_IGNORE_INDEX)
    ]
    if task_finetuning:
        metrics.append(ExactMatch(ignore_index=_HF_IGNORE_INDEX))
    return HuggingFaceModelWithZLoss(model=model,
                                     tokenizer=tokenizer,
                                     metrics=metrics,
                                     z_loss=z_loss)
