# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

"""Implements a Hugging Causal LM wrapped inside a :class:`.ComposerModel`."""

from accelerate import init_empty_weights
from composer.metrics.nlp import LanguageCrossEntropy, Perplexity
from omegaconf import DictConfig
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from examples.llm.src.models.hf.model_wrapper import HuggingFaceModelWithZLoss

__all__ = ['ComposerHFCausalLM']


class ComposerHFCausalLM(HuggingFaceModelWithZLoss):
    """Configures a :class:`.HuggingFaceModel` around a Causal LM.

    Args:
        cfg (DictConfig): An omegaconf dictionary used to configure the model:
            cfg.pretrained_model_name_or_path (str): The name of or local path to
                the HF Causal LM (e.g., `gpt2` to instantiate a GPT2LMHeadModel).
            cfg.config_overrides (dict, optional): An optional dictionary of keyword
                arguments that override the default configuration associated with
                cfg.pretrained_model_name_or_path.
            cfg.pretrained (bool): Whether to instantiate the model with pre-trained
                weights coming from cfg.pretrained_model_name_or_path. If ``True``,
                cfg.config_overrides must be compatible with the pre-trained weights.
            cfg.init_device ('cpu' | 'meta'): Which device, 'cpu' or 'meta', to
                initialize the model on. Currently, `meta` is only supported when
                cfg.pretrained is ``False``. Default: ``'cpu'``.
            cfg.add_exact_match (bool, optional): CURRENTLY UNUSED. Whether to add ExactMatch metric used
                in some fine-tuning settings. Default: ``False``.
            cfg.add_rouge (bool, optional): CURRENTLY UNUSED. Whether to add RougeWithDetokenizer metric
                to validation metrics. Default: ``False``.
    """

    def __init__(self, cfg: DictConfig):
        config = AutoConfig.from_pretrained(cfg.pretrained_model_name_or_path,
                                            **cfg.get('config_overrides', {}))

        tokenizer = AutoTokenizer.from_pretrained(
            cfg.pretrained_model_name_or_path)

        metrics = [LanguageCrossEntropy(len(tokenizer)), Perplexity()]

        init_device = cfg.get('init_device', 'cpu')
        if init_device == 'cpu':
            if cfg.pretrained:
                model = AutoModelForCausalLM.from_pretrained(
                    cfg.pretrained_model_name_or_path, config=config)
            else:
                model = AutoModelForCausalLM.from_config(config)
        elif init_device == 'meta':
            if cfg.pretrained:
                raise ValueError(
                    'Setting cfg.pretrained=True is not supported when init_device="meta".'
                )
            with init_empty_weights(include_buffers=False):
                model = AutoModelForCausalLM.from_config(config)
        else:
            raise ValueError(
                f'init_device="{init_device}" must be either "cpu" or "meta".')

        # if cfg.add_exact_match:
        #     metrics.append(ExactMatch(ignore_index=_HF_IGNORE_INDEX))

        composer_model = super().__init__(model=model,
                                          tokenizer=tokenizer,
                                          metrics=metrics,
                                          z_loss=cfg.get('z_loss', 0.0))

        # if cfg.add_rouge:
        #     rouge_metric = RougeWithDetokenizer(detokenizer=tokenizer)
        #     composer_model.val_metrics[RougeWithDetokenizer.__name__] = rouge_metric

        return composer_model
