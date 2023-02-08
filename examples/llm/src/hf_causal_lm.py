# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F
from composer.metrics.nlp import (HFCrossEntropy, LanguageCrossEntropy,
                                  Perplexity)
from composer.models.huggingface import HuggingFaceModel
from omegaconf import DictConfig
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutput

from examples.common.hf_fsdp import prepare_hf_causal_lm_model_for_fsdp


class ComposerHFCausalLM(HuggingFaceModel):

    def __init__(self, cfg: DictConfig):
        config = AutoConfig.from_pretrained(cfg.pretrained_model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.pretrained_model_name_or_path)

        if cfg.pretrained:
            model = AutoModelForCausalLM.from_pretrained(
                cfg.pretrained_model_name_or_path, config=config)
            metrics = [HFCrossEntropy(), Perplexity()]
        else:
            model = AutoModelForCausalLM.from_config(config)
            metrics = [LanguageCrossEntropy(len(tokenizer)), Perplexity()]

        prepare_hf_causal_lm_model_for_fsdp(model)

        super().__init__(model=model,
                         tokenizer=tokenizer,
                         metrics=metrics,
                         use_logits=True)

    def get_targets(self, batch: dict):
        targets = torch.roll(batch['labels'], shifts=-1)
        targets[:, -1] = -100
        return targets

    def forward(self, batch: dict) -> CausalLMOutput:
        return self.model(input_ids=batch['input_ids'],
                          attention_mask=batch['attention_mask'].bool())

    def loss(self, outputs: CausalLMOutput, batch: dict):
        logits = outputs.logits
        targets = self.get_targets(batch)
        return F.cross_entropy(logits.view(-1, logits.size(-1)),
                               targets.view(-1),
                               ignore_index=-100)
