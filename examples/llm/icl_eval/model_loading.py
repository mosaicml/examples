# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Optional, Union

import torch
import transformers
from composer.metrics.nlp import HFCrossEntropy, Perplexity
from composer.models.huggingface import HuggingFaceModel
from composer.utils import reproducibility
from omegaconf import OmegaConf as om
from src.model_registry import COMPOSER_MODEL_REGISTRY


def init_huggingface_causal_lm(model_name: str) -> Dict[str, HuggingFaceModel]:
    model = transformers.AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_name, **{})
    return {
        'model':
            HuggingFaceModel(model=model,
                             tokenizer=None,
                             metrics=[HFCrossEntropy(),
                                      Perplexity()])
    }


def init_composer_config_from_yaml(
    config: str,) -> Dict[str, Union[torch.nn.Module, Dict]]:
    """Construct a MosaicGPT model from a config file and the FSDP config.

    Constructs the MosaicGPT model from the yaml config and extracts the FSDP config
    to be passed to Trainer.

    Args:
        config (str): YAML config. Must be an absolute path

    Returns:
        dictionary Dict[str, Union[MosaicGPT, Dict]]:
            Model and FSDP config to be passed into Trainer
    """
    with open(config) as f:
        cfg = om.load(f)
    print('Building MosaicGPT w/ config: ')
    print(om.to_yaml(cfg))
    seed = cfg.seed if hasattr(cfg, 'seed') else 17
    reproducibility.seed_all(seed)

    # Build Model
    print('Initializing model...')
    if 'fsdp_config' not in cfg:
        cfg.model.device = 'cpu'

    model = COMPOSER_MODEL_REGISTRY[cfg.model.name](cfg.model)

    fsdp_config = cfg.get('fsdp_config', None)
    fsdp_config = om.to_container(fsdp_config,
                                  resolve=True) if fsdp_config else None

    n_params = sum(p.numel() for p in model.parameters())
    print(f'{n_params=:.2e}')

    return {
        'model': model,
        'fsdp_config': fsdp_config,
    }


def load_model(model_type: str, config: str, checkpoint: Optional[str] = None):
    if model_type == 'pretrained_hf':
        return init_huggingface_causal_lm(config)
    elif model_type == 'mosaic_gpt':
        ret = init_composer_config_from_yaml(config)
        ret['checkpoint'] = checkpoint
        return ret
    else:
        raise Exception(f'Unrecogized model type: {model_type}')
