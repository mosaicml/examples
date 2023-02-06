# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Optional, Union

import torch
from composer.models.gpt2 import create_gpt2
from composer.utils import reproducibility
from omegaconf import OmegaConf as om
from src.model_registry import COMPOSER_MODEL_REGISTRY
from src.tokenizer import LLMTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer


def init_huggingface_causal_lm(
        config: str) -> Dict[str, Union[AutoModelForCausalLM, AutoTokenizer]]:
    return {
        'model': create_gpt2(use_pretrained=True, pretrained_model_name=config)
    }


def init_composer_ckpt_from_yaml(
    config: str, checkpoint: Optional[str]
) -> Dict[str, Union[torch.nn.Module, LLMTokenizer, str]]:
    """Load a MosaicGPT model and LLMTokenizer from a checkpoint.

    Constructs the MosaicGPT model and LLMTokenizer from the yaml config and
    attempts to initialize its weights from the state dict in `checkpoint`.
    If there is an error during state dict loading, returns a randomly
    initialized model.

    Args:
        checkpoint (str): Pytorch .pt checkpoint path. Must be located in
            `CHECKPOINT_DIR` or on s3
        config (str): YAML config. Must be an absolute path

    Returns:
        model [MosaicGPT]:
            Model and tokenizer to be used to build the lm_eval.base.LM wrapper
            as well as precision context.
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
        'checkpoint': checkpoint
    }


def load_model(model_type: str, config: str, checkpoint: Optional[str] = None):
    if model_type == 'pretrained_hf':
        return init_huggingface_causal_lm(config)
    elif model_type == 'mosaic_gpt':
        return init_composer_ckpt_from_yaml(config, checkpoint)
    else:
        raise Exception(f'Unrecogized model type: {model_type}')
