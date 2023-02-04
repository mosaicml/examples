# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Any, Dict
from urllib.parse import urlparse

import torch
from composer.utils import reproducibility
from composer.utils.file_helpers import get_file
from omegaconf import OmegaConf as om
from transformers import AutoModelForCausalLM, AutoTokenizer

from examples.llm.src.huggingface.hf_causal_lm import ComposerHFCausalLM
from examples.llm.src.mosaic_gpt.model import ComposerMosaicGPT
from examples.llm.src.tokenizer import TOKENIZER_REGISTRY

CHECKPOINT_DOWNLOAD_DIR = f'{os.path.dirname(__file__)}/model_checkpoints'


def _get_checkpoint_name_from_path(path: str) -> str:
    """To go from checkpoint name to path, replace '/' with '|'."""
    return path.lstrip('/').replace('/', '|')


def download_to_local_path(path: str) -> str:
    """If necessary, downloads the file and returns a local path.

    Args:
        path (str): a local or http:// or s3:// path.
    """
    os.makedirs(CHECKPOINT_DOWNLOAD_DIR, exist_ok=True)

    parsed_path = urlparse(path)
    if parsed_path.scheme == '':
        local_path = path
    else:
        checkpoint_name = _get_checkpoint_name_from_path(parsed_path.path)
        local_path = os.path.join(CHECKPOINT_DOWNLOAD_DIR, checkpoint_name)
        print(f'Downloading from {path} to local path {local_path}')
        if not os.path.exists(local_path):
            get_file(path=path, destination=local_path, progress_bar=True)

    return local_path


def load_huggingface_checkpoint(
        pretrained_model_name_or_path: str) -> Dict[str, Any]:
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
    return {'model': model, 'tokenizer': tokenizer}


def load_composer_checkpoint(checkpoint_path: str,
                             config_path: str) -> Dict[str, Any]:
    """Load a MosaicGPT model and LLMTokenizer from a checkpoint.

    Constructs the MosaicGPT model and LLMTokenizer from the yaml config and
    attempts to initialize its weights from the state dict in `checkpoint`.
    If there is an error during state dict loading, returns a randomly
    initialized model.

    Args:
        checkpoint_path (str): Path to a Composer Pytorch .pt checkpoint. Must be located locally or on s3
        config_path (str): Path to a YAML config. Must be located locally or on s3

    Returns:
        model, tok, precision (Dict[str, Union[MosaicGPT, LLMTokenizer, str]]):
            Model and tokenizer to be used to build the lm_eval.base.LM wrapper
            as well as precision context.
    """
    print('Downloading checkpoint and config to local directory...')
    local_checkpoint_path = download_to_local_path(checkpoint_path)
    local_config_path = download_to_local_path(config_path)

    with open(local_config_path) as f:
        cfg = om.load(f)
    print('Building MosaicGPT w/ config: ')
    print('*' * 30)
    print(om.to_yaml(cfg.model, resolve=True))
    print(om.to_yaml(cfg.tokenizer, resolve=True))
    print('*' * 30)
    seed = cfg.seed if hasattr(cfg, 'seed') else 17
    reproducibility.seed_all(seed)

    # Build Model
    print('Initializing model...')
    cfg.model.init_device = 'cpu'

    if cfg.model.name == 'mosaic_gpt':
        model = ComposerMosaicGPT(cfg.model)
    elif cfg.model.name == 'hf_causal_lm':
        model = ComposerHFCausalLM(cfg.model)
    else:
        raise ValueError(
            f'Invalid composer model name={cfg.model.name=}. Must be either "mosaic_gpt" orr "hf_causal_lm"'
        )

    pre = next(model.parameters()).clone().data
    checkpoint = torch.load(local_checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['state']['model'])

    post = next(model.parameters()).clone().data
    assert not torch.equal(pre, post)
    print('Successfully loaded model weights')

    n_params = sum(p.numel() for p in model.parameters())
    print(f'{n_params=:.2e}')

    tokenizer = TOKENIZER_REGISTRY[cfg.tokenizer.type](**cfg.tokenizer.args)
    return {
        'model': model.model,
        'tokenizer': tokenizer,
        'precision': cfg.precision
    }
