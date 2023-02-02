# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

import copy
import os
import warnings
from typing import cast

import pytest
import torch
import torch.nn as nn
from composer.optim import DecoupledAdamW
from composer.utils import reproducibility
from omegaconf import DictConfig
from omegaconf import OmegaConf as om
from examples.llm.src.model_registry import COMPOSER_MODEL_REGISTRY
from examples.llm.src.tokenizer import TOKENIZER_REGISTRY


def get_config(conf_path='yamls/mosaic_gpt/branching_125m.yaml') -> DictConfig:
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    print(conf_path)
    with open(conf_path) as f:
        test_cfg = om.load(f)
    return cast(DictConfig, test_cfg)


def get_objs(conf_path='yamls/mosaic_gpt/branching_125m.yaml'):
    warnings.filterwarnings(
        action='ignore',
        message='Torchmetrics v0.9 introduced a new argument class property')
    test_cfg = get_config(conf_path=conf_path)
    _ = TOKENIZER_REGISTRY[test_cfg.tokenizer.type](
        **test_cfg.tokenizer.args)  # make sure tokenizer in registry

    reproducibility.seed_all(test_cfg.seed)

    # Read FSDP Config as a dict
    fsdp_config = test_cfg.get('fsdp_config', None)
    fsdp_config = om.to_container(fsdp_config,
                                  resolve=True) if fsdp_config else None

    # Build Model
    # For fast initialization, use `meta` device
    print('Initializing model...')
    device = 'cpu'
    test_cfg.precision = 'fp32'
    test_cfg.model.attn_impl = 'torch'
    # device = 'cuda'
    # test_cfg.precision = 'amp'
    test_cfg.model.device = device
    test_cfg.device = device

    test_cfg.global_train_batch_size = 2
    test_cfg.device_eval_batch_size = 2
    test_cfg.device_train_microbatch_size = 2

    model = COMPOSER_MODEL_REGISTRY[test_cfg.model.name](test_cfg.model)
    # Optimizer
    assert test_cfg.optimizer.name == 'decoupled_adamw'
    optimizer = DecoupledAdamW(model.parameters(),
                               lr=test_cfg.optimizer.lr,
                               betas=test_cfg.optimizer.betas,
                               eps=test_cfg.optimizer.eps,
                               weight_decay=test_cfg.optimizer.weight_decay)

    return test_cfg, model, optimizer


def gen_random_batch(batch_size, test_cfg):
    # generate input batch of random data
    batch = {}
    batch['input_ids'] = torch.randint(
        low=0,
        high=test_cfg.model.vocab_size,
        size=(batch_size, test_cfg.max_seq_len)).to(test_cfg.device)
    batch['labels'] = torch.randint(low=0,
                                    high=test_cfg.model.vocab_size,
                                    size=(batch_size, test_cfg.max_seq_len)).to(
                                        test_cfg.device)
    batch['attention_mask'] = torch.ones(size=(batch_size,
                                               test_cfg.max_seq_len),
                                         dtype=torch.int64).to(test_cfg.device)
    return batch


def test_full_forward_and_backward(batch_size=2):
    test_cfg, model, optimizer = get_objs(
        conf_path='yamls/mosaic_gpt/branching_125m.yaml')

    batch = gen_random_batch(batch_size, test_cfg)

    assert batch['input_ids'].shape == torch.Size(
        [batch_size, test_cfg.max_seq_len])
    model.train()
    original_params = next(model.parameters()).clone().data
    outputs = model(batch)
    loss = model.loss(outputs, batch)
    loss.backward()
    optimizer.step()
    updated_params = next(model.parameters()).clone().data
    assert not torch.equal(original_params, updated_params)


    metrics = model.get_metrics()
    for m in metrics.values():
        model.update_metric(batch, outputs, m)
