# Copyright 2022 MosaicML Benchmarks authors
# SPDX-License-Identifier: Apache-2.0

import os
import torch
from omegaconf import OmegaConf as om

from src.data_c4 import build_c4_dataloader


def get_config(conf_path="yamls/mosaic_gpt/125m.yaml"):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    with open(conf_path) as f:
        test_cfg = om.load(f)
    return test_cfg


def test_correct_padding(batch_size=32):
    test_cfg = get_config(conf_path="yamls/mosaic_gpt/125m.yaml")

    # Dataloaders
    eval_loader = build_c4_dataloader(test_cfg.eval_loader, batch_size)
    batch = next(iter(eval_loader))
    
    assert batch['input_ids'].shape == torch.Size([batch_size, 2048])
    assert batch['input_ids'].type() == 'torch.LongTensor'

    # we follow the convention (from huggingface) that non-attended tokens are 0 in the attn mask and -100 in the labels
    a =  batch['attention_mask'] == 0
    b =  batch['labels'] == -100
    assert torch.equal(a, b)
