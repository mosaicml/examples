# Copyright 2022 MosaicML Benchmarks authors
# SPDX-License-Identifier: Apache-2.0

import os
import unittest
import torch
from omegaconf import OmegaConf as om

from src.data_c4 import build_c4_dataloader
from main import calculate_batch_size_info


class DataloaderTests(unittest.TestCase):
    def setUp(self):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        conf_path = "yamls/mosaic_gpt/125m.yaml"
        with open(conf_path) as f:
            self.test_cfg = om.load(f)
    
    def test_correct_padding(self):
        # Get batch size info
        device_batch_size, _, _ = calculate_batch_size_info(
            self.test_cfg.global_train_batch_size, self.test_cfg.device_train_microbatch_size)

        # Dataloaders
        eval_loader = build_c4_dataloader(self.test_cfg.eval_loader, device_batch_size)
        batch = next(iter(eval_loader))
        
        self.assertEqual(batch['input_ids'].shape, torch.Size([device_batch_size, 2048]))
        self.assertEqual(batch['input_ids'].type(), 'torch.LongTensor')
        # we follow the convention (from huggingface) that non-attended tokens are 0 in the attn mask and -100 in the labels
        a =  batch['attention_mask'] == 0
        b =  batch['labels'] == -100
        self.assertTrue(torch.equal(a, b))
