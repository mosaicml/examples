# Copyright 2022 MosaicML Benchmarks authors
# SPDX-License-Identifier: Apache-2.0

import os
import unittest
import warnings
import torch
import torch.nn as nn
from omegaconf import OmegaConf as om
from composer.utils import reproducibility
from composer.optim import DecoupledAdamW

from main import calculate_batch_size_info
from src.data_c4 import build_c4_dataloader

from src.tokenizer import TOKENIZER_REGISTRY
from src.model_registry import COMPOSER_MODEL_REGISTRY


class ModelTests(unittest.TestCase):
    def setUp(self):
        warnings.filterwarnings(action='ignore', message='Torchmetrics v0.9 introduced a new argument class property')
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        conf_path = "yamls/mosaic_gpt/125m.yaml"
        with open(conf_path) as f:
            self.test_cfg = om.load(f)
        self.tokenizer = TOKENIZER_REGISTRY[self.test_cfg.tokenizer.type](**self.test_cfg.tokenizer.args)
        
        reproducibility.seed_all(self.test_cfg.seed)

        # Read FSDP Config as a dict
        fsdp_config = self.test_cfg.get('fsdp_config', None)
        fsdp_config = om.to_container(fsdp_config, resolve=True) if fsdp_config else None

        # Build Model
        # For fast initialization, use `meta` device
        print('Initializing model...')
        device = 'cpu'
        self.test_cfg.precision = 'fp32'
        self.test_cfg.model.attn_impl = 'torch'
        # device = 'cuda'
        # self.test_cfg.precision = 'amp'
        self.test_cfg.model.device = device
        self.test_cfg.device = device

        self.test_cfg.global_train_batch_size = 2
        self.test_cfg.device_eval_batch_size = 2
        self.test_cfg.device_train_microbatch_size = 2

        self.model = COMPOSER_MODEL_REGISTRY[self.test_cfg.model.name](self.test_cfg.model)
        # Optimizer
        assert self.test_cfg.optimizer.name == 'decoupled_adamw'
        self.optimizer = DecoupledAdamW(
            self.model.parameters(),
            lr=self.test_cfg.optimizer.lr,
            betas=self.test_cfg.optimizer.betas,
            eps=self.test_cfg.optimizer.eps,
            weight_decay=self.test_cfg.optimizer.weight_decay)
    
    def test_full_forward_and_backward(self):
        # Get batch size info
        device_batch_size, _, _ = calculate_batch_size_info(
            self.test_cfg.global_train_batch_size, self.test_cfg.device_train_microbatch_size)

        # Dataloaders
        eval_loader = build_c4_dataloader(self.test_cfg.eval_loader, device_batch_size)
        batch = next(iter(eval_loader))
        batch['input_ids'] = batch['input_ids'].to(self.test_cfg.device)
        batch['labels'] = batch['labels'].to(self.test_cfg.device)
        batch['attention_mask'] = batch['attention_mask'].to(self.test_cfg.device)
        
        self.assertEqual(batch['input_ids'].shape, torch.Size([device_batch_size, self.test_cfg.max_seq_len]))
        self.model.train()
        original_params = next(self.model.parameters()).clone().data
        outputs = self.model(batch)
        loss = self.model.loss(outputs, batch)
        loss.backward()
        self.optimizer.step()
        updated_params = next(self.model.parameters()).clone().data
        self.assertFalse(torch.equal(original_params, updated_params))

    def test_attention_mechanism(self):
        # Get batch size info
        device_batch_size, _, _ = calculate_batch_size_info(
            self.test_cfg.global_train_batch_size, self.test_cfg.device_train_microbatch_size)

        # Dataloaders
        eval_loader = build_c4_dataloader(self.test_cfg.eval_loader, device_batch_size)
        batch = next(iter(eval_loader))
        self.assertEqual(batch['input_ids'].shape, torch.Size([device_batch_size, self.test_cfg.max_seq_len]))
        
        self.model.eval()
        # run a partial forward where we explicitly inspect the attention_mask from the causal_attn block
        input_ids, key_padding_mask = batch['input_ids'], batch['attention_mask'].bool()

        _, S = input_ids.size()
        assert (
            S <= self.test_cfg.max_seq_len
        ), f"Cannot forward input with seq_len={S}, this model only supports seq_len<={self.test_cfg.max_seq_len}"
        pos = torch.arange(
            0, S, dtype=torch.long,
            device=input_ids.device).unsqueeze(0)

        tok_emb = self.model.model.transformer.wte(input_ids)
        pos_emb = self.model.model.transformer.wpe(pos)
        x = self.model.model.transformer.emb_drop(tok_emb + pos_emb)

        # basically the attention mask should be a tensor shape (bsz, seqlen, seqlen)
        # wih -inf along the upper triangle as well as wherever there are any pad tokens
        # and with 0 everywhere else
        expected_zerod_weights = nn.Transformer.generate_square_subsequent_mask(self.test_cfg.max_seq_len)\
            .reshape(1, self.test_cfg.max_seq_len, self.test_cfg.max_seq_len)
        expected_zerod_weights = torch.isneginf(torch.cat(
            device_batch_size*[expected_zerod_weights]
        ))
        torch_key_padding = torch.cat(
            self.test_cfg.max_seq_len*[(~key_padding_mask).reshape(device_batch_size, 1, self.test_cfg.max_seq_len)], 
            axis=1)
        expected_zerod_weights |= torch_key_padding
        

        for block in self.model.model.transformer.blocks:
            a = block.ln_1(x)
            b, attention_weights = block.causal_attn(a, key_padding_mask)

            zerod_weights = (attention_weights == 0)
            self.assertTrue(torch.equal(expected_zerod_weights, zerod_weights))
            x = x + block.resid_attn_dropout(b)
            m = block.ln_2(x)
            n = block.mlp(m)
            x = x + block.resid_mlp_dropout(n)

    def test_full_forward_and_backward_gpt_neo(self):
        warnings.filterwarnings(action='ignore', message='Torchmetrics v0.9 introduced a new argument class property')
        conf_path = "yamls/hf_causal_lm/gpt-neo-125m.yaml"
        with open(conf_path) as f:
            neo_cfg = om.load(f)
        device = self.test_cfg.device

        model = COMPOSER_MODEL_REGISTRY[neo_cfg.model.name](neo_cfg.model).to(device)

        assert neo_cfg.optimizer.name == 'decoupled_adamw'
        optimizer = DecoupledAdamW(
            model.parameters(),
            lr=neo_cfg.optimizer.lr,
            betas=neo_cfg.optimizer.betas,
            eps=neo_cfg.optimizer.eps,
            weight_decay=neo_cfg.optimizer.weight_decay)
        # Get batch size info
        neo_cfg.device_train_microbatch_size = 2
        device_batch_size, _, _ = calculate_batch_size_info(
            self.test_cfg.global_train_batch_size, self.test_cfg.device_train_microbatch_size)

        # Dataloaders
        eval_loader = build_c4_dataloader(neo_cfg.eval_loader, device_batch_size)
        batch = next(iter(eval_loader))
        batch['input_ids'] = batch['input_ids'].to(self.test_cfg.device)
        batch['labels'] = batch['labels'].to(self.test_cfg.device)
        batch['attention_mask'] = batch['attention_mask'].to(self.test_cfg.device)
        
        self.assertEqual(batch['input_ids'].shape, torch.Size([device_batch_size, self.test_cfg.max_seq_len]))
        model.train()
        original_params = next(model.parameters()).clone().data
        outputs = model(batch)
        loss = model.loss(outputs, batch)
        loss.backward()
        optimizer.step()
        updated_params = next(model.parameters()).clone().data
        self.assertFalse(torch.equal(original_params, updated_params))
