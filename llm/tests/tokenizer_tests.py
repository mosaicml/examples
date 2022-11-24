# Copyright 2022 MosaicML Benchmarks authors
# SPDX-License-Identifier: Apache-2.0

import unittest
from omegaconf import OmegaConf as om

from src.tokenizer import TOKENIZER_REGISTRY


class TokenizerTests(unittest.TestCase):
    def setUp(self):
        conf_path = "yamls/mosaic_gpt/125m.yaml"
        with open(conf_path) as f:
            self.test_cfg = om.load(f)

    def test_load_tokenizer(self):
        truncation = True
        padding = 'max_length'

        tokenizer = TOKENIZER_REGISTRY[self.test_cfg.tokenizer.type](**self.test_cfg.tokenizer.args)
        self.assertEqual(tokenizer.tokenizer.vocab_size, 50257)
        self.assertEqual(tokenizer.tokenizer.name_or_path, 'gpt2')

        # test explicitly call tokenizer
        self.assertEqual(tokenizer.tokenizer.encode("hello\n\nhello"), [
                31373,
                198,
                198,
                31373])

        # tokenizer __call__
        self.assertEqual(tokenizer.tokenizer("hello\n\nhello")['input_ids'], [
                31373,
                198,
                198,
                31373])
        
        # tokenizer  __call__ with kwargs
        padded_tokenize = tokenizer.tokenizer("hello\n\nhello", 
                truncation=truncation,
                padding=padding,
                max_length=tokenizer.max_seq_len
            )['input_ids']        
        self.assertEqual(
            padded_tokenize, [
                31373,
                198,
                198,
                31373] + [50256] * (tokenizer.max_seq_len - 4))

        # wrapper class __call__
        self.assertEqual(tokenizer("hello\n\nhello")['input_ids'], [
                31373,
                198,
                198,
                31373])

        # wrapper class __call__ with kwargs
        padded_tokenize = tokenizer("hello\n\nhello", 
                truncation=truncation,
                padding=padding,
                max_length=tokenizer.max_seq_len
            )['input_ids']    

        attention_mask = tokenizer("hello\n\nhello", 
                truncation=truncation,
                padding=padding,
                max_length=tokenizer.max_seq_len
            )['attention_mask']    
        self.assertEqual(
            attention_mask, [
                1,
                1,
                1,
                1] + [0] * (tokenizer.max_seq_len - 4))
