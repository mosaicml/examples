# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import warnings
import pytest

import torch
from composer.utils import reproducibility
from omegaconf import OmegaConf as om

from src.mosaic_gpt import ComposerMosaicGPT
from src.hf_gpt2 import ComposerHFCausalLM


def test_compare_hf_v_mosaic_gpt():
    warnings.filterwarnings(action='ignore', message='Torchmetrics v0.9 introduced a new argument class property')

    with open('yamls/mosaic_gpt/125m.yaml') as f:
        cfg = om.load(f)
    
    print("Training using config: ")
    print(om.to_yaml(cfg))

    # modify seq len for HF GPT2 compatibility
    cfg.max_seq_len = 1024
    cfg.model.max_seq_len = cfg.max_seq_len
    cfg.train_loader.max_seq_len = cfg.max_seq_len
    cfg.eval_loader.max_seq_len = cfg.max_seq_len
    # set dropout prob
    cfg.model.resid_pdrop = 0.1
    cfg.model.emb_pdrop = 0.1

    # set seed
    reproducibility.seed_all(cfg.seed)

    device = 'cuda'
    cfg.model.device = device

    # Build Model
    print('Initializing model...')

    model = ComposerMosaicGPT(cfg.model).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    # print(f'{n_params=:.5e}')

    hf_cfg = om.create({'name': 'hf_gpt2', 'hf_config': 'gpt2'})
    hf_model = ComposerHFCausalLM(hf_cfg).to(device)
    hf_n_params = sum(p.numel() for p in hf_model.parameters())
    # print(f'{hf_n_params=:.5e}')

    assert hf_n_params == n_params

    # Get batch size info
    batch_size = 2
    
    # get random input data
    batch = {}
    batch['input_ids']      = torch.randint(low=0, high=49673, size=(batch_size, cfg.max_seq_len)).to(device)
    batch['attention_mask'] = torch.ones(size=(batch_size, cfg.max_seq_len), dtype=torch.int64).to(device)
    batch['labels']         = torch.randint(low=0, high=49673, size=(batch_size, cfg.max_seq_len)).to(device)

    model.model.transformer.emb_drop.p = 0.0
    hf_model.model.transformer.drop.p = 0.0

    for b in model.model.transformer.blocks:    b.resid_mlp_dropout.p = 0.0
    for b in hf_model.model.transformer.h:      b.mlp.dropout.p = 0.0

    for b in model.model.transformer.blocks:    b.resid_attn_dropout.p = 0.0
    for b in hf_model.model.transformer.h:      b.attn.resid_dropout.p = 0.0

    # for b in model.model.transformer.blocks:    b.attn_dropout.p = 0.0 # set in cfg
    for b in hf_model.model.transformer.h:      b.attn.attn_dropout.p = 0.0

    hf_model.train()
    model.train()

    # UTIL: can be used to verify that models are not the same at init
    # with torch.autocast(device_type='cuda', dtype=torch.float16):
    #     torch.manual_seed(0)
    #     hf_model_fwd = hf_model(batch)
    #     _hf_model_fwd = hf_model_fwd
    #     torch.manual_seed(0)
    #     model_fwd = model(batch)
    # can be used to verify that models are not the same at init
    # print(f"{hf_model_fwd['logits'].mean().item()=}")
    # print(f"{model_fwd.mean().item()=}")

    hf_model_statedict = hf_model.state_dict()

    # convert hf gpt statedict to mosaic gpt statedict
    hf_keys_ignore = [".attn.masked_bias", ".attn.bias"]
    _transpose = [".attn.c_attn.", ".attn.c_proj.", ".mlp.c_fc.", ".mlp.c_proj."]
    hf_2_mosaic_key_mods = {
        "model.transformer.h.": "model.transformer.blocks.",
        ".attn.c_attn.":        ".causal_attn.mha.Wqkv.",
        ".attn.c_proj.":        ".causal_attn.mha.out_proj.",
        ".mlp.c_fc.":           ".mlp.mlp_up.",
        ".mlp.c_proj.":         ".mlp.mlp_down.",
    }

    _hf_model_statedict = {}
    for k, v in hf_model_statedict.items():
        skip = False
        for _k in hf_keys_ignore:
            if _k in k:
                skip = True
                continue
        for _k in _transpose:
            if _k in k:
                v = v.t()

        for _k, _v in hf_2_mosaic_key_mods.items():
            if _k in k:
                k = k.replace(_k, _v)
        if not skip:
            _hf_model_statedict[k] = v

    model.load_state_dict(_hf_model_statedict)

    with torch.autocast(device_type='cuda', dtype=torch.float16):
        torch.manual_seed(0)
        hf_model_fwd = hf_model(batch)
        torch.manual_seed(0)
        model_fwd = model(batch)

    print(f"{hf_model_fwd['logits'].mean().item()=}")
    print(f"{model_fwd.mean().item()=}")

    # When dropouts are aligned, these are near identical
    assert hf_model_fwd['logits'].mean().allclose(model_fwd.mean())

    assert hf_model_fwd['logits'].allclose(model_fwd, rtol=1e-02, atol=1e-02)
