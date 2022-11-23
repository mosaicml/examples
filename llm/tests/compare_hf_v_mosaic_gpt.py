# Copyright 2022 MosaicML Benchmarks authors
# SPDX-License-Identifier: Apache-2.0

import pytest
import warnings

import torch
from omegaconf import OmegaConf as om
from composer.utils import reproducibility

from main import build_composer_model


def test_compare_hf_v_mosaic_gpt():
    warnings.filterwarnings(action='ignore', message='Torchmetrics v0.9 introduced a new argument class property')
    device = 'cuda'

    # get 125m config
    with open('yamls/mosaic_gpt/125m.yaml') as f:
        cfg = om.load(f)

    # modify cfg for HF GPT2 compatibility
    cfg.max_seq_len = 1024  # GPT2 seq len (GPT3 uses 2048)
    cfg.model.max_seq_len = cfg.max_seq_len
    cfg.train_loader.max_seq_len = cfg.max_seq_len
    cfg.eval_loader.max_seq_len = cfg.max_seq_len
    cfg.model.device = device
    cfg.precision = 'fp16'
    # set dropout prob
    cfg.model.resid_pdrop = 0.1
    cfg.model.emb_pdrop = 0.1
    # attn_dropout is imtegrated into the FlashMHA kernel
    # given this, it will generate different drop idx when compared to nn.Dropout
    # reguradless of if rng is seeded.
    cfg.model.attn_pdrop = 0.0

    # get equivalent config for HF model
    hf_cfg = om.create({'name': 'hf_causal_lm', 'hf_config_name_or_path': 'gpt2'})

    # set seed
    reproducibility.seed_all(cfg.seed)

    # Build Model
    print('Initializing model...')

    print(cfg)
    model = build_composer_model(cfg.model)
    n_params = sum(p.numel() for p in model.parameters())

    hf_model = build_composer_model(hf_cfg).to(device)
    hf_n_params = sum(p.numel() for p in hf_model.parameters())

    assert hf_n_params == n_params

    # Get batch size info
    batch_size = 2
    
    # generate random input branch
    batch = {}
    batch['input_ids']      = torch.randint(low=0, high=cfg.model.vocab_size, size=(batch_size, cfg.max_seq_len)).to(device)
    batch['attention_mask'] = torch.ones(size=(batch_size, cfg.max_seq_len), dtype=torch.int64).to(device)
    batch['labels']         = torch.randint(low=0, high=cfg.model.vocab_size, size=(batch_size, cfg.max_seq_len)).to(device)

    # # set dropout prob to 0 for potentially more numerically precise comparisons
    # model.model.transformer.emb_drop.p = 0.0
    # hf_model.model.transformer.drop.p = 0.0

    # for b in model.model.transformer.blocks:    b.resid_mlp_dropout.p = 0.0
    # for b in hf_model.model.transformer.h:      b.mlp.dropout.p = 0.0

    # for b in model.model.transformer.blocks:    b.resid_attn_dropout.p = 0.0
    # for b in hf_model.model.transformer.h:      b.attn.resid_dropout.p = 0.0

    # # attn_dropout is imtegrated into the FlashMHA kernel and must therefore be set to 0
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
    # HF keys which are ignored
    hf_keys_ignore = [".attn.masked_bias", ".attn.bias"]
    # HF params which need to be transposed
    _transpose = [".attn.c_attn.", ".attn.c_proj.", ".mlp.c_fc.", ".mlp.c_proj."]
    # HF keys which need to be replaced by the associated value
    hf_2_mosaic_key_mods = {
        "model.transformer.h.": "model.transformer.blocks.",
        ".attn.c_attn.":        ".causal_attn.mhsa.Wqkv.",
        ".attn.c_proj.":        ".causal_attn.mhsa.out_proj.",
        ".mlp.c_fc.":           ".mlp.mlp_up.",
        ".mlp.c_proj.":         ".mlp.mlp_down.",
    }

    # convert hf gpt statedict to mosaic gpt statedict using the dict and list above
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

    # load hf model weights into mosaic gpt model
    model.load_state_dict(_hf_model_statedict)

    with torch.autocast(device_type=device, dtype=torch.float16):
        torch.manual_seed(cfg.seed)
        hf_model_fwd = hf_model(batch)
        torch.manual_seed(cfg.seed)
        model_fwd = model(batch)

    print(hf_model_fwd.mean().item(), model_fwd.mean().item())
    print(hf_model_fwd, model_fwd)

    # given dropout seeded the same way, the mean of the outputs is extreamly similar
    assert hf_model_fwd.mean().allclose(model_fwd.mean())
    assert hf_model_fwd.allclose(model_fwd, rtol=1e-02, atol=1e-02)
