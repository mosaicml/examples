# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from composer.utils import reproducibility
from omegaconf import OmegaConf as om

from examples.llm.src.models.layers.attention import (
    FlashCausalAttention, TorchCausalAttention, TritonFlashCausalAttention)

RTOL = 1e-2
ATOL = 1e-2


def test_flash_torch():
    reproducibility.seed_all(7)

    cfg = om.create({
        'd_model': 256,
        'n_heads': 2,
        'attn_pdrop': 0,
    })

    n, s, f = 2, 16, cfg.d_model

    fca = FlashCausalAttention(cfg).to('cuda')
    tca = TorchCausalAttention(cfg).to('cuda')

    def gen_tca_mask():
        ms = TorchCausalAttention.mask_shape(cfg.n_heads, s, False)
        attn_mask = torch.empty(*ms).to('cuda')
        TorchCausalAttention.attn_mask_(attn_mask, cfg.n_heads, s)
        return attn_mask

    # clone weights
    tca.mhsa.in_proj_weight.data = fca.mhsa.Wqkv.weight.data.clone().detach()
    tca.mhsa.in_proj_bias.data = fca.mhsa.Wqkv.bias.data.clone().detach()
    tca.mhsa.out_proj.weight.data = fca.mhsa.out_proj.weight.data.clone(
    ).detach()
    tca.mhsa.out_proj.bias.data = fca.mhsa.out_proj.bias.data.clone().detach()

    key_padding_mask = torch.ones(n, s).to('cuda').bool()
    x0 = torch.randn(n, s, f).to('cuda')
    x1 = x0.clone().detach()
    x0.requires_grad = True
    x1.requires_grad = True

    with torch.autocast(x0.device.type):
        y0, _ = fca(x0, key_padding_mask, attn_mask=None)
        y1, _ = tca(x1, key_padding_mask, attn_mask=gen_tca_mask())
        y0 *= key_padding_mask.unsqueeze(-1)
        y1 *= key_padding_mask.unsqueeze(-1)

        loss0 = y0.sum()
        loss1 = y1.sum()

    loss0.backward()
    loss1.backward()

    assert y0.allclose(y1, rtol=RTOL, atol=ATOL)

    assert tca.mhsa.out_proj.bias.grad.allclose(fca.mhsa.out_proj.bias.grad,
                                                rtol=RTOL,
                                                atol=ATOL)
    assert tca.mhsa.out_proj.weight.grad.allclose(fca.mhsa.out_proj.weight.grad,
                                                  rtol=RTOL,
                                                  atol=ATOL)
    assert tca.mhsa.in_proj_bias.grad.allclose(fca.mhsa.Wqkv.bias.grad,
                                               rtol=RTOL,
                                               atol=ATOL)
    assert tca.mhsa.in_proj_weight.grad.allclose(fca.mhsa.Wqkv.weight.grad,
                                                 rtol=RTOL,
                                                 atol=ATOL)

    assert x0.grad.allclose(x1.grad, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize('attn_clip_qkv,attn_qk_ln', [
    (False, False),
    (False, True),
    (True, False),
    (True, True),
])
def test_flash_triton(attn_clip_qkv, attn_qk_ln):
    reproducibility.seed_all(7)

    cfg = om.create({
        'd_model': 256,
        'n_heads': 2,
        'attn_pdrop': 0,
        'attn_clip_qkv': attn_clip_qkv,
        'attn_qk_ln': attn_qk_ln,
    })

    n, s, f = 2, 16, cfg.d_model

    fca = FlashCausalAttention(cfg).to('cuda')
    tfca = TritonFlashCausalAttention(cfg).to('cuda')
    # clone weights
    if cfg.attn_qk_ln or cfg.attn_clip_qkv:
        tfca.Wqkv.weight.data = fca.W_qkv.weight.data.clone().detach()
        tfca.Wqkv.bias.data = fca.W_qkv.bias.data.clone().detach()
        tfca.out_proj.weight.data = fca.out_proj.weight.data.clone().detach()
        tfca.out_proj.bias.data = fca.out_proj.bias.data.clone().detach()
        if cfg.attn_qk_ln:
            tfca.q_ln.weight.data = fca.q_ln.weight.data.clone().detach()
            tfca.q_ln.bias.data = fca.q_ln.bias.data.clone().detach()
            tfca.k_ln.weight.data = fca.k_ln.weight.data.clone().detach()
            tfca.k_ln.bias.data = fca.k_ln.bias.data.clone().detach()
    else:
        tfca.mhsa.Wqkv.weight.data = fca.mhsa.Wqkv.weight.data.clone().detach()
        tfca.mhsa.Wqkv.bias.data = fca.mhsa.Wqkv.bias.data.clone().detach()
        tfca.mhsa.out_proj.weight.data = fca.mhsa.out_proj.weight.data.clone(
        ).detach()
        tfca.mhsa.out_proj.bias.data = fca.mhsa.out_proj.bias.data.clone(
        ).detach()

    key_padding_mask = torch.ones(n, s).to('cuda')
    x0 = torch.randn(n, s, f).to('cuda')
    x1 = x0.clone().detach()
    x0.requires_grad = True
    x1.requires_grad = True

    with torch.autocast(x0.device.type):
        y0, _ = fca(x0, key_padding_mask, attn_mask=None)
        y1, _ = tfca(x1, key_padding_mask, attn_mask=None)
        y0 *= key_padding_mask.unsqueeze(-1)
        y1 *= key_padding_mask.unsqueeze(-1)

        loss0 = y0.sum()
        loss1 = y1.sum()

    loss0.backward()
    loss1.backward()

    assert y0.allclose(y1, rtol=RTOL, atol=ATOL)

    if cfg.attn_qk_ln or cfg.attn_clip_qkv:
        assert tfca.out_proj.bias.grad.allclose(fca.out_proj.bias.grad,
                                                rtol=RTOL,
                                                atol=ATOL)
        assert tfca.out_proj.weight.grad.allclose(fca.out_proj.weight.grad,
                                                  rtol=RTOL,
                                                  atol=ATOL)
        if cfg.attn_qk_ln:
            assert tfca.q_ln.bias.grad.allclose(fca.q_ln.bias.grad,
                                                rtol=RTOL,
                                                atol=ATOL)
            assert tfca.q_ln.weight.grad.allclose(fca.q_ln.weight.grad,
                                                  rtol=RTOL,
                                                  atol=ATOL)
            assert tfca.k_ln.bias.grad.allclose(fca.k_ln.bias.grad,
                                                rtol=RTOL,
                                                atol=ATOL)
            assert tfca.k_ln.weight.grad.allclose(fca.k_ln.weight.grad,
                                                  rtol=RTOL,
                                                  atol=ATOL)
        assert tfca.Wqkv.bias.grad.allclose(fca.W_qkv.bias.grad,
                                            rtol=RTOL,
                                            atol=ATOL)
        assert tfca.Wqkv.weight.grad.allclose(fca.W_qkv.weight.grad,
                                              rtol=RTOL,
                                              atol=ATOL)
    else:
        assert tfca.mhsa.out_proj.bias.grad.allclose(
            fca.mhsa.out_proj.bias.grad, rtol=RTOL, atol=ATOL)
        assert tfca.mhsa.out_proj.weight.grad.allclose(
            fca.mhsa.out_proj.weight.grad, rtol=RTOL, atol=ATOL)
        assert tfca.mhsa.Wqkv.bias.grad.allclose(fca.mhsa.Wqkv.bias.grad,
                                                 rtol=RTOL,
                                                 atol=ATOL)
        assert tfca.mhsa.Wqkv.weight.grad.allclose(fca.mhsa.Wqkv.weight.grad,
                                                   rtol=RTOL,
                                                   atol=ATOL)

    assert x0.grad.allclose(x1.grad, rtol=RTOL, atol=ATOL)
