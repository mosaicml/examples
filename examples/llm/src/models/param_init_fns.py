# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

import math
import warnings
from functools import partial

import torch
from torch import nn


def torch_defualt_param_init_fn(module, cfg):
    if hasattr(module, 'reset_parameters'):
        module.reset_parameters()


def generic_param_init_fn(module, cfg, init_fn):
    warnings.warn(f'If model has bias parameters they are initialized to 0.')

    # enable user to divide _is_residual weights by math.sqrt(2 * cfg.n_layers)
    div_is_residual = cfg.get('init_div_is_residual', True)
    if div_is_residual:
        warnings.warn(
            f'Initializing _is_residual layers then dividing them by math.sqrt(2 * n_layers).' +\
            f'set `init_div_is_residual: false` in model config to disable this.'
        )

    # Linear
    if isinstance(module, nn.Linear):
        if hasattr(module, 'fused'):
            # if a layer is fused, its parameter init, if based on shapes, should be of the param sizes when not fused
            dim, splits = module.fused
            if dim not in [0, 1, 2]:
                raise NotImplementedError(
                    f'init for fused layers only implemented on dim 0, 1, and 2.'
                )
            splits = [0, *splits, module.weight.size(dim)]
            for s, e in zip(splits[:-1], splits[1:]):
                if dim == 0:
                    init_fn(module.weight[s:e])
                elif dim == 1:
                    init_fn(module.weight[:, s:e])
                elif dim == 2:
                    init_fn(module.weight[:, :, s:e])
        else:
            init_fn(module.weight)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)

        if div_is_residual and getattr(module, '_is_residual', False):
            with torch.no_grad():
                module.weight.div_(math.sqrt(2 * cfg.n_layers))

    # Embedding
    if isinstance(module, nn.Embedding):
        init_fn(module.weight)

    # LayerNorm
    if isinstance(module, nn.LayerNorm):
        warnings.warn(
            f'LayerNorm gamma weights are set to 1. If the layer has a bias it is initialized to 0.'
        )
        torch.nn.init.ones_(module.weight)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)

    # torch's MultiheadAttention
    if isinstance(module, nn.MultiheadAttention):
        if module._qkv_same_embed_dim:
            assert module.in_proj_weight is not None
            assert module.q_proj_weight is None and module.k_proj_weight is None and module.v_proj_weight is None
            # in_proj_weight is actually 3 layers and should be split up for width based init
            _d = cfg.d_model
            splits = [0, _d, 2 * _d, 3 * _d]
            for s, e in zip(splits[:-1], splits[1:]):
                init_fn(module.in_proj_weight[s:e])
        else:
            assert module.q_proj_weight is not None and module.k_proj_weight is not None and module.v_proj_weight is not None
            assert module.in_proj_weight is None
            init_fn(module.q_proj_weight)
            init_fn(module.k_proj_weight)
            init_fn(module.v_proj_weight)

        # bias
        if module.in_proj_bias is not None:
            torch.nn.init.zeros_(module.in_proj_bias)
        if module.bias_k is not None:
            torch.nn.init.zeros_(module.bias_k)
        if module.bias_v is not None:
            torch.nn.init.zeros_(module.bias_v)

        # out proj
        init_fn(module.out_proj.weight)
        if div_is_residual and getattr(module.out_proj, '_is_residual', False):
            with torch.no_grad():
                module.out_proj.weight.div_(math.sqrt(2 * cfg.n_layers))
        if module.out_proj.bias is not None:
            torch.nn.init.zeros_(module.out_proj.bias)


def baseline_param_init_fn(module, cfg):
    init_fn = partial(torch.nn.init.normal_, mean=0.0, std=cfg.init_std)
    generic_param_init_fn(module, cfg, init_fn)


def kaiming_uniform_param_init_fn(module, cfg):
    # set nn.init.kaiming_uniform_ defaults
    # note: not many ppl modify the defaults but maybe we should...
    init_gain = cfg.get('init_gain', 0)
    fan_mode = cfg.get('fan_mode', 'fan_in')
    init_nonlinearity = cfg.get('init_nonlinearity', 'leaky_relu')

    kaiming_uniform_ = partial(nn.init.kaiming_uniform_,
                               a=init_gain,
                               mode=fan_mode,
                               nonlinearity=init_nonlinearity)

    generic_param_init_fn(module, cfg, kaiming_uniform_)


def kaiming_normal_param_init_fn(module, cfg):
    # set torch.nn.init.kaiming_normal_ defaults
    # note: not many ppl modify the defaults but maybe we should...
    init_gain = cfg.get('init_gain', 0)
    fan_mode = cfg.get('fan_mode', 'fan_in')
    init_nonlinearity = cfg.get('init_nonlinearity', 'leaky_relu')

    kaiming_normal_ = partial(torch.nn.init.kaiming_normal_,
                              a=init_gain,
                              mode=fan_mode,
                              nonlinearity=init_nonlinearity)

    generic_param_init_fn(module, cfg, kaiming_normal_)


def xavier_uniform_param_init_fn(module, cfg):
    # set nn.init.xavier_uniform_ defaults
    # note: not many ppl modify the defaults but maybe we should...
    init_gain = cfg.get('init_gain', 1.0)
    xavier_uniform_ = partial(torch.nn.init.xavier_uniform_, gain=init_gain)

    generic_param_init_fn(module, cfg, xavier_uniform_)


def xavier_normal_param_init_fn(module, cfg):
    # set nn.init.xavier_normal_ defaults
    # note: not many ppl modify the defaults but maybe we should...
    init_gain = cfg.get('init_gain', 1.0)
    xavier_normal_ = partial(torch.nn.init.xavier_normal_, gain=init_gain)

    generic_param_init_fn(module, cfg, xavier_normal_)


MODEL_INIT_REGISTRY = {
    'default': torch_defualt_param_init_fn,
    'baseline': baseline_param_init_fn,
    'kaiming_uniform': kaiming_uniform_param_init_fn,
    'kaiming_normal': kaiming_normal_param_init_fn,
    'xavier_uniform': xavier_uniform_param_init_fn,
    'xavier_normal_': xavier_normal_param_init_fn,
}
