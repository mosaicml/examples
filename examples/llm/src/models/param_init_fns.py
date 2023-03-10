# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2021 EleutherAI
# SPDX-License-Identifier: Apache-2.0

import math
import warnings
from functools import partial
from typing import Any, Callable, Optional

import torch
from omegaconf import DictConfig
from torch import nn

InitFunction = Callable[[Any], Any]  # Tensor -> Tensor, but raises errors


def torch_default_param_init_fn_(module: nn.Module, cfg: DictConfig):
    if cfg.get('verbose') and cfg.get('verbose') > 1:
        warnings.warn(
            f"Initializing network using module's reset_parameters attribute")

    if hasattr(module, 'reset_parameters'):
        module.reset_parameters()


def fused_init_helper_(module: nn.Module, init_fn_: InitFunction):
    # parameter initialization is often based on the parameters shape.
    # If a layer is fused, initialization should be based on the shapes
    # of the original tensor instead of the shape of the fused tensor.
    # Layers which are fused should have the _fused attribute defined.
    # The first element of _fused is the dimension along which the tensor is fused.
    # This is followed by an iterable of split indices."

    _fused = getattr(module, '_fused', None)

    if _fused is None:
        raise RuntimeError(f'Internal logic error')

    dim, splits = _fused
    splits = (0, *splits, module.weight.size(dim))
    for s, e in zip(splits[:-1], splits[1:]):
        slice_indices = [slice(None)] * module.weight.ndim
        slice_indices[dim] = slice(s, e)
        init_fn_(module.weight[slice_indices])


def generic_param_init_fn_(module: nn.Module,
                           cfg: DictConfig,
                           init_fn_: InitFunction,
                           mlp_down_init_fn_: Optional[InitFunction] = None):
    if mlp_down_init_fn_ is None:
        mlp_down_init_fn_ = init_fn_
    if cfg.get('verbose') and cfg.get('verbose') > 1:
        warnings.warn(
            f'If model has bias parameters they are initialized to 0.')

    # enable user to divide _is_residual weights by math.sqrt(2 * cfg.n_layers)
    div_is_residual = cfg.get('init_div_is_residual', True)
    if div_is_residual:
        if cfg.get('verbose', 0) > 1:
            warnings.warn(
                f'Initializing _is_residual layers then dividing them by math.sqrt(2 * n_layers).' +\
                f'set `init_div_is_residual: false` in model config to disable this.'
            )

    if isinstance(module, nn.Linear):
        # Linear
        if getattr(module, '_is_residual', False):
            # allow separate init function for the down projection,
            # as is done in GPT-NeoX
            if hasattr(module, '_fused'):
                fused_init_helper_(module, mlp_down_init_fn_)
            else:
                mlp_down_init_fn_(module.weight)

            if div_is_residual:
                with torch.no_grad():
                    module.weight.div_(math.sqrt(2 * cfg.n_layers))
        else:
            if hasattr(module, '_fused'):
                fused_init_helper_(module, init_fn_)
            else:
                init_fn_(module.weight)

        if module.bias is not None:  # type: ignore
            torch.nn.init.zeros_(module.bias)

    elif isinstance(module, nn.Embedding):
        # Embedding
        init_fn_(module.weight)

    elif isinstance(module, nn.LayerNorm):
        # LayerNorm
        if cfg.get('verbose', 0) > 1:
            warnings.warn(
                f'LayerNorm gamma weights are set to 1. If the layer has a bias it is initialized to 0.'
            )
        torch.nn.init.ones_(module.weight)
        if module.bias is not None:  # type: ignore
            torch.nn.init.zeros_(module.bias)

    elif isinstance(module, nn.MultiheadAttention):
        # torch's MultiheadAttention
        if module._qkv_same_embed_dim:
            assert module.in_proj_weight is not None
            assert module.q_proj_weight is None and module.k_proj_weight is None and module.v_proj_weight is None
            # in_proj_weight is actually 3 layers and should be split up for width based init
            _d = cfg.d_model
            splits = (0, _d, 2 * _d, 3 * _d)
            for s, e in zip(splits[:-1], splits[1:]):
                init_fn_(module.in_proj_weight[s:e])
        else:
            assert module.q_proj_weight is not None and module.k_proj_weight is not None and module.v_proj_weight is not None
            assert module.in_proj_weight is None
            init_fn_(module.q_proj_weight)
            init_fn_(module.k_proj_weight)
            init_fn_(module.v_proj_weight)

        # bias
        if module.in_proj_bias is not None:  # type: ignore
            torch.nn.init.zeros_(module.in_proj_bias)
        if module.bias_k is not None:
            torch.nn.init.zeros_(module.bias_k)
        if module.bias_v is not None:
            torch.nn.init.zeros_(module.bias_v)

        # out proj
        if getattr(module.out_proj, '_is_residual', False):
            mlp_down_init_fn_(module.out_proj.weight)
            if div_is_residual:
                with torch.no_grad():
                    module.out_proj.weight.div_(math.sqrt(2 * cfg.n_layers))
        else:
            init_fn_(module.out_proj.weight)

        if module.out_proj.bias is not None:  # type: ignore
            torch.nn.init.zeros_(module.out_proj.bias)

    else:
        for _ in module.parameters(recurse=False):
            # raise error if uninitialized module has any parameters
            raise NotImplementedError(
                f'{module.__class__.__name__} parameters are not initialized by param_init_fn.'
            )


def baseline_param_init_fn_(module: nn.Module, cfg: DictConfig):
    init_fn_ = partial(torch.nn.init.normal_, mean=0.0, std=cfg.init_std)

    if cfg.get('verbose', 0) > 1:
        warnings.warn(
            f'Using torch.nn.init.normal_ init fn mean=0.0, std={cfg.init_std}')

    generic_param_init_fn_(module, cfg, init_fn_)


def kaiming_uniform_param_init_fn_(module: nn.Module, cfg: DictConfig):
    # set nn.init.kaiming_uniform_ defaults
    # note: not many ppl modify the defaults but maybe we should...
    init_gain = cfg.get('init_gain', 0)
    fan_mode = cfg.get('fan_mode', 'fan_in')
    init_nonlinearity = cfg.get('init_nonlinearity', 'leaky_relu')

    if cfg.get('verbose', 0) > 1:
        warnings.warn(
            f'Using nn.init.kaiming_uniform_ init fn with parameters: ' +\
            f'a={init_gain}, mode={fan_mode}, nonlinearity={init_nonlinearity}'
        )

    kaiming_uniform_ = partial(nn.init.kaiming_uniform_,
                               a=init_gain,
                               mode=fan_mode,
                               nonlinearity=init_nonlinearity)

    generic_param_init_fn_(module, cfg, kaiming_uniform_)


def kaiming_normal_param_init_fn_(module: nn.Module, cfg: DictConfig):
    # set torch.nn.init.kaiming_normal_ defaults
    # note: not many ppl modify the defaults but maybe we should...
    init_gain = cfg.get('init_gain', 0)
    fan_mode = cfg.get('fan_mode', 'fan_in')
    init_nonlinearity = cfg.get('init_nonlinearity', 'leaky_relu')

    if cfg.get('verbose', 0) > 1:
        warnings.warn(
            f'Using nn.init.kaiming_normal_ init fn with parameters: ' +\
            f'a={init_gain}, mode={fan_mode}, nonlinearity={init_nonlinearity}'
        )

    kaiming_normal_ = partial(torch.nn.init.kaiming_normal_,
                              a=init_gain,
                              mode=fan_mode,
                              nonlinearity=init_nonlinearity)

    generic_param_init_fn_(module, cfg, kaiming_normal_)


def xavier_uniform_param_init_fn_(module: nn.Module, cfg: DictConfig):
    # set nn.init.xavier_uniform_ defaults
    # note: not many ppl modify the defaults but maybe we should...
    init_gain = cfg.get('init_gain', 1.0)
    xavier_uniform_ = partial(torch.nn.init.xavier_uniform_, gain=init_gain)

    if cfg.get('verbose', 0) > 1:
        warnings.warn(
            f'Using torch.nn.init.xavier_uniform_ init fn with parameters: ' +\
            f'gain={init_gain}'
        )

    generic_param_init_fn_(module, cfg, xavier_uniform_)


def xavier_normal_param_init_fn_(module: nn.Module, cfg: DictConfig):
    # set nn.init.xavier_normal_ defaults
    # note: not many ppl modify the defaults but maybe we should...
    init_gain = cfg.get('init_gain', 1.0)
    xavier_normal_ = partial(torch.nn.init.xavier_normal_, gain=init_gain)

    if cfg.get('verbose', 0) > 1:
        warnings.warn(
            f'Using torch.nn.init.xavier_normal_ init fn with parameters: ' +\
            f'gain={init_gain}'
        )

    generic_param_init_fn_(module, cfg, xavier_normal_)


def small_param_init_fn_(module: nn.Module, cfg: DictConfig):
    """Introduced as `SmallInit` in Tw/oT.

    Transformers without Tears: Improving the Normalization of Self-Attention Nguyen, T.

    & Salazar, J. (2020)
    see https://github.com/EleutherAI/gpt-neox/blob/9610391ab319403cef079b438edd016a2443af54/megatron/model/init_functions.py#L134
    """
    std = math.sqrt(2 / (5 * cfg.d_model))
    init_fn_ = partial(torch.nn.init.normal_, mean=0.0, std=std)

    if cfg.get('verbose', 0) > 1:
        warnings.warn(
            f'Using torch.nn.init.normal_ init fn mean=0.0, std={std}')

    generic_param_init_fn_(module, cfg, init_fn_)


def neox_param_init_fn_(module: nn.Module, cfg: DictConfig):
    """From section 2.3.1 of GPT-NeoX-20B:

    An Open-Source AutoregressiveLanguage Model — Black et. al. (2022)
    see https://github.com/EleutherAI/gpt-neox/blob/9610391ab319403cef079b438edd016a2443af54/megatron/model/init_functions.py#L151
    and https://github.com/EleutherAI/gpt-neox/blob/main/megatron/model/transformer.py
    """
    small_std = math.sqrt(2 / (5 * cfg.d_model))
    small_init_fn_ = partial(torch.nn.init.normal_, mean=0.0, std=small_std)
    wang_std = 2 / cfg.n_layers / math.sqrt(cfg.d_model)
    wang_init_fn_ = partial(torch.nn.init.normal_, mean=0.0, std=wang_std)

    # we normally want this to default to True, but Wang initialization does something similar
    if cfg.get('init_div_is_residual', None) is None:
        cfg['init_div_is_residual'] = False
    elif cfg.get('init_div_is_residual', None) is True:
        warnings.warn(
            'You are using both Wang initialization with ' +
            'init_div_is_residual, both of which have similar functions. ' +
            'If this is an error, set init_div_is_residual to False or ' +
            'use a different param_init_fn in your model config')

    if cfg.get('verbose', 0) > 1:
        warnings.warn(
            f'Using torch.nn.init.normal_ init fn mean=0.0, std={small_std} for '
            + f'most layers and std={wang_std} for MLP down projections')

    generic_param_init_fn_(module,
                           cfg,
                           init_fn_=small_init_fn_,
                           mlp_down_init_fn_=wang_init_fn_)


MODEL_INIT_REGISTRY = {
    'default_': torch_default_param_init_fn_,
    'baseline_': baseline_param_init_fn_,
    'kaiming_uniform_': kaiming_uniform_param_init_fn_,
    'kaiming_normal_': kaiming_normal_param_init_fn_,
    'xavier_uniform_': xavier_uniform_param_init_fn_,
    'xavier_normal_': xavier_normal_param_init_fn_,
    'small_init_': small_param_init_fn_,
    'neox_init_': neox_param_init_fn_,
}
