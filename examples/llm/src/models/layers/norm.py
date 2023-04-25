# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0
import torch
from composer.algorithms.low_precision_layernorm.low_precision_layernorm import \
    LPLayerNorm
from torch import nn


def rms_norm(x, weight=None, eps=1e-5):
    output = x / torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    if weight is not None:
        return output * weight
    return output


class RMSNorm(torch.nn.Module):

    def __init__(
        self,
        normalized_shape,
        eps=1e-5,
        weight=True,
        dtype=None,
        device=None,
    ):
        super().__init__()
        self.eps = eps
        if weight:
            self.weight = nn.Parameter(
                torch.ones(normalized_shape, dtype=dtype, device=device))
        else:
            self.register_parameter('weight', None)

    def forward(self, x):
        return rms_norm(x.float(), self.weight, self.eps).to(dtype=x.dtype)


def _cast_if_autocast_enabled(tensor):
    if torch.is_autocast_enabled():
        if tensor.device.type == 'cuda':
            dtype = torch.get_autocast_gpu_dtype()
        elif tensor.device.type == 'cpu':
            dtype = torch.get_autocast_cpu_dtype()
        else:
            raise NotImplementedError()
        return tensor.to(dtype=dtype)
    return tensor


class LPRMSNorm(RMSNorm):

    def __init__(
        self,
        normalized_shape,
        eps=1e-5,
        weight=True,
        dtype=None,
        device=None,
    ):
        super().__init__(
            normalized_shape=normalized_shape,
            eps=eps,
            weight=weight,
            dtype=dtype,
            device=device,
        )

    def forward(self, x):
        downcast_x = _cast_if_autocast_enabled(x)
        downcast_weight = _cast_if_autocast_enabled(
            self.weight) if self.weight is not None else self.weight
        with torch.autocast(enabled=False, device_type=x.device.type):
            return rms_norm(downcast_x, downcast_weight,
                            self.eps).to(dtype=x.dtype)


NORM_CLASS_REGISTRY = {
    'layernorm': nn.LayerNorm,
    'ln': nn.LayerNorm,  # alias
    'low_precision_layernorm': LPLayerNorm,
    'lplayernorm': LPLayerNorm,  # alias
    'lpln': LPLayerNorm,  # alias
    'rmsorm': RMSNorm,
    'low_precision_rmsorm': LPRMSNorm,
    'lprmsorm': LPRMSNorm,  # alias
}
