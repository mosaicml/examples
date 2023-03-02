# Copyright 2022 MosaicML Agent authors
# SPDX-License-Identifier: Apache-2.0

# Adapted from https://github.com/mlcommons/training_results_v2.0/blob/main/HazyResearch/benchmarks/bert/implementations/pytorch/model/ops/fused_dense.py

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import custom_bwd, custom_fwd
import fused_dense_lib as fused_dense_cuda

class FusedDenseResGeluDenseFunc(torch.autograd.Function):
    """
    Residual version of DenseGeluDense returns a tuple (output, input_tensor)
    and modifies the backward pass correspondingly.
    """

    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, x, weight1, bias1, weight2, bias2):
        batch_shape, n = x.shape[:-1], x.shape[-1]
        batch_dim = np.prod(batch_shape)
        output1, gelu_in = fused_dense_cuda.linear_gelu_forward(x.reshape(batch_dim, n), weight1, bias1)
        output2 = fused_dense_cuda.linear_bias_forward(output1, weight2, bias2)
        ctx.save_for_backward(x, weight1, weight2, gelu_in, output1)

        return output2.reshape(*batch_shape, output2.shape[-1]), x

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output, grad_input):
        x, weight1, weight2, gelu_in, output1 = ctx.saved_tensors
        batch_shape, n = x.shape[:-1], x.shape[-1]
        batch_dim = np.prod(batch_shape)
        grad_input, grad_weight1, grad_bias1, grad_weight2, grad_bias2 = fused_dense_cuda.linear_residual_gelu_linear_backward(
            x.reshape(batch_dim, n), gelu_in, output1, weight1, weight2,
            grad_output.reshape(batch_dim, grad_output.shape[-1]), grad_input.reshape(batch_dim, n))
        return grad_input.reshape_as(x), grad_weight1, grad_bias1, grad_weight2, grad_bias2, None


fused_dense_res_gelu_dense_function = FusedDenseResGeluDenseFunc.apply


class DenseResGeluDense(nn.Module):

    def __init__(self, in_features, intermediate_features, out_features, bias=True, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        assert bias == True, 'DenseGeluDense module without bias is currently not supported'
        self.fc1 = nn.Linear(in_features, intermediate_features, bias=bias, **factory_kwargs)
        self.fc2 = nn.Linear(intermediate_features, out_features, bias=bias, **factory_kwargs)

        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        return fused_dense_res_gelu_dense_function(x, self.fc1.weight, self.fc1.bias, self.fc2.weight, self.fc2.bias)
