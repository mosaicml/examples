# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Brute Force Initialize weights so that layer output has unit norm."""

import math
import warnings
from typing import Any, Dict

import torch
from composer.core import Callback, State
from composer.loggers import Logger
from composer.utils import dist
from torch import nn, Tensor


class BruteForceInit(Callback):
    """Brute Force Initialize weights to preserve flow stability.

    Args:
        init_mode (str): string dictating which variance to optimizer for
            Options: ``'forward'``, ``'backward'``, ``'forward_backward_avg'``
            Default: ``'forward'``
        init_norm (int, str): desired norm of layer outputs
        mode (str): string dictating if bfi is applied per feature or layer
            Options: ``'feature'``, ``'layer'``
            Default: ``'layer'``
    """

    def __init__(self, init_mode='forward', init_norm=1., mode='feature'):
        self.init_mode = init_mode.lower()
        self.init_norm = init_norm
        if mode not in ['feature', 'layer']:
            raise ValueError(
                f"Mode option not recognized. Options: 'feature' or 'layer'.")
        self.mode = mode.lower()

        self.initialized = False

    def state_dict(self) -> Dict[str, Any]:
        # add to state dict so BFI is never run again
        return {'initialized': self.initialized}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        # add to state dict so BFI is never run again
        self.initialized = state['initialized']

    def before_forward(self, state: State, logger: Logger):
        if not self.initialized:
            if state.model.model.cfg.get('verbose', 0) > 1:
                warnings.warn(
                    f'Brute force initializing model weights with init_mode={self.init_mode}, init_norm={self.init_norm} and mode={self.mode}.'
                )

            if self.init_mode in ['forward', 'fwd', 'f']:
                self._bfi_forward(state, logger)
            elif self.init_mode in ['backward', 'bwd', 'b']:
                self._bfi_backward(state, logger)
            elif self.init_mode in [
                    'forward_backward_avg', 'fwd_bwd_avg', 'fba'
            ]:
                self._bfi_forward_backward_avg(state, logger)
            else:
                raise ValueError(
                    f"BFI init_mode not recognized. Options: 'forward', 'backward', 'forward_backward_avg'"
                )

    def _bfi_forward(self, state: State, logger: Logger):
        from examples.llm.src.models.param_init_fns import \
            get_div_is_residual_factor

        init_div_is_residual, div_is_residual = get_div_is_residual_factor(
            state.model.model.cfg)

        hooks = {}  # dict storing all hooks
        init_done = set()
        self.init_layer = None


        def init_pre_fwd_hook(module, input) -> None:
            # removes fwd pre hooks so that they are not called recussivly
            hook = hooks.pop(module)
            hook.remove()

            with torch.no_grad():
                if hasattr(module, 'bias') and module.bias is not None:
                    module.bias.fill_(0.)

                x, = input
                y = module(x)

                reduce_dims = list(range(y.dim()))[:-1]
                sec_mom = (y * y).mean(reduce_dims).sqrt()

                if isinstance(self.init_norm, str):
                    in_scale = (x * x).mean(reduce_dims).sqrt().mean()
                    if self.init_norm == 'same':
                        scale_factor = sec_mom.mean() / in_scale
                    elif self.init_norm.beginswith('mul'):
                        mul = float(self.init_norm.strip('mul'))
                        scale_factor = sec_mom.mean() / (mul * in_scale)
                    else:
                        raise NotImplementedError
                else:
                    scale_factor = sec_mom / math.sqrt(self.init_norm)

                    if self.mode == 'layer':
                        scale_factor = scale_factor.mean()
                    elif self.mode == 'feature':
                        # do nothing, this is the default behavior
                        pass

                if init_div_is_residual is not False and getattr(
                        module, '_is_residual', False):
                    scale_factor *= div_is_residual

                dist.all_reduce(scale_factor)

                if self.init_layer is None:
                    self.init_layer = (module, module.weight, scale_factor)


        # this initial first pass (without hooks) verifies everything works
        device_type = state.batch['input_ids'].device.type
        with torch.autocast(device_type=device_type):
            loss = state.model(state.batch).mean()  # proxy loss
        loss.backward()
        state.model.zero_grad(set_to_none=True)

        while True:
            # this looping is only required because FSDP does not enable summon_full_params within a fwd or bwd pass

            # register initialization hooks
            for module in state.model.modules():
                if isinstance(module, (nn.Linear,)):  # nn.Embedding
                    if module not in init_done:
                        hooks[module] = module.register_forward_pre_hook(
                            init_pre_fwd_hook)

            if not hooks:
                break

            # forawrd pass and initialize model
            with torch.no_grad():
                with torch.autocast(device_type=device_type):
                    y = state.model(state.batch)

            with state.model.model.summon_full_params(
                    state.model.model,
                    writeback=True,
            ):  # get params from FSDP and enable writeback
                with torch.no_grad():
                    module, param, scale_factor = self.init_layer
                    if isinstance(module, nn.Linear):
                        # required because linear layer's weights are transpose before being applied
                        module.weight.div_(scale_factor.unsqueeze(-1))
                    else:
                        raise NotImplementedError(
                            f'BFI not implemented for {module}.')

            init_done.add(module)
            self.init_layer = None

        self.initialized = True

    def _bfi_backward(self, state: State, logger: Logger):
        from examples.llm.src.models.param_init_fns import \
            get_div_is_residual_factor

        init_div_is_residual, div_is_residual = get_div_is_residual_factor(
            state.model.model.cfg)

        hooks = {}  # dict storing all hooks
        init_done = set()
        self.init_layer = None


        def init_bwd_hook(module, grad_input, grad_output):
            # removes fwd pre hooks so that they are not called recussivly
            hook = hooks.pop(module)
            hook.remove()

            with torch.no_grad():

                dy, = grad_output
                dx = dy @ module.weight.to(dy.dtype)

                reduce_dims = list(range(y.dim()))[:-1]
                sec_mom = (dx * dx).mean(reduce_dims).sqrt()

                if isinstance(self.init_norm, str):
                    in_scale = (dy * dy).mean(reduce_dims).sqrt().mean()
                    if self.init_norm == 'same':
                        scale_factor = sec_mom.mean() / in_scale
                    elif self.init_norm.beginswith('mul'):
                        mul = float(self.init_norm.strip('mul'))
                        scale_factor = sec_mom.mean() / (mul * in_scale)
                    else:
                        raise NotImplementedError
                else:
                    scale_factor = sec_mom / math.sqrt(self.init_norm)

                    if self.mode == 'layer':
                        scale_factor = scale_factor.mean()
                    elif self.mode == 'feature':
                        # do nothing, this is the default behavior
                        pass

                if init_div_is_residual is not False and getattr(
                        module, '_is_residual', False):
                    scale_factor *= div_is_residual

                dist.all_reduce(scale_factor)

                if self.init_layer is None:
                    self.init_layer = (module, module.weight, scale_factor)


        # this initial first pass (without hooks) verifies everything works
        device_type = state.batch['input_ids'].device.type
        with torch.autocast(device_type=device_type):
            loss = state.model(state.batch).mean()  # proxy loss
        loss.backward()
        state.model.zero_grad(set_to_none=True)

        # zero all bias if model has bias
        with state.model.model.summon_full_params(
                    state.model.model,
                    writeback=True,
            ):  # get params from FSDP and enable writeback
            with torch.no_grad():
                for module in state.model.modules():
                    if hasattr(module, 'bias'):
                        module.bias.fill_(0)

        while True:
            # this looping is only required because FSDP does not enable summon_full_params within a fwd or bwd pass

            # register initialization hooks
            for module in state.model.modules():
                if isinstance(module, (nn.Linear)):
                    if module not in init_done:
                        hooks[module] = module.register_full_backward_hook(
                            init_bwd_hook)

            if not hooks:
                break

            with torch.autocast(device_type=device_type):
                y = state.model(state.batch)
            loss = state.model.loss(y, state.batch)
            loss.backward()
            # state.model.zero_grad(set_to_none=True)

            with state.model.model.summon_full_params(
                    state.model.model,
                    writeback=True,
            ):  # get params from FSDP and enable writeback
                with torch.no_grad():
                    module, param, scale_factor = self.init_layer
                    if isinstance(module, nn.Linear):
                        # required because linear layer's weights are transpose before being applied
                        try:
                            module.weight.div_(scale_factor)
                        except:
                            module.weight.div_(scale_factor.unsqueeze(-1))
                    else:
                        raise NotImplementedError(
                            f'BFI not implemented for {module}.')

            init_done.add(module)
            self.init_layer = None

        self.initialized = True

    def _bfi_forward_backward_avg(self, state: State, logger: Logger):
        raise NotImplementedError
