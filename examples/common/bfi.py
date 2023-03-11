# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Brute Force Initialize weights so that layer output has unit norm."""

from typing import Any, Dict

import torch
from composer.core import Callback, State
from composer.loggers import Logger
from composer.utils import dist
from torch import nn

from examples.llm.src.models.param_init_fns import get_div_is_residual_factor


class BruteForceInit(Callback):
    """Brute Force Initialize weights to preserve flow stability.

    Args:
        init_mode (str): string dictating which variance to optimizer for
            Options: ``'forward'``, ``'backward'``, ``'forward_backward_avg'``
            Default: ``'forward'``
        mode (str): string dictating if bfi is applied per feature or layer
            Options: ``'feature'``, ``'layer'``
            Default: ``'layer'``
    """

    def __init__(self, init_mode='forward', mode='feature'):
        self.init_mode = init_mode.lower()
        if mode not in ['feature', 'layer']:
            raise ValueError(f"Mode option not recognized. Options: 'feature' or 'layer'.")
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
            if self.init_mode in ['forward', 'fwd', 'f']:
                self._bfi_forward(state, logger)
            elif self.init_mode in ['backward', 'bwd', 'b']:
                self._bfi_backward(state, logger)
            elif self.init_mode in [
                    'forward_backward_avg', 'fwd_bwd_avg', 'fba'
            ]:
                self._bfi_forward_backward_avg(state, logger)

    def _bfi_forward(self, state: State, logger: Logger):

        init_div_is_residual, div_is_residual = get_div_is_residual_factor(
            state.model.model.cfg)

        hooks = {}  # dict storing all hooks

        def init_pre_fwd_hook(module, input) -> None:
            # removes fwd pre hooks so that they are not called recussivly
            hooks[module].remove()

            with torch.no_grad():
                with state.model.model.summon_full_params(
                        module):  # get params from FSDP
                    if hasattr(module, 'bias') and module.bias is not None:
                        module.bias.fill_(0.)

                    x, = input
                    y = module(x)

                    reduce_dims = list(range(y.dim()))[:-1]
                    sec_mom = (y * y).mean(reduce_dims).sqrt()
                    if self.mode == 'layer':
                        sec_mom = sec_mom.mean()
                    elif self.mode == 'feature':
                        # do nothing, this is the default behavior
                        pass
                    dist.all_reduce(sec_mom)

                    if isinstance(module, nn.Linear):
                        # required because linear layer's weights are transpose before being applied
                        module.weight.div_(sec_mom.unsqueeze(-1))
                    else:  # nn.Embedding
                        module.weight.div_(sec_mom)

                    if init_div_is_residual is not False and getattr(
                            module, '_is_residual', False):
                        with torch.no_grad():
                            module.weight.div_(div_is_residual)

        # if using fsdp this initial first pass (without hooks) instantiates and shards parameters
        with torch.no_grad():
            y = state.model(state.batch)

        # register initialization hooks
        for module in state.model.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                hooks[module] = module.register_forward_pre_hook(
                    init_pre_fwd_hook)

        # forawrd pass and initialize model
        with torch.no_grad():
            y = state.model(state.batch)

        self.initialized = True

    def _bfi_backward(self, state: State, logger: Logger):
        raise NotImplementedError

    def _bfi_forward_backward_avg(self, state: State, logger: Logger):
        raise NotImplementedError
