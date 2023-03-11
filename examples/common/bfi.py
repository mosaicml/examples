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


class BruteForceInit(Callback):
    """Brute Force Initialize weights so that layer output has unit norm."""

    def __init__(self):
        self.initialized = False

    def state_dict(self) -> Dict[str, Any]:
        # add to state dict so BFI is never run again
        return {'initialized': self.initialized}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        # add to state dict so BFI is never run again
        self.initialized = state['initialized']

    def before_forward(self, state: State, logger: Logger):
        if not self.initialized:
            # TODO: handel div_is_residual

            hooks = {}  # dict storing all hooks

            def init_pre_fwd_hook(module, input) -> None:
                # removes fwd pre hooks so that they are not called recussivly
                hooks[module].remove()

                with torch.no_grad():
                    with state.model.model.summon_full_params(module):  # get params from FSDP
                        if hasattr(module, 'bias') and module.bias is not None:
                            module.bias.fill_(0.)

                        x, = input
                        y = module(x)

                        reduce_dims = list(range(y.dim()))[:-1]
                        sec_mom = (y * y).mean(reduce_dims).sqrt()
                        dist.all_reduce(sec_mom)

                        if isinstance(module, nn.Linear):
                            # required because linear layer's weights are transpose before being applied
                            module.weight.div_(sec_mom.unsqueeze(-1))
                        else:  # nn.Embedding
                            module.weight.div_(sec_mom)
                            pass

            # if using fsdp this initial first pass instantiates and shards parameters
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
