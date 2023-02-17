from composer.composer.algorithms.layer_freezing.layer_freezing import _remove_param_from_optimizers
import torch
import collections
from composer.core import Callback, State
from composer.loggers import Logger
from typing import List

__all__ = ['ResumptionCallback', 'RESUMPTION_STRATEGIES']


class ResumptionCallback(Callback):
    def __init__(self,):
          pass
       
    def fit_start(self, state: State, logger: Logger):
        raise NotImplementedError


class LayerFreezing(ResumptionCallback):
    def __init__(self, layer_names: List[str]):
        self.layer_names = layer_names
       
    def fit_start(self, state: State, logger: Logger):
        for name, p in state.model.named_parameters():
            if p.requires_grad and name in self.layer_names:
                p.requires_grad = False
                _remove_param_from_optimizers(p, state.optimizers)

RESUMPTION_STRATEGIES = {
    "layer_freezing": LayerFreezing
}
