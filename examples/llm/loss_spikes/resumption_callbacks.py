import itertools
from composer.algorithms.layer_freezing.layer_freezing import _remove_param_from_optimizers
import torch
import collections
from composer.core import Callback, State, TimeUnit
from composer.loggers import Logger
from typing import List

__all__ = ['ResumptionCallback', 'RESUMPTION_STRATEGIES', 'GlobalLRLowering']


class ResumptionCallback(Callback):
    def __init__(self,):
          pass
       


class GlobalLRLowering(ResumptionCallback):
    def __init__(self, lr_scale: float):
        self.lr_scale = lr_scale
       
    def fit_start(self, state: State, logger: Logger):
        assert state.optimizers is not None, 'optimizers must be defined'
        for optimizer in state.optimizers:
            for group in optimizer.param_groups:
                group['lr'] *= self.lr_scale
        
        for scheduler in state.schedulers:
            scheduler.base_lrs = [self.lr_scale * lr for lr in scheduler.base_lrs]
            
class LayerwiseLRLowering(ResumptionCallback):
    def __init__(self, layer_names: List[str], lr_scale: float):
        self.layer_names = layer_names
        self.lr_scale = lr_scale
       
    def fit_start(self, state: State, logger: Logger):
        assert state.optimizers is not None, 'optimizers must be defined'

        count = 0
        layer_to_scale = {}
        idx = 0

        for name, p in state.model.named_parameters():
            if p.requires_grad:
                if name in self.layer_names:
                    count += 1
                    layer_to_scale[p] = self.lr_scale
                else:
                    layer_to_scale[p] = 1.0

            idx += 1

        for optimizer in state.optimizers:
            optimizer.layer_to_scale = layer_to_scale

        if count == 0:
            raise Exception(f"Tried to run LayerwiseLRLowering but didn't lower LRs for any layers {count}")


class FastForwardDataloader(ResumptionCallback):

    def __init__(self, num_batches):
        self.num_batches=num_batches

    def epoch_start(self, state, logger):
        breakpoint()
        if state.dataloader_len is None:
            dataloader_iter = iter(state.dataloader)
        else:
            dataloader_iter = itertools.islice(state.dataloader, int(state.dataloader_len))

        for _ in range(self.num_batches):
            batch = next(dataloader_iter)
class ResetOptimizer(ResumptionCallback):
    def __init__(self):
        pass
       
    def fit_start(self, state: State, logger: Logger):
        assert state.optimizers is not None, 'optimizers must be defined'
        for optimizer in state.optimizers:
            optimizer.reset_state()

class LayerFreezing(ResumptionCallback):
    def __init__(self, layer_names: List[str]):
        self.layer_names = layer_names
       
    def fit_start(self, state: State, logger: Logger):
        model_layers = set(name for name, _ in state.model.named_parameters())
        for layer in self.layer_names:
            if layer not in model_layers:
                raise Exception(f"Attempted to freeze layer not found in model: {layer}\nAvailable layers: {model_layers}")

        count = 0
        for name, p in state.model.named_parameters():
            if p.requires_grad and name in self.layer_names:
                p.requires_grad = False
                count += 1
        
        if count == 0:
            raise Exception(f"Tried to run LayerFreezing but didn't freeze any layers {count}")

RESUMPTION_STRATEGIES = {
    "layer_freezing": LayerFreezing,
    'global_lr_lowering': GlobalLRLowering,
    'layerwise_lr_lowering': LayerwiseLRLowering,
    "reset_optimizer": ResetOptimizer,
    "fastforward": FastForwardDataloader
}
