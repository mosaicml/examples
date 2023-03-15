import itertools
from composer.algorithms.layer_freezing.layer_freezing import _remove_param_from_optimizers
import torch
import collections
from composer.core import Callback, State, TimeUnit
from composer.loggers import Logger
from composer.optim.scheduler import _convert_time
from typing import List

__all__ = ['ResumptionCallback', 'RESUMPTION_STRATEGIES',]


class ResumptionCallback(Callback):
    def __init__(self,):
          pass
       
class GlobalLRScaling(ResumptionCallback):
    def __init__(self, lr_scale: float, wd_pct: float = 0.0):
        self.lr_scale = lr_scale
        self.wd_pct = wd_pct
       
    def fit_start(self, state: State, logger: Logger):
        assert state.optimizers is not None, 'optimizers must be defined'
        for optimizer in state.optimizers:
            for group in optimizer.param_groups:
                group['lr'] *= self.lr_scale
                group['weight_decay'] = group['lr'] * self.wd_pct
                if 'initial_lr' in group:
                    group['initial_lr'] *= self.lr_scale
                print(f"Set LR and WD to {group['lr']}, {group['weight_decay']}")
        
        for scheduler in state.schedulers:
            scheduler.base_lrs = [self.lr_scale * lr for lr in scheduler.base_lrs]
            
class LayerwiseLRScaling(ResumptionCallback):
    def __init__(self, layer_names: List[str], lr_scale: float):
        self.layer_names = layer_names

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
                print(f"Froze layer: {name}")
                p.requires_grad = False
                print(f"Param: {p}")
                count += 1
        
        if count == 0:
            raise Exception(f"Tried to run LayerFreezing but didn't freeze any layers {count}")


class ProgressiveLayerFreezing(ResumptionCallback):
    def __init__(self, batch_interval=150):
        self.batch_interval = batch_interval
        self.freeze_times = {}
    
    def fit_start(self, state: State, logger: Logger):
        total_train_dur = _convert_time('1dur', state)
        print(f"Total train dur: {total_train_dur}")
        params = [name for name, _ in state.model.named_parameters()]


        self.freeze_times = {int(total_train_dur-(len(params)-idx)*self.batch_interval) : p  for idx, p in enumerate(params)}
        
        print(f"Following freezing schedule: {self.freeze_times}")
       
       
    def batch_end(self, state: State, logger: Logger):
        curr_time = int(state.timestamp.get(TimeUnit.BATCH).value)
        if curr_time in self.freeze_times:
            layer_name = self.freeze_times[curr_time]
            for name, p in state.model.named_parameters():
                if p.requires_grad and name == layer_name:
                    print(f"Froze layer: {name} at time {curr_time}")
                    p.requires_grad = False
                    print(f"Param: {p}")



class LayerwiseMasking(ResumptionCallback):
    def __init__(self, layer_names: List[str], timeout: int):
        self.freeze_all =  any(x == 'all' for x in layer_names)
        self.layer_names = layer_names
        self.timeout = timeout
        self.counter = 0
    
    def batch_end(self, state: State, logger: Logger):        
        self.counter += 1
        if self.counter == self.timeout:
            for name, p in state.model.named_parameters():
                if p.requires_grad and (name in self.layer_names or self.freeze_all):
                    for opt in state.optimizers:
                        if hasattr(opt, 'set_masking'):
                            # turn off masking
                            opt.set_masking(p, False)

    def fit_start(self, state: State, logger: Logger):        
        count = 0
        for name, p in state.model.named_parameters():
            if p.requires_grad and (name in self.layer_names or self.freeze_all):
                for opt in state.optimizers:
                    if hasattr(opt, 'set_masking'):
                        print(f"Masked layer: {name}")
                        opt.set_masking(p, True)
                        count += 1
        
        if count == 0:
            raise Exception(f"Tried to run LayerFreezing but didn't freeze any layers {count}")

RESUMPTION_STRATEGIES = {
    "layer_freezing": LayerFreezing,
    'global_lr_scaling': GlobalLRScaling,
    'layerwise_lr_lowering': LayerwiseLRScaling,
    "reset_optimizer": ResetOptimizer,
    "layerwise_masking": LayerwiseMasking,
    "progressive_layer_freezing": ProgressiveLayerFreezing
}
