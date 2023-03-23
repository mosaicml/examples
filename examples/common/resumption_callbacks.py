from composer.core import Callback, State
from composer.loggers import Logger
from typing import List

__all__ = ['GlobalLRScaling', 'LayerFreezing',]

class GlobalLRScaling(Callback):
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

class LayerFreezing(Callback):
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

