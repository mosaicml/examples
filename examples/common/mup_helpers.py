"""Monitor gradients during training."""

import warnings
from functools import partial
from typing import Any, List, Optional, Sequence, Union

import torch

from composer.core import Callback, State, Time, TimeUnit

from composer.loggers import Logger
from composer.utils import dist
import os
import pickle

import numpy as np

def get_infshapes_custom(model):
    return {name: param.infshape.width_mult() for name, param in model.named_parameters()}

class GetActivationNorms(Callback):
    
    def __init__(self,
                 interval: Union[int, str, Time] = '1ba',
                 ignore_module_types: Optional[List[str]] = None,
                 only_log_wandb: bool = True,
                 log_input=False):
        self.ignore_module_types = ignore_module_types
        self.only_log_wandb = only_log_wandb

        self.batch_num = 0
        self.handles = []

        # Check that the interval timestring is parsable and convert into time object
        if isinstance(interval, int):
            self.interval = Time(interval, TimeUnit.BATCH)
        elif isinstance(interval, str):
            self.interval = Time.from_timestring(interval)
        if self.interval.unit not in [TimeUnit.BATCH, TimeUnit.EPOCH]:
            raise ValueError(f'Invalid time unit for parameter interval: '
                             f'{self.interval.unit}')

        self.last_train_time_value_logged = -1
        self.module_names = {}

        self.metrics_in_a_batch = []
        self.log_input = log_input

    def batch_end(self, state: State, logger: Logger):

        output_metrics = []
        optimizer_metrics = {}
        for batch_dict in self.metrics_in_a_batch:
            metric_name = ".".join(list(batch_dict.keys())[0].split(".")[:-1])
            optimizer_metrics[metric_name] = float(np.mean(list(batch_dict.values())))

        output_path = f"/root/{state.run_name}" 
        per_batch_output = os.path.join(output_path, f"batch_{self.batch_num}.pickle")


        
        os.makedirs(output_path, exist_ok=True)


        print(f"writing to {per_batch_output}")

        print(optimizer_metrics)


        with open(per_batch_output, 'wb') as fp:
            pickle.dump(optimizer_metrics, fp)

        if state.fsdp_enabled and dist.get_world_size() > 0 and self.log_optimizer_metrics:
            
            raise Exception("FSDP not supported yet!")

        self.batch_num += 1
        logger.log_metrics(optimizer_metrics)
        self.metrics_in_a_batch = []

    def before_forward(self, state: State, logger: Logger):
            current_time_value = state.timestamp.get(self.interval.unit).value

            if current_time_value % self.interval.value == 0 and current_time_value != self.last_train_time_value_logged:
                if not self.module_names:
                    self.create_module_names(state.model)

                self.attach_forward_hooks(state, logger)

    def after_forward(self, state: State, logger: Logger):
        current_time_value = state.timestamp.get(self.interval.unit).value

        if current_time_value % self.interval.value == 0 and current_time_value != self.last_train_time_value_logged:
            self.last_train_time_value_logged = current_time_value
            self.remove_forward_hooks()

    def attach_forward_hooks(self, state: State, logger: Logger):
        step = state.timestamp.batch.value
        self.register_forward_hook(state.model, logger, state.fsdp_enabled, state.run_name, step)

    def remove_forward_hooks(self):
        for handle in self.handles:
            handle.remove()
        # Resetting handles we track
        self.handles = []

    def register_forward_hook(self, model: torch.nn.Module, logger: Logger, fsdp_enabled: bool, run_name: str, step: Optional[int]):
        model.apply(partial(self._register_forward_hook, logger, fsdp_enabled, run_name, step))

    def _register_forward_hook(self, logger: Logger,  fsdp_enabled: bool, run_name: str, step: Optional[int], module: torch.nn.Module):
        self.handles.append(module.register_forward_hook(partial(self.forward_hook, logger,fsdp_enabled, run_name, step)))

    def forward_hook(self, logger: Logger, fsdp_enabled: bool, run_name: str, step: Optional[int], module: torch.nn.Module, input: Sequence,
                    output: Sequence):
        module_name = self.module_names[module]

        if self.ignore_module_types is not None:
            for ignore_module_type in self.ignore_module_types:
                if ignore_module_type in module_name:
                    return

        metrics = {}
        if input is not None and self.log_input:
            for i, val in enumerate(input):
                if val is None or isinstance(val, dict):
                    continue
                if isinstance(val, str) and isinstance(input, dict):
                    self.recursively_add_metrics(metrics, module_name, f'_input.{i}', output[val])  # type: ignore
                else:
                    self.recursively_add_metrics(metrics, module_name, f'_input.{i}', val)

        if output is not None:
            for i, val in enumerate(output):
                if val is None or isinstance(val, dict):
                    continue
                if isinstance(val, str) and isinstance(output, dict):
                    self.recursively_add_metrics(metrics, module_name, f'_output.{i}', output[val])  # type: ignore
                else:
                    self.recursively_add_metrics(metrics, module_name, f'_output.{i}', val)


        self.metrics_in_a_batch.append(metrics)

    def recursively_add_metrics(self, metrics: dict, name: str, suffix: str, values: Any):
        # Becuase of the recursive diving, we need this call to prevent infinite recursion.
        if isinstance(values, str):
            return
        # Keep recursively diving if the value is a sequence
        if isinstance(values, Sequence):
            for i, value in enumerate(values):
                self.recursively_add_metrics(metrics, f'{name}_{i}', suffix, value)
            return
        else:
            self.add_metrics(metrics, name, suffix, values)

    def add_metrics(self, metrics: dict, name: str, suffix: str, value: torch.Tensor):
        # We shouldn't log booleans
        if value.dtype == torch.bool:
            return
        if value.is_floating_point() or value.is_complex():
            metrics[f'activations/l1_norm/{name}{suffix}'] = torch.abs(value).mean().item()

    def create_module_names(self, model: torch.nn.Module):
        self.module_names = {m: name for name, m in model.named_modules()}
