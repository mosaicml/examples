"""Monitor gradients during training."""

import warnings
from functools import partial
from typing import Any, List, Optional, Sequence, Union

import torch

from composer.core import Callback, State, Time, TimeUnit

from composer.core import Callback, State
from composer.loggers import Logger
from composer.utils import dist
import os
# import json
import pickle

from collections import OrderedDict
import numpy as np

# TODO (sasha): Fix hacky name based filtering, maybe filter on module type?
def _is_input_type_layer(name):
    return "wte" in name or "wpe" in name
def _is_layernorm_type(name):
    return "ln_1" in name or "ln_2" in name or "ln_f" in name
def _is_ffn_down(name):
    return "mlp_down" in name

def _filter_named_parameters(named_parameters):
    """
    returns groups that will be used for param_groups
    
    unscaled param group: LR scale: 1.0
    ffn_down: LR scale: / d_ffn_ratio
    rest_of_network: LR scale: / d_model_ratio

    """
    unscaled_param_group = {}
    ffn_down_group = {}
    rest_of_network_group = {}

    for name, param in named_parameters:
        # print(f"{name} {param.shape}")

        if _is_input_type_layer(name) or _is_layernorm_type(name):
            print(f"unscaled param type: {name}")
            unscaled_param_group[name] = param
        elif _is_ffn_down(name):
            print(f"ffn down:{name}")
            ffn_down_group[name] = param
        else:
            rest_of_network_group[name] = param

    print(f"{list(unscaled_param_group.keys())=}")
    print(f"{list(ffn_down_group.keys())=}")
    print(f"{list(rest_of_network_group.keys())=}")
    return unscaled_param_group, ffn_down_group, rest_of_network_group

        
def mup_setup_lrs_for_params(named_parameters, mup_config, base_lr):
    filtered_groups = _filter_named_parameters(named_parameters)

    param_groups = [] 

    for group, scale_factor in zip(filtered_groups, [1.0, mup_config.ffn_scale_ratio, mup_config.d_model_scale_ratio ]):
        print(list(group.keys()))
        print(base_lr / scale_factor)
        param_groups.append({"params" : list(group.values()), "lr": base_lr / float(scale_factor)})

    print(param_groups)
    
    return param_groups

    # Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0


class GetParamNorms(Callback):
    
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

        # print(self.metrics_in_a_batch)


        output_metrics = []
        optimizer_metrics = {}
        for batch_dict in self.metrics_in_a_batch:
            metric_name = ".".join(list(batch_dict.keys())[0].split(".")[:-1])
            optimizer_metrics[metric_name] = float(np.mean(list(batch_dict.values())))
            # print(np.std(list(batch_dict.values())))

        output_path = f"/root/{state.run_name}" 
        per_batch_output = os.path.join(output_path, f"batch_{self.batch_num}.pickle")

        # for name, p in state.model.named_parameters():
        #     if p.grad is not None and p.requires_grad:
        #         param_norm = torch.abs(p.data).mean()
        #         optimizer_metrics[f'l1_norm/param/{name}'] = param_norm
        
        os.makedirs(output_path, exist_ok=True)

        # np_friendly_version_metrics = {}
        # for param_name, param_norm in optimizer_metrics.items():
        #     np_friendly_version_metrics[param_name] = param_norm.item()

        print(f"writing to {per_batch_output}")

        print(optimizer_metrics)


        with open(per_batch_output, 'wb') as fp:
            # json.dump(optimizer_metrics, fp)
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
        # logger.log_metrics(metrics)

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
