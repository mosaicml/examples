"""Monitor gradients during training."""

import torch

from composer.core import Callback, State
from composer.loggers import Logger
from composer.utils import dist
import os
import json

from collections import OrderedDict

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

    return unscaled_param_group, ffn_down_group, rest_of_network_group

        
def mup_setup_lrs_for_params(named_parameters, mup_config, base_lr):
    filtered_groups = _filter_named_parameters(named_parameters)

    param_groups = [] 

    for group, scale_factor in zip(filtered_groups, [1.0, mup_config.ffn_scale_ratio, mup_config.d_model_scale_ratio ]):
        param_groups.append({"params" : list(group.values()), "lr": base_lr / float(scale_factor)})

    print(param_groups)
    
    return param_groups

    # Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0


class GetParamNorms(Callback):
    
    def __init__(self):
        self.batch_num = 0
    def batch_end(self, state: State, logger: Logger):

        optimizer_metrics = OrderedDict()
        output_path = f"/root/{state.run_name}" 
        per_batch_output = os.path.join(output_path, f"batch_{self.batch_num}.json")

        for name, p in state.model.named_parameters():
            if p.grad is not None and p.requires_grad:
                param_norm = torch.abs(p.data).mean()
                optimizer_metrics[f'l1_norm/param/{name}'] = param_norm
        
        os.makedirs(output_path, exist_ok=True)

        np_friendly_version_metrics = {}
        for param_name, param_norm in optimizer_metrics.items():
            np_friendly_version_metrics[param_name] = param_norm.item()

        print(f"writing to {per_batch_output}")

        with open(per_batch_output, 'w') as fp:
            json.dump(np_friendly_version_metrics, fp)

        if state.fsdp_enabled and dist.get_world_size() > 0 and self.log_optimizer_metrics:
            
            raise Exception("FSDP not supported yet!")
            # # If FSDP is enabled, the optimizer state lives on different ranks and must be reduced
            # # and combined before we can compute metrics.
            # # Each metric has a different way of being reduced, so the optimizer is responsible for implementing
            # # the reduction process.
            # # It occurs first via a pre-reduce, where the metric on each rank is modified and prepared
            # # then an all-reduce where the modified metric on each rank is combined into the correct metric across all ranks.
            # #
            # # For example, L2 norms are squared on each rank before we apply all_reduce(SUM) and take the sqrt on each rank
            # pre_reduce_metrics = getattr(state.optimizers[0], 'pre_reduce_metrics', None)
            # if callable(pre_reduce_metrics) and self.log_optimizer_metrics:
            #     optimizer_metrics = pre_reduce_metrics(optimizer_metrics)

            # dist_reduce_metrics = getattr(state.optimizers[0], 'dist_reduce_metrics', None)
            # if callable(dist_reduce_metrics) and self.log_optimizer_metrics:
            #     optimizer_metrics = dist_reduce_metrics(optimizer_metrics)

        self.batch_num += 1
        logger.log_metrics(optimizer_metrics)
