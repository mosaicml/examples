# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Monitor throughput during training."""
from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, Union
import warnings
import torch

from composer.core import Callback, State
from composer.loggers import Logger
from composer.utils import dist

from composer.callbacks import SpeedMonitor

GPU_AVAILABLE_FLOPS = {
    # source: https://resources.nvidia.com/en-us-tensor-core/nvidia-tensor-core-gpu-datasheet
    'h100-sxm':     1.979e15 / 2,   # nvidia publishes spec sheet with a 2x sparsity factor
    'h100-pcie':    1.513e15 / 2,   # nvidia publishes spec sheet with a 2x sparsity factor
    # source: https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf
    'a100':         312e12,         # sxm and pcie have same flop counts
    # source: https://images.nvidia.com/content/technologies/volta/pdf/tesla-volta-v100-datasheet-letter-fnl-web.pdf
    'v100-sxm':     125e12,
    'v100-pcie':    112e12,
    # source: https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf
    't4':           65e12,          # sxm and pcie have same flop counts
}

__all__ = ['SpeedMonitorMFU']


def get_gpu_flops_available():
    gpu_flops_available = None

    # torch.cuda.get_device_name() ex output: 'NVIDIA A100-SXM4-40GB' 
    dev_name = torch.cuda.get_device_name().lower()
    if 'h100-sxm' in dev_name:
        gpu_flops_available = GPU_AVAILABLE_FLOPS['h100-sxm']
    elif 'h100-pcie' in dev_name:
        gpu_flops_available = GPU_AVAILABLE_FLOPS['h100-pcie']
    elif 'a100' in dev_name:
        gpu_flops_available = GPU_AVAILABLE_FLOPS['a100']
    elif 'v100-sxm' in dev_name:
        gpu_flops_available = GPU_AVAILABLE_FLOPS['v100-sxm']
    elif 'v100-pcie' in dev_name:
        gpu_flops_available = GPU_AVAILABLE_FLOPS['v100-pcie']
    elif 't4' in dev_name:
        gpu_flops_available = GPU_AVAILABLE_FLOPS['t4']
    
    if gpu_flops_available:
        warnings.warn(
            f'Using {gpu_flops_available=} when calculate MFU (assumes fp16 or bf16 precision using {dev_name}).'
        )
    else:
        warnings.warn(
            f'gpu_flop count not found for {dev_name=}. '
            f'gpu_flops_available can be manually overridden by setting gpu_flops_available in SpeedMonitorMFU()'
        )

    return gpu_flops_available


class SpeedMonitorMFU(SpeedMonitor):
    """Logs the training throughput and MFU.
    """
    def __init__(
        self,
        window_size: int = 100,
        gpu_flops_available: Union[float, int] = None
    ):
        super().__init__(window_size=window_size)
        if gpu_flops_available:
            self.gpu_flops_available = gpu_flops_available
            warnings.warn(
                f'gpu_flops_available is manaually set to {gpu_flops_available} when calculate MFU.'
            )
        else:
            self.gpu_flops_available = get_gpu_flops_available()

    def batch_end(self, state: State, logger: Logger):
        batch_num_samples = int(state.timestamp.sample) - self.batch_start_num_samples
        batch_wct = state.timestamp.total_wct.total_seconds() - self.batch_start_wct

        # Add the new element
        self.batch_wct_buffer.append(batch_wct)
        self.batch_num_samples_buffer.append(batch_num_samples)

        # Log the throughput
        if len(self.batch_num_samples_buffer) == self.window_size:
            world_size = dist.get_world_size()
            grad_accum = state.grad_accum if state.grad_accum else 1
            global_batch_size = state.device_train_microbatch_size * grad_accum * world_size
            throughput = sum(self.batch_num_samples_buffer) / sum(self.batch_wct_buffer)
            dev_throughput = throughput / world_size
            logger.log_metrics({'throughput/samples_per_sec': throughput})
            logger.log_metrics({'throughput/batches_per_sec':  throughput / global_batch_size})
            logger.log_metrics({'throughput/device/samples_per_sec': dev_throughput})
            logger.log_metrics({'throughput/device/batches_per_sec': dev_throughput / global_batch_size})

            if hasattr(state.dataloader.dataset, 'max_seq_len'):
                # only applicable to seq data / models
                logger.log_metrics({'throughput/tokens_per_sec': throughput * state.dataloader.dataset.max_seq_len})
                logger.log_metrics({'throughput/device/tokens_per_sec': dev_throughput * state.dataloader.dataset.max_seq_len})

            if hasattr(state.model, 'num_fwd_flops'):
                flops = (3 * state.model.num_fwd_flops) * throughput
                logger.log_metrics({'throughput/flops': flops})
                dev_flops = flops / world_size
                logger.log_metrics({'throughput/device/flops': dev_flops})
                if self.gpu_flops_available:
                    mfu = dev_flops / self.gpu_flops_available
                    logger.log_metrics({'throughput/device/mfu': mfu})

        # Log the time
        # `state.timestamp` excludes any time spent in evaluation
        logger.log_metrics({
            'wall_clock/train': state.timestamp.total_wct.total_seconds(),
            'wall_clock/val': self.total_eval_wct,
            'wall_clock/total': (state.timestamp.total_wct.total_seconds() + self.total_eval_wct),
        })
