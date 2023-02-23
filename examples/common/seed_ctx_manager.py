# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

from typing import List, Tuple

import torch
from torch.utils.checkpoint import set_device_states


def get_device_states(*tensors,
                      gpu_devices=[]) -> Tuple[List[int], List[torch.Tensor]]:
    """Extends pytorch torch.utils.checkpoint.get_device_states.

    This variant enables user pass in which devices need rng state cached.
    """
    fwd_gpu_devices = list(
        tensor.get_device()
        for tensor in tensors
        if isinstance(tensor, torch.Tensor) and tensor.is_cuda)
    fwd_gpu_devices = list(set(gpu_devices + fwd_gpu_devices))

    fwd_gpu_states = []
    for device in fwd_gpu_devices:
        with torch.cuda.device(device):
            fwd_gpu_states.append(torch.cuda.get_rng_state())

    return fwd_gpu_devices, fwd_gpu_states


class SeedContextManager:
    """SeedContextManager for managing seeds.

    Upon entering, SeedContextManager caches rng state and re-sets it on exit.

    Useful for initializing layers that need different rngs on different devices
    """

    def __init__(self, *tensors, gpu_devices=[], seed=None):
        self.seed = seed
        self.gpu_devices = gpu_devices
        self.tensors = tensors

        self.cpu_rng_state = None
        self.fwd_gpu_devices = None
        self.gpu_rng_states = None

    def __enter__(self):
        # cache cpu rng state
        self.cpu_rng_state = torch.get_rng_state()
        # cache rng state for gpu devices
        if torch.cuda.is_initialized():
            self.fwd_gpu_devices, self.gpu_rng_states = get_device_states(
                *self.tensors, gpu_devices=self.gpu_devices)

        # set seed
        if self.seed:
            torch.manual_seed(self.seed)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # set previously cached cpu rng state
        torch.set_rng_state(self.cpu_rng_state)
        # set previously cached rng state for gpu devices
        if torch.cuda.is_initialized():
            set_device_states(self.fwd_gpu_devices, self.gpu_rng_states)
