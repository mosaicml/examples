# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Monitor throughput during training."""
from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict

from composer.core import Callback, State
from composer.loggers import Logger
from composer.utils import dist

from composer.callbacks import SpeedMonitor

GPU_AVAILABLE_FLOPS = 312_000_000_000_000

__all__ = ['SpeedMonitorMFU']


class SpeedMonitorMFU(SpeedMonitor):
    """Logs the training throughput and MFU.
    """

    def batch_end(self, state: State, logger: Logger):
        batch_num_samples = int(state.timestamp.sample) - self.batch_start_num_samples
        batch_wct = state.timestamp.total_wct.total_seconds() - self.batch_start_wct

        # Add the new element
        self.batch_wct_buffer.append(batch_wct)
        self.batch_num_samples_buffer.append(batch_num_samples)

        # Log the throughput
        if len(self.batch_num_samples_buffer) == self.window_size:
            throughput = sum(self.batch_num_samples_buffer) / sum(self.batch_wct_buffer)
            logger.log_metrics({'throughput/samples_per_sec': throughput})
            if hasattr('num_fwd_flops', state.model):
                mfu = 3 * state.model.num_fwd_flops * throughput / (dist.get_world_size() * GPU_AVAILABLE_FLOPS)
                logger.log_metrics({'throughput/mfu': mfu})

        # Log the time
        # `state.timestamp` excludes any time spent in evaluation
        logger.log_metrics({
            'wall_clock/train': state.timestamp.total_wct.total_seconds(),
            'wall_clock/val': self.total_eval_wct,
            'wall_clock/total': (state.timestamp.total_wct.total_seconds() + self.total_eval_wct),
        })
