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
            world_size = dist.get_world_size()
            throughput = sum(self.batch_num_samples_buffer) / sum(self.batch_wct_buffer)
            logger.log_metrics({'throughput/samples_per_sec': throughput})
            logger.log_metrics({'throughput/batches_per_sec':  throughput / state.global_train_batch_size})

            dev_samples_per_sec = throughput / world_size
            logger.log_metrics({'throughput/device/samples_per_sec': dev_samples_per_sec})
            logger.log_metrics({'throughput/device/batches_per_sec': dev_samples_per_sec / state.global_train_batch_size})

            if hasattr(self.model.cfg, 'max_seq_len'):
                # only applicable to seq data
                logger.log_metrics({'throughput/tokens_per_sec': throughput * self.model.cfg.max_seq_len})
                logger.log_metrics({'throughput/device/tokens_per_sec': throughput * self.model.cfg.max_seq_len / world_size})


            if hasattr(state.model, 'num_fwd_flops'):
                flops = (3 * state.model.num_fwd_flops) * throughput
                logger.log_metrics({'throughput/flops': flops})
                
                device_flops = flops / world_size
                logger.log_metrics({'throughput/device/flops': flops / world_size})

                mfu = device_flops / GPU_AVAILABLE_FLOPS
                logger.log_metrics({'throughput/device/mfu': mfu})

        # Log the time
        # `state.timestamp` excludes any time spent in evaluation
        logger.log_metrics({
            'wall_clock/train': state.timestamp.total_wct.total_seconds(),
            'wall_clock/val': self.total_eval_wct,
            'wall_clock/total': (state.timestamp.total_wct.total_seconds() + self.total_eval_wct),
        })
