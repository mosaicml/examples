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

    The training throughput in terms of number of samples per second is logged on the
    :attr:`.Event.BATCH_END` event if we have reached the ``window_size`` threshold.

    The wall clock train time is logged on every :attr:`.Event.BATCH_END` event.

    The average throughout over an epoch is logged on the :attr:`.Event.EPOCH_END` event.

    Example:
        .. doctest::

            >>> from composer import Trainer
            >>> from composer.callbacks import SpeedMonitor
            >>> # constructing trainer object with this callback
            >>> trainer = Trainer(
            ...     model=model,
            ...     train_dataloader=train_dataloader,
            ...     eval_dataloader=eval_dataloader,
            ...     optimizers=optimizer,
            ...     max_duration='1ep',
            ...     callbacks=[SpeedMonitor(window_size=100, model_flops=<MODEL_FWD_FLOPS>)],
            ... )

    The training throughput is logged by the :class:`.Logger` to the following keys as
    described below.

    +----------------------------------+-------------------------------------------------------------+
    | Key                              | Logged data                                                 |
    +==================================+=============================================================+
    |                                  | Rolling average (over ``window_size`` most recent           |
    | ``throughput/samples_per_sec``   | batches) of the number of samples processed per second      |
    |                                  |                                                             |
    +----------------------------------+-------------------------------------------------------------+
    |                                  | Rolling average (over ``window_size`` most recent           |
    | ``throughput/mfu``               | batches) of mfu                                             |
    |                                  |                                                             |
    +----------------------------------+-------------------------------------------------------------+
    | ``wall_clock/train``             | Total elapsed training time                                 |
    +----------------------------------+-------------------------------------------------------------+
    | ``wall_clock/val``               | Total elapsed validation time                               |
    +----------------------------------+-------------------------------------------------------------+
    | ``wall_clock/total``             | Total elapsed time (wall_clock/train + wall_clock/val)      |
    +----------------------------------+-------------------------------------------------------------+

    Args:
        window_size (int, optional): Number of batches to use for a rolling average of throughput.
            Defaults to 100.
        model_flops (int, optional): Number of FLOPs model uses in the fwd pass per sample. If used, callback will track training MFU.
            Defaults to None (disabled)
    """

    def __init__(self, window_size: int = 100, model_flops: int = None):
        super().__init__(window_size=window_size)
        self.model_flops = model_flops

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
            if self.model_flops is not None:
                mfu = 3 * self.model_flops * throughput / (dist.get_world_size() * GPU_AVAILABLE_FLOPS)
                logger.log_metrics({'throughput/mfu': mfu})

        # Log the time
        # `state.timestamp` excludes any time spent in evaluation
        logger.log_metrics({
            'wall_clock/train': state.timestamp.total_wct.total_seconds(),
            'wall_clock/val': self.total_eval_wct,
            'wall_clock/total': (state.timestamp.total_wct.total_seconds() + self.total_eval_wct),
        })
