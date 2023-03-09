# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Monitor rate of change of loss."""
from __future__ import annotations

from typing import Any, Dict

from composer.core import Callback, State
from composer.loggers import Logger


class FDiffLoss(Callback):
    """Rate of chage of loss.

    tracks and plots the rate of change of loss effectively taking the numerical
    derivative of the loss
    """

    def __init__(self):
        self.train_prev_loss = None
        self.train_prev_metric = {}
        self.eval_prev_metric = {}

    def state_dict(self) -> Dict[str, Any]:
        return {
            'train_prev_loss': self.train_prev_loss,
            'train_prev_metric': self.train_prev_metric,
            'eval_prev_metric': self.eval_prev_metric,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.train_prev_loss = state['train_prev_loss']
        self.train_prev_metric = state['train_prev_metric']
        self.eval_prev_metric = state['eval_prev_metric']

    def batch_end(self, state: State, logger: Logger):
        if self.train_prev_loss:
            logger.log_metrics({
                'loss/train/total_fdiff':
                    state.loss.item() - self.train_prev_loss
            })

        self.train_prev_loss = state.loss.item()

        if self.train_prev_metric:
            for k in state.train_metric_values.keys():
                logger.log_metrics({
                    f'metrics/train/{k}_fdiff':
                        state.train_metric_values[k].item() -
                        self.train_prev_metric[k]
                })

        for k in state.train_metric_values.keys():
            self.train_prev_metric[k] = state.train_metric_values[k].item()

    def eval_end(self, state: State, logger: Logger):
        if self.eval_prev_metric:
            for k in state.eval_metric_values.keys():
                logger.log_metrics({
                    f'metrics/eval/{k}_fdiff':
                        state.eval_metric_values[k].item() -
                        self.eval_prev_metric[k]
                })

        for k in state.eval_metric_values.keys():
            self.eval_prev_metric[k] = state.eval_metric_values[k].item()
