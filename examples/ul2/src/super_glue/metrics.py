# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import torch
from torchmetrics import Metric


class ExactMatch(Metric):
    is_differentiable = False
    higher_is_better = True
    full_state_update = False

    def __init__(self, ignore_index: Optional[int] = -100):
        super().__init__()
        self.ignore_index = ignore_index
        self.add_state('correct', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')

    @staticmethod
    def _input_format(preds: torch.Tensor, target: torch.Tensor):
        if preds.ndim == target.ndim:
            assert preds.shape == target.shape
            return preds, target

        else:
            preds = preds.argmax(-1)
            assert preds.shape == target.shape
            return preds, target

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds, target = self._input_format(preds, target)

        if self.ignore_index is not None:
            token_match = torch.logical_or(preds == target,
                                           target == self.ignore_index)
        else:
            token_match = preds == target
        exact_match = torch.all(token_match, dim=-1)

        self.correct += torch.sum(exact_match)
        self.total += exact_match.numel()

    def compute(self):
        return self.correct.float() / self.total
