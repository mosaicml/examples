# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

from composer.core import State, TimeUnit
from composer.optim.scheduler import ComposerScheduler

__all__ = ['InverseSquareRootScheduler']


class InverseSquareRootScheduler(ComposerScheduler):
    """LR scheduler based on an inverse-square-root decay.

    TODO(Alex): Complete docstring
    """

    def __init__(
        self,
        alpha_max: float = 0.01,
        scale: float = 1.0,
    ):
        self.alpha_max = alpha_max
        assert 0 < self.alpha_max <= 1.0
        self.scale = scale
        assert self.scale > 0.0

    def __call__(self, state: State, ssr: float = 1.0):
        current_time = state.timestamp.get(
            TimeUnit.BATCH).value / (ssr * self.scale)
        current_factor = min(self.alpha_max,
                             1.0 / ((current_time + 1e-12)**0.5))
        return current_factor
