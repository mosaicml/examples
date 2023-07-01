from typing import Union
import multiprocessing

from composer import Callback, State, Logger, Event, Time
from composer.optim.scheduler import (ComposerScheduler, ConstantScheduler,
                                      CosineAnnealingScheduler, _convert_time,
                                      LinearScheduler)
from omegaconf import DictConfig
"""
Definition of schedulers and callbacks for setting the masking rate dynamically
"""


# Define special case of step-wise scheduling where decay is only performed
# once and as such define by start and terminal masking rates
class StepScheduler(ComposerScheduler):
    r"""Decays the masking rate by discrete step to new rate.
    Args:
        alpha_i (float): Multiplier of initial masking rate. Default = ``0.3``.
        alpha_f (float): Masking rate to end at. Default = ``0.15``.
        t_step (str | Time): The time step to switch masking rate. Default = ``"0.5dur"``.
    """

    def __init__(self,
                 alpha_i: float = 1,
                 alpha_f: float = 0.5,
                 t_step: Union[str, Time] = '0.5dur'):
        self.alpha_i = alpha_i
        self.alpha_f = alpha_f
        self.t_step = t_step

    def __call__(self, state: State, ssr: float = 1.0):
        t_step = _convert_time(self.t_step, state, ssr=ssr)
        current_time = state.timestamp.get(t_step.unit)

        if t_step.value > current_time.value:
            return self.alpha_i

        return self.alpha_f


class MaskingRateSetter(Callback):

    def __init__(self, scheduler: ComposerScheduler,
                 initial_masking_rate: float,
                 dynamic_masking_rate: multiprocessing.Value):
        super().__init__()
        self.scheduler = scheduler
        self.initial_masking_rate = initial_masking_rate
        self.dynamic_masking_rate = dynamic_masking_rate

    def run_event(self, event: Event, state: State, logger: Logger):
        if event == Event.BATCH_END:
            masking_rate = self.scheduler(state) * self.initial_masking_rate

            self.dynamic_masking_rate.value = masking_rate

            logger.log_metrics({'mlm_schedule/masking_rate': masking_rate})


def build_mlm_scheduler_callback(
        cfg: DictConfig, distributed_masking_rate: multiprocessing.Value):
    initial_masking_rate = cfg.initial_masking_rate
    final_masking_rate = cfg.final_masking_rate
    alpha_f = final_masking_rate / initial_masking_rate  # Multiple to reach final mlm rate

    if cfg.name == 'constant':
        mlm_schedule = ConstantScheduler()
    elif cfg.name == 'cosine':
        mlm_schedule = CosineAnnealingScheduler(alpha_f=alpha_f)
    elif cfg.name == 'linear':
        mlm_schedule = LinearScheduler(alpha_f=alpha_f)
    elif cfg.name == 'step':
        mlm_schedule = StepScheduler(alpha_f=alpha_f)
    else:
        raise ValueError(
            f'Not sure how to build masking rate scheduler: {cfg.name}')

    return MaskingRateSetter(mlm_schedule,
                             initial_masking_rate=initial_masking_rate,
                             dynamic_masking_rate=distributed_masking_rate)
