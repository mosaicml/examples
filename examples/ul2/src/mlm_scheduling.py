from typing import Union
from multiprocessing import Value

from composer import Callback, State, Logger, Event, Time
from composer.optim.scheduler import (ComposerScheduler, ConstantScheduler,
                                      CosineAnnealingScheduler, _convert_time,
                                      LinearScheduler)
import torch
import transformers
"""
Definition of schedulers and callbacks for setting the masking rate dynamically
"""


# Define special case of step-wise scheduling where decay is only performed
# once and as such define by start and terminal masking rates
class StepScheduler(ComposerScheduler):
    r"""Decays the masking rate by discrete step to new rate.
    Args:
        alpha_i (float): Masking rate to start at. Default = ``0.3``.
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


class MLMRateSetter(Callback):

    def __init__(self, scheduler: ComposerScheduler, span_length: int,
                 initial_mlm_rate: float, dynamic_mlm_rate: Value):
        super().__init__()
        self.scheduler = scheduler
        self.span_length = span_length
        self.initial_mlm_rate = initial_mlm_rate
        self.dynamic_mlm_rate = dynamic_mlm_rate

    def run_event(self, event: Event, state: State, logger: Logger):
        if event == Event.BATCH_END:
            mlm_rate = self.scheduler(state) * self.initial_mlm_rate

            self.dynamic_mlm_rate.value = mlm_rate

            logger.log_metrics(
                {f'mlm_schedule/span_{self.span_length}_mlm_rate': mlm_rate})


def build_mlm_scheduler_callback(cfg, span_mean_lengths_and_ratios):
    rate_schedules = []
    cfg_span_mean_lengths_and_ratios = cfg.mixture_of_denoisers.get(
        'span_mean_lengths_and_ratios')
    for cfg_span_mean_length_and_ratio, span_mean_lengths_and_ratio in zip(
            cfg_span_mean_lengths_and_ratios, span_mean_lengths_and_ratios):
        initial_mlm_rate = cfg_span_mean_length_and_ratio[1]
        final_mlm_rate = cfg_span_mean_length_and_ratio[2]
        alpha_f = final_mlm_rate / initial_mlm_rate

        name = cfg.schedule_name
        if name == 'constant':
            mlm_schedule = ConstantScheduler()
        elif name == 'cosine':
            mlm_schedule = CosineAnnealingScheduler(alpha_f=alpha_f)
        elif name == 'linear':
            mlm_schedule = LinearScheduler(alpha_f=alpha_f)
        elif name == 'step':
            mlm_schedule = StepScheduler(alpha_f=alpha_f)
        else:
            raise ValueError(
                f'Not sure how to build masking rate scheduler: {name}')

        rate_schedules.append(
            MLMRateSetter(mlm_schedule,
                          span_length=cfg_span_mean_length_and_ratio[0],
                          initial_mlm_rate=initial_mlm_rate,
                          dynamic_mlm_rate=span_mean_lengths_and_ratio[1]))

    return rate_schedules
