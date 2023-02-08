from typing import Union
from multiprocessing import Value

from composer import Callback, State, Logger, Event, Time
from composer.optim.scheduler import (ComposerScheduler, ConstantScheduler,
                                      CosineAnnealingScheduler, _convert_time,
                                      LinearScheduler)
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

    def __init__(self, scheduler: ComposerScheduler, initial_mlm_rate: float,
                 dynamic_mlm_rate: Value):
        super().__init__()
        self.scheduler = scheduler
        self.initial_mlm_rate = initial_mlm_rate
        self.dynamic_mlm_rate = dynamic_mlm_rate

    def run_event(self, event: Event, state: State, logger: Logger):
        if event == Event.BATCH_END:
            mlm_rate = self.scheduler(state) * self.initial_mlm_rate

            self.dynamic_mlm_rate.value = mlm_rate

            logger.log_metrics({'mlm_schedule/mlm_rate': mlm_rate})


def build_mlm_scheduler_callback(cfg, distributed_mlm_rate: Value):
    initial_mlm_rate = cfg.initial_mlm_rate
    final_mlm_rate = cfg.final_mlm_rate
    alpha_f = final_mlm_rate / initial_mlm_rate

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

    return MLMRateSetter(mlm_schedule,
                         initial_mlm_rate=initial_mlm_rate,
                         dynamic_mlm_rate=distributed_mlm_rate)


"""
Definition of Language Modelling Collator that uses dynamic masking rate
"""


class ScheduledDataCollatorForLanguageModeling(
        transformers.DataCollatorForLanguageModeling):

    def __init__(self, dist_mlm_probability, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dist_mlm_probability = dist_mlm_probability

    @property
    def mlm_probability(self):
        return self.dist_mlm_probability.value

    @mlm_probability.setter
    def mlm_probability(self, _):
        return
