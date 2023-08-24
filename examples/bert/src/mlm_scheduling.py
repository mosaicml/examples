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

    def __init__(self,
                 dist_mlm_probability,
                 subset_masking_rate=None,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.dist_mlm_probability = dist_mlm_probability
        self.subset_masking_rate = subset_masking_rate

    @property
    def mlm_probability(self):
        return self.dist_mlm_probability.value

    @mlm_probability.setter
    def mlm_probability(self, _):
        return

    def torch_mask_tokens(self, inputs, special_tokens_mask=None):
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """

        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(
                    val, already_has_special_tokens=True)
                for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask,
                                               dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        if self.subset_masking_rate:  # Defining loss on subset of total masked tokens
            step_subset_mask_rate = self.subset_masking_rate / self.mlm_probability
            subset_probability_matrix = torch.zeros_like(
                probability_matrix, dtype=probability_matrix.dtype)
            subset_probability_matrix[probability_matrix !=
                                      0.0] = step_subset_mask_rate
            subset_indices = torch.bernoulli(subset_probability_matrix).bool()
            labels[~subset_indices] = -100

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(
            labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(
            labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer),
                                     labels.shape,
                                     dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels


"""
Definition of scheduled random token substition collator
"""


class ScheduledDataCollatorForRTS(transformers.DataCollatorForLanguageModeling):

    def __init__(self,
                 dist_mlm_probability,
                 subset_masking_rate=None,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.dist_mlm_probability = dist_mlm_probability
        self.subset_masking_rate = subset_masking_rate

    @property
    def mlm_probability(self):
        return self.dist_mlm_probability.value

    @mlm_probability.setter
    def mlm_probability(self, _):
        return

    def torch_mask_tokens(self, inputs, special_tokens_mask=None):
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """

        labels = torch.zeros_like(inputs)
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(inputs.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(
                    val, already_has_special_tokens=True)
                for val in inputs.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask,
                                               dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[masked_indices] = 1  # Set label to be positive for swapped words

        random_words = torch.randint(len(self.tokenizer),
                                     labels.shape,
                                     dtype=torch.long)
        inputs[masked_indices] = random_words[masked_indices]

        ## 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        #indices_replaced = torch.bernoulli(torch.full(labels.shape,
        #0.8)).bool() & masked_indices
        #inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(
        #self.tokenizer.mask_token)

        ## 10% of the time, we replace masked input tokens with random word
        #indices_random = torch.bernoulli(torch.full(
        #labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        #random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        #inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels
