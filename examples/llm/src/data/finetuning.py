# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any, Dict, List, Mapping, Optional, Union

import torch
import transformers
from composer.utils import dist
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from examples.llm.src.data._tasks import dataset_constructor

log = logging.getLogger(__name__)

# HuggingFace hardcodes the ignore index to -100
_HF_IGNORE_INDEX = -100


def build_finetuning_dataloader(cfg: Mapping[str, Any], device_batch_size: int):
    """Builds a dataloader for training or evaluating.

    Args:
        cfg (Mapping): The config for your dataset/dataloader. The config needs
            to define the following:
            - cfg.dataset.name (e.g. "HuggingFaceH4/alpaca"; must be registered in `dataset_constructor`)
            - cfg.dataset.split (e.g. "train" or "validation")
            - cfg.dataset.tokenizer_name (e.g. "gpt2")
            - cfg.dataset.max_seq_length (e.g., 512)
            - cfg.dataset.decoder_only_format (should be True for a GPT model)
            - cfg.drop_last (e.g. False)
            - cfg.shuffle (e.g. True for training, False for validation)
            - cfg.num_workers (e.g. 8)
            - cfg.pin_memory (e.g. True)
            - cfg.prefetch_factor (e.g. 2)
            - cfg.persistent_workers (e.g. True)
            - cfg.timeout (e.g. 30)

        device_batch_size (int): The batch size that should be loaded on each device.

    Returns:
        A pytorch dataloader
    """
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        cfg.dataset.tokenizer_name,
        model_max_length=cfg.dataset.max_seq_length,
        padding_side='left' if cfg.dataset.decoder_only_format else
        'right')  #type: ignore (thirdparty)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # custom for P3
    dataset = dataset_constructor.build(cfg.dataset.name, cfg.dataset.subset,
                                        tokenizer, cfg.dataset.split)

    return DataLoader(
        dataset,
        collate_fn=Seq2SeqFinetuningCollator(
            tokenizer,
            max_seq_length=cfg.dataset.max_seq_length,
            decoder_only_format=cfg.dataset.decoder_only_format,
            allow_pad_trimming=cfg.dataset.get('allow_pad_trimming', False),
        ),
        batch_size=device_batch_size,
        sampler=dist.get_sampler(dataset,
                                 drop_last=cfg.drop_last,
                                 shuffle=cfg.shuffle),
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        prefetch_factor=cfg.prefetch_factor,
        persistent_workers=cfg.persistent_workers,
        timeout=cfg.timeout,
    )


class Seq2SeqFinetuningCollator:

    def __init__(
        self,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        max_seq_length: int,
        decoder_only_format: bool,
        allow_pad_trimming: bool = False,
        separator_text: Optional[Union[str, bool]] = None,
        format_for_generation: bool = False,
        batch_metadata: Optional[Dict[str, Any]] = None,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.decoder_only_format = decoder_only_format
        self.format_for_generation = format_for_generation
        self.batch_metadata = batch_metadata or {}

        # Trimming will always be skipped on at least the first __call__
        self._allow_pad_trimming = allow_pad_trimming
        self._seen_first_batch = False

        illegal_keys = [
            'input_ids', 'labels', 'attention_mask', 'decoder_input_ids',
            'decoder_attention_mask', 'generate_output'
        ]
        found_keys = []
        for illegal_key in illegal_keys:
            if illegal_key in self.batch_metadata:
                found_keys.append(illegal_key)
        if found_keys:
            raise ValueError(
                f'The following keys are in batch_metadata but are not allowed: {", ".join(found_keys)}.'
            )
        if self.format_for_generation:
            self.batch_metadata['generate_output'] = True

        if (max_seq_length % 8) != 0:
            log.warning(
                'For performance, a max_seq_length as a multiple of 8 is recommended.'
            )

        if self.tokenizer.pad_token_id is None:
            raise ValueError(
                f'{self.__class__.__name__} requires that the tokenizer has the pad token set, but it is None'
            )

        self.separator_tokens = []
        if separator_text and decoder_only_format:
            if separator_text == True:
                # Use the tokenizer's sep token or throw an error if undefined
                if self.tokenizer.sep_token_id is None:
                    raise ValueError(
                        'Setting separator_text=True requires that the tokenizer has sep_token_id but it has not been set. ' +\
                        'Please pass a string argument for separator_text or set sep_token_id in the tokenizer.'
                    )
                self.separator_tokens = [self.tokenizer.sep_token_id]
            else:
                # Convert the string separator_text into token(s)
                self.separator_tokens = tokenizer(
                    separator_text, add_special_tokens=False).input_ids

    def __call__(self, examples: List[Dict[str,
                                           Any]]) -> Dict[str, torch.Tensor]:
        for check_key in ['input_ids', 'labels', 'attention_mask']:
            if check_key not in examples[0]:
                raise KeyError(
                    f'Examples returned by dataset do not include required key: {check_key}'
                )

        def ensure_list(x: Union[List, torch.Tensor]):
            if isinstance(x, torch.Tensor):
                x = list(x.flatten())
            assert isinstance(x, list)
            return x

        # The encoder-decoder case is has some gotchas.
        # Steps are explained in comments.
        if not self.decoder_only_format:
            for example in examples:
                context = ensure_list(example['input_ids'])
                target = ensure_list(example['labels'])
                # ... first, get rid of any padding that was already applied
                context = [
                    t for t in context if t != self.tokenizer.pad_token_id
                ]
                target = [t for t in target if t != self.tokenizer.pad_token_id]
                # ... second, ensure that the target text ends with an eos tag
                if target[-1] != self.tokenizer.eos_token_id:
                    target = target + [self.tokenizer.eos_token_id]
                # ... third, we need to pad labels ourselves. Because HF.
                if len(target) < self.max_seq_length:
                    i_pad = [_HF_IGNORE_INDEX
                            ] * (self.max_seq_length - len(target))
                    target = target + i_pad
                else:
                    raise ValueError(
                        f'max_seq_length={self.max_seq_length} is too small to fit ' +\
                        f'the target token sequence of length={len(target)}.')

                # We might need to truncate the context. Preserve the end.
                context = context[-self.max_seq_length:]

                # Back into the example
                example['input_ids'] = context
                example['attention_mask'] = [1] * len(context)
                example['labels'] = target

            # Batch examples into a single dict (this also pads)
            batch = self.tokenizer.pad(
                examples,
                padding='max_length',
                max_length=self.max_seq_length,
                return_tensors='pt',
            )
            # We're still missing decoder_input_ids and decoder_attention_mask
            batch['decoder_input_ids'] = torch.cat([
                torch.full((len(examples), 1), self.tokenizer.pad_token_id),
                batch['labels']
            ],
                                                   dim=1)
            batch['decoder_input_ids'].masked_fill_(
                batch['decoder_input_ids'] == _HF_IGNORE_INDEX,
                self.tokenizer.pad_token_id)
            batch['decoder_attention_mask'] = torch.not_equal(
                batch['labels'], _HF_IGNORE_INDEX)

            # This logic prevents trimming on at least the first batch
            if not (self._allow_pad_trimming and self._seen_first_batch):
                self._seen_first_batch = True
                return batch
            self._seen_first_batch = True

            # The batch is now valid, but we can trim padding for efficiency
            multiple_of = 8
            # (first for the encoder)
            n_non_padding = batch['attention_mask'].sum(dim=1).max()
            keep_tokens = int(multiple_of *
                              torch.ceil(n_non_padding / multiple_of))
            for k in ['input_ids', 'attention_mask']:
                batch[k] = batch[k][:, :keep_tokens].contiguous()
            # (then for the decoder)
            n_non_padding = batch['decoder_attention_mask'].sum(dim=1).max()
            keep_tokens = int(multiple_of *
                              torch.ceil(n_non_padding / multiple_of))
            for k in ['decoder_input_ids', 'decoder_attention_mask', 'labels']:
                batch[k] = batch[k][:, :keep_tokens].contiguous()

            # Add batch_metadata
            batch_size = batch['input_ids'].shape[0]
            batch.update({
                k: torch.tensor([v] * batch_size)
                for k, v in self.batch_metadata.items()
            })

            return batch

        # The decoder-only case is also somewhat involved...
        for example in examples:
            context = ensure_list(example['input_ids'])
            target = ensure_list(example['labels'])
            # First, get rid of any padding tokens
            context = [t for t in context if t != self.tokenizer.pad_token_id]
            target = [t for t in target if t != self.tokenizer.pad_token_id]
            # Second, append any separator tokens to the context tokens
            if self.separator_tokens:
                context = context + self.separator_tokens
            # Third, ensure that the target text ends with an eos tag
            if target[-1] != self.tokenizer.eos_token_id:
                target = target + [self.tokenizer.eos_token_id]

            n_context = len(context)
            n_target = len(target)

            if n_target > self.max_seq_length:
                raise ValueError(
                    f'max_seq_length={self.max_seq_length} is too small to fit ' +\
                    f'the target token sequence of length={n_target}.')

            if self.format_for_generation:
                # When formatting for generation, we need to keep input_ids and
                # labels separate. The input_ids (context) will be fed into the
                # generator and the labels will be used by the eval metric.
                input_ids = context[-self.max_seq_length:]
                n_context = len(input_ids)
                attention_mask = [1] * n_context
                bidirectional_mask = [1] * n_context
                # Annoyingly, we need to pad the everything but input_ids
                # and attention_mask ourselves
                i_pad = [self.tokenizer.pad_token_id
                        ] * (self.max_seq_length - n_target)
                z_pad = [0] * (self.max_seq_length - n_context)
                if self.tokenizer.padding_side == 'left':
                    labels = i_pad + target
                    bidirectional_mask = z_pad + bidirectional_mask
                else:
                    labels = target + i_pad
                    bidirectional_mask = bidirectional_mask + z_pad

            else:
                # We need to concatenate the context and target to get the
                # full input sequence, cutting off any excess tokens from the
                # start of the context
                if n_context + n_target > self.max_seq_length:
                    n_context = self.max_seq_length - n_target
                    context = context[-n_context:]
                n_total = n_context + n_target

                input_ids = context + target
                labels = ([_HF_IGNORE_INDEX] * n_context) + target
                attention_mask = [1] * n_total
                # bidirectional_mask is used by our prefix lm model variants
                bidirectional_mask = ([1] * n_context) + ([0] * n_target)

                # Annoyingly, we need to pad the everything but input_ids
                # and attention_mask ourselves
                i_pad = [_HF_IGNORE_INDEX] * (self.max_seq_length - n_total)
                z_pad = [0] * (self.max_seq_length - n_total)
                if self.tokenizer.padding_side == 'left':
                    labels = i_pad + labels
                    bidirectional_mask = z_pad + bidirectional_mask
                else:
                    labels = labels + i_pad
                    bidirectional_mask = bidirectional_mask + z_pad

            # Update the example
            example['input_ids'] = input_ids
            example['labels'] = labels
            example['attention_mask'] = attention_mask
            example['bidirectional_mask'] = bidirectional_mask

        batch = self.tokenizer.pad(
            examples,
            padding='max_length',
            max_length=self.max_seq_length,
            return_tensors='pt',
        )

        # This logic prevents trimming on at least the first batch
        if not (self._allow_pad_trimming and self._seen_first_batch):
            self._seen_first_batch = True
            return batch
        self._seen_first_batch = True

        # The batch is ready, but we can trim padding for efficiency
        multiple_of = 8

        n_non_padding = batch['attention_mask'].sum(dim=1).max()
        keep_tokens = int(multiple_of * torch.ceil(n_non_padding / multiple_of))
        for k, v in batch.items():
            if len(v.shape) < 2:
                continue
            if k == 'labels' and self.format_for_generation:
                continue
            if self.tokenizer.padding_side == 'left':
                batch[k] = v[:, -keep_tokens:].contiguous()
            else:
                batch[k] = v[:, :keep_tokens].contiguous()

        # Add batch_metadata
        batch_size = batch['input_ids'].shape[0]
        batch.update({
            k: torch.tensor([v] * batch_size)
            for k, v in self.batch_metadata.items()
        })

        return batch


if __name__ == '__main__':
    from omegaconf import OmegaConf as om
    cfg = om.create({
        'dataset': {
            'name': 'HuggingFaceH4/alpaca',
            'split': 'train',
            'tokenizer_name': 'gpt2',
            'max_seq_length': 2048,
            'decoder_only_format': True,
            'separator_text': False,
            'allow_pad_trimming': False,
        },
        'drop_last': False,
        'shuffle': True,
        'num_workers': 0,
        'pin_memory': False,
        'prefetch_factor': 2,
        'persistent_workers': False,
        'timeout': 0
    })

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        cfg.dataset.tokenizer_name)

    # dataloader = build_super_glue_task_dataloader(cfg, device_batch_size=2, mode='train')
    dataloader = build_finetuning_dataloader(cfg, device_batch_size=2)

    # for evaluator in evaluators:
    #     dataloader = evaluator.dataloader.dataloader
    import torch
    for i, batch in enumerate(dataloader):
        print(f'-----Batch {i}-----')
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                print(k, v.shape)
            else:
                print(k, v)
        for j in range(dataloader.batch_size):
            print(f'--- Sample {j} ---')
            if cfg.dataset.decoder_only_format:
                print(
                    'INPUT IDS:',
                    tokenizer.decode(
                        batch['input_ids'][j, batch['attention_mask'][j] == 1],
                        skip_special_tokens=True))
                print(
                    'CONTEXT:  ',
                    tokenizer.decode(
                        batch['input_ids'][j,
                                           batch['bidirectional_mask'][j] == 1],
                        skip_special_tokens=True))
                print(
                    'TARGET:   ',
                    tokenizer.decode(batch['input_ids'][
                        j, batch['labels'][j] != _HF_IGNORE_INDEX],
                                     skip_special_tokens=True))
            else:
                print(
                    'CONTEXT:  ',
                    tokenizer.decode(
                        batch['input_ids'][j, batch['attention_mask'][j] == 1],
                        skip_special_tokens=True))
                print(
                    'TARGET:   ',
                    tokenizer.decode(batch['labels'][
                        j, batch['decoder_attention_mask'][j] == 1],
                                     skip_special_tokens=True))
        print('   ')
        if i >= 5:
            break
