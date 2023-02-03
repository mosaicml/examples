# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

"""A preliminary implementation of SuperGLUE for EncDec fine-tuning."""

import logging
from typing import Any, List, Mapping, Optional, cast

import datasets
import transformers
from composer.core.evaluator import Evaluator
from composer.core.types import Dataset
from composer.utils import dist
from omegaconf.listconfig import ListConfig
from torch.utils.data import DataLoader

__all__ = ['build_super_glue_task_dataloader']

log = logging.getLogger(__name__)

# HuggingFace hardcodes the ignore index to -100
_HF_IGNORE_INDEX = -100

_task_prefix_column_names = {
    'boolq': ['passage', 'question'],
    'cb': ['hypothesis', 'premise'],
    'copa': ['choice1', 'choice2', 'premise', 'question'],
    'multirc': ['question', 'answer', 'paragraph'],
    'record': ['passage', 'query'],
    'rte': ['hypothesis', 'premise'],
    'wic': ['sentence1', 'sentence2', 'word'],
    'wsc': ['text', 'span1_text', 'span2_text'],
}
_task_target_maps = {
    'boolq': {
        0: 'False',
        1: 'True'
    },
    'cb': {
        0: 'entailment',
        1: 'not_entailment',
        2: 'neutral'
    },
    'copa': {
        0: 'choice1',
        1: 'choice2'
    },
    'multirc': {
        0: 'False',
        1: 'True'
    },
    'record':
        lambda example: ''
        if len(example['answers']) == 0 else example['answers'][0],
    'rte': {
        0: 'entailment',
        1: 'not_entailment',
        2: 'neutral'
    },
    'wic': {
        0: 'False',
        1: 'True'
    },
    'wsc': {
        0: 'False',
        1: 'True'
    },
}


def build_super_glue_task_dataloader(cfg: Mapping[str, Any],
                                     device_batch_size: int,
                                     mode: Optional[str] = None):
    """Builds a dataloader for training or evaluating SuperGLUE task(s)"""
    if cfg.get('name', 'super_glue') != 'super_glue':
        raise NameError(
            f'Tried to build super_glue dataloader with cfg.name={cfg.name}')

    # One task
    if isinstance(cfg.dataset.task, str):
        dataset = create_super_glue_dataset(
            cfg.dataset.task,
            cfg.dataset.tokenizer_name,
            cfg.dataset.split,
            cfg.dataset.max_seq_length,
            cfg.dataset.decoder_only_format,
            extra_prefix=cfg.dataset.get('extra_prefix'))
        return _build_dataloader(dataset, cfg, device_batch_size)

    # Task mixture
    elif isinstance(cfg.dataset.task, (list, tuple, ListConfig)):
        tasks = [task.lower() for task in cfg.dataset.task]

        if mode not in ['train', 'eval']:
            raise ValueError(
                'When using multiple tasks, `mode` argument must be set to ' +\
                f'either `train` or `eval`, but got mode={mode}.'
            )

        super_glue_datasets = [
            create_super_glue_dataset(
                task,
                cfg.dataset.tokenizer_name,
                cfg.dataset.split,
                cfg.dataset.max_seq_length,
                cfg.dataset.decoder_only_format,
                extra_prefix=cfg.dataset.get('extra_prefix')) for task in tasks
        ]

        # Merge these task datasets into one dataset, sampling from each with
        if mode == 'train':
            d_lengths = [len(dset) for dset in super_glue_datasets]
            probs = [d_length / sum(d_lengths) for d_length in d_lengths]
            dataset = datasets.interleave_datasets(
                super_glue_datasets,
                probabilities=probs,
                stopping_strategy='all_exhausted')
            return _build_dataloader(dataset, cfg, device_batch_size)

        # Create a list of evaluators, with one evaluator per task
        else:
            evaluators = []
            for task, dataset in zip(tasks, super_glue_datasets):
                evaluator = Evaluator(label=task,
                                      dataloader=_build_dataloader(
                                          dataset, cfg, device_batch_size),
                                      metric_names=['ExactMatch'])
                evaluators.append(evaluator)
            return evaluators

    else:
        raise ValueError(
            f'Unrecognized task/type: {cfg.dataset.task}, with type {type(cfg.dataset.task)}'
        )


def create_super_glue_dataset(
    task: str,
    tokenizer_name: str,
    split: str,
    max_seq_length: int = 256,
    decoder_only_format: bool = False,
    max_retries: int = 10,
    num_workers: int = 0,
    extra_prefix: Optional[str] = None,
):
    if task not in _task_prefix_column_names:
        raise ValueError(
            f'task ({task}) must be one of {_task_prefix_column_names.keys()}')

    if (max_seq_length % 8) != 0:
        log.warning(
            'For performance, a max_seq_length as a multiple of 8 is recommended.'
        )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_name,
        model_max_length=max_seq_length)  #type: ignore (thirdparty)

    log.info(f'Loading {task.upper()} on rank {dist.get_global_rank()}')
    download_config = datasets.DownloadConfig(max_retries=max_retries)
    dataset = datasets.load_dataset(
        'super_glue',
        task,
        split=split,
        download_config=download_config,
    )

    log.info(
        f'Starting tokenization by preprocessing over {num_workers} threads!')
    prefix_column_order = _task_prefix_column_names[task]
    target_extractor_map = _task_target_maps[task]
    if isinstance(target_extractor_map, dict):
        target_extractor = lambda ex: target_extractor_map[ex['label']]
    else:
        assert callable(target_extractor_map)
        target_extractor = target_extractor_map

    def convert_padding_token(input_ids: List,
                              attention_mask: List,
                              new_pad_id: int = _HF_IGNORE_INDEX):
        n_non_padded = sum(attention_mask)
        n_total = len(input_ids)

        if n_non_padded == n_total:
            return input_ids

        non_padded_input_ids = input_ids[:n_non_padded]
        return non_padded_input_ids + ([new_pad_id] * (n_total - n_non_padded))

    def tokenize_function(inp: Mapping[str, Any]):
        # truncates sentences to max_length or pads them to max_length
        if extra_prefix is not None:
            prefix = f'{extra_prefix} {task}'
        else:
            prefix = task
        text = [prefix]
        for prefix_column in prefix_column_order:
            text.append(f'{prefix_column}: {inp[prefix_column]}')
        encoder_text = '\n'.join(text)

        decoder_text = target_extractor(inp)

        if decoder_only_format:
            raise NotImplementedError  # TODO(Alex): Implement this!

        else:
            encoder_dict = tokenizer(
                text=encoder_text,
                padding='max_length',
                max_length=max_seq_length,
                truncation=True,
            )
            decoder_dict = tokenizer(
                text=decoder_text,
                padding='max_length',
                max_length=20,
                truncation=True,
            )

            # HF hardcodes the ignore_index to -100 in the loss, so ensure that
            # the padding id is -100 for the loss-generating batch keys
            decoder_dict['input_ids'] = convert_padding_token(
                decoder_dict['input_ids'],
                decoder_dict['attention_mask'],
                new_pad_id=_HF_IGNORE_INDEX)

            return {
                'input_ids': encoder_dict.input_ids,
                'attention_mask': encoder_dict.attention_mask,
                'labels': decoder_dict.input_ids,
                'decoder_attention_mask': decoder_dict.attention_mask,
            }

    assert isinstance(dataset, datasets.Dataset)
    columns_to_remove = list(dataset[0].keys())
    dataset = dataset.map(
        tokenize_function,
        batched=False,
        num_proc=None if num_workers == 0 else num_workers,
        remove_columns=columns_to_remove,
        new_fingerprint=
        f'{task}-{tokenizer_name}-tokenization-{max_seq_length}-{split}',
        load_from_cache_file=False,
    )
    return dataset


def _build_dataloader(dataset, cfg, device_batch_size):
    dataset = cast(Dataset, dataset)

    return DataLoader(
        dataset,
        collate_fn=transformers.default_data_collator,
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
