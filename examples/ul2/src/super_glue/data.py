# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

"""A preliminary implementation of SuperGLUE for EncDec fine-tuning."""

import logging
from typing import Any, Mapping, Optional, Union

import datasets
import transformers
from composer.core.evaluator import Evaluator
from composer.core.types import Dataset
from composer.utils import dist
from omegaconf.listconfig import ListConfig
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from examples.ul2.src.utils import Seq2SeqFinetuningCollator

__all__ = ['build_super_glue_task_dataloader']

log = logging.getLogger(__name__)

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

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        cfg.dataset.tokenizer_name,
        model_max_length=cfg.dataset.max_seq_length)  #type: ignore (thirdparty)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def _build_dataloader(dataset: Dataset):
        return DataLoader(
            dataset,
            collate_fn=Seq2SeqFinetuningCollator(
                tokenizer,
                max_seq_length=cfg.dataset.max_seq_length,
                decoder_only_format=cfg.dataset.decoder_only_format,
                separator_text=cfg.dataset.get('separator_text'),
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

    # One task
    if isinstance(cfg.dataset.task, str):
        task = cfg.dataset.task.lower()
        dataset = create_super_glue_dataset(
            task,
            tokenizer,
            cfg.dataset.split,
            cfg.dataset.decoder_only_format,
            extra_prefix=cfg.dataset.get('extra_prefix'),
            include_extra_example_info=True)
        return _build_dataloader(dataset)

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
                tokenizer,
                cfg.dataset.split,
                cfg.dataset.decoder_only_format,
                extra_prefix=cfg.dataset.get('extra_prefix'),
                include_extra_example_info=(mode == 'eval')) for task in tasks
        ]

        # Merge these task datasets into one dataset, sampling from each with
        # probability proportional to their number of examples
        if mode == 'train':
            d_lengths = [len(dset) for dset in super_glue_datasets]
            probs = [d_length / sum(d_lengths) for d_length in d_lengths]
            dataset = datasets.interleave_datasets(
                super_glue_datasets,
                probabilities=probs,
                stopping_strategy='all_exhausted')
            return _build_dataloader(dataset)

        # Create a list of evaluators, with one evaluator per task
        else:
            evaluators = []
            for task, dataset in zip(tasks, super_glue_datasets):
                evaluator = Evaluator(label=task,
                                      dataloader=_build_dataloader(dataset),
                                      metric_names=['ExactMatch'])
                evaluators.append(evaluator)
            return evaluators

    else:
        raise ValueError(
            f'Unrecognized task/type: {cfg.dataset.task}, with type {type(cfg.dataset.task)}'
        )


def create_super_glue_dataset(
    task: str,
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    split: str,
    max_retries: int = 10,
    num_workers: int = 0,
    extra_prefix: Optional[str] = None,
    include_extra_example_info: bool = False,
):
    if task not in _task_prefix_column_names:
        raise ValueError(
            f'task ({task}) must be one of {_task_prefix_column_names.keys()}')

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

    def tokenize_function(inp: Mapping[str, Any]):
        """Format the text string and simply tokenize."""
        if extra_prefix is not None:
            prefix = f'{extra_prefix} {task}'
        else:
            prefix = task
        text = [prefix]
        for prefix_column in prefix_column_order:
            text.append(f'{prefix_column}: {inp[prefix_column]}')
        context = '\n'.join(text)

        target = target_extractor(inp)

        model_args = tokenizer(
            text=context,
            text_target=target,
        )

        if include_extra_example_info:
            # Keep things like `idx` and `label`
            model_args.update(
                {k: v for k, v in inp.items() if not isinstance(v, str)})

        return model_args

    assert isinstance(dataset, datasets.Dataset)

    columns_to_remove = list(dataset[0].keys())

    safe_tok_name = tokenizer.name_or_path.replace('/', ',')
    fingerprint = f'{task}-{safe_tok_name}-tokenization-{split}'
    if include_extra_example_info:
        fingerprint += '-extras'

    dataset = dataset.map(
        tokenize_function,
        batched=False,
        num_proc=None if num_workers == 0 else num_workers,
        remove_columns=columns_to_remove,
        new_fingerprint=fingerprint,
        load_from_cache_file=True,
    )
    return dataset
