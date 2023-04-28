# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

import os
from functools import partial
from typing import Any, Callable, Dict, Optional, Union

import datasets
import transformers
from omegaconf import DictConfig
from streaming import StreamingDataset
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

__all__ = ['dataset_constructor']

Tokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]


class StreamingFinetuningDataset(StreamingDataset):
    """Finetuning dataset with flexible tokenization using StreamingDataset.

    Args:
        local (str): Local dataset directory where shards are cached by split.
        tokenizer_name (str): The name of the HuggingFace tokenizer to use to
            tokenize samples.
        tokenize_function (callable): A function that takes a text sample (dict) and a tokenizer,
            and returns the tokenized sample (dict). The tokenized sample must have `input_ids`
            and `labels`, with `input_ids` providing the input sequence and `labels` providing the
            (target) output sequence, in a sequence-to-sequence task.
        remote (str, optional): Download shards from this remote path or directory. If None, this
            rank and worker's partition of the dataset must all exist locally. Defaults to ``None``.
        split (str, optional): Which dataset split to use, if any. Defaults to ``None``.
        shuffle (bool): Whether to iterate over the samples in randomized order. Defaults to ``False``.
        predownload (int, optional): Target number of samples ahead to download the shards of while
            iterating. Defaults to ``100_000``.
        keep_zip (bool, optional): Whether to keep or delete the compressed file when
            decompressing downloaded shards. If set to None, keep if remote is local. Defaults to
            ``None``.
        download_retry (int): Number of download re-attempts before giving up. Defaults to ``2``.
        download_timeout (float): Number of seconds to wait for a shard to download before raising
            an exception. Defaults to ``60``.
        validate_hash (str, optional): Optional hash or checksum algorithm to use to validate
            shards. Defaults to ``None``.
        shuffle_seed (int): Seed for Deterministic data shuffling. Defaults to ``9176``.
        num_canonical_nodes (int, optional): Canonical number of nodes for shuffling with resumption.
            If ``None``, defaults to the number of nodes of the initial run. Defaults to 128.
        batch_size (int, optional): Batch size of its DataLoader, which affects how the dataset is
            partitioned over the workers. Defaults to ``None``.
    """

    def __init__(self,
                 local: str,
                 tokenizer_name: str,
                 tokenize_function: Callable[[Dict, Tokenizer], Dict],
                 remote: Optional[str] = None,
                 split: Optional[str] = None,
                 shuffle: bool = False,
                 predownload: Optional[int] = 100_000,
                 keep_zip: Optional[bool] = None,
                 download_retry: int = 2,
                 download_timeout: float = 60,
                 validate_hash: Optional[str] = None,
                 shuffle_seed: int = 9176,
                 num_canonical_nodes: Optional[int] = 128,
                 batch_size: Optional[int] = None,
                 **kwargs: Any):

        if len(kwargs) > 0:
            raise ValueError(
                f'StreamingTextDataset() got an unexpected keyword argument: {kwargs}'
            )

        if remote is None or (local == remote):
            if os.path.isdir(local):
                contents = set(os.listdir(local))
                if split not in contents:
                    raise ValueError(
                        f'local directory {local} does not contain split {split}'
                    )

        # Build Dataset
        super().__init__(local=local,
                         remote=remote,
                         split=split,
                         shuffle=shuffle,
                         predownload=predownload,
                         keep_zip=keep_zip,
                         download_retry=download_retry,
                         download_timeout=download_timeout,
                         validate_hash=validate_hash,
                         shuffle_seed=shuffle_seed,
                         num_canonical_nodes=num_canonical_nodes,
                         batch_size=batch_size)

        # Build tokenizer
        os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            tokenizer_name)

        # suppress warnings when using longer 'max_seq_len'
        self.tokenizer.model_max_length = int(1e30)

        # Use EOS as the pad token if none exists
        if self.tokenizer.pad_token is None:  # type: ignore
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.tokenize_function = tokenize_function

    # How to process a sample
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = super().__getitem__(idx)
        return self.tokenize_function(sample, self.tokenizer)


class DatasetConstructor:

    def __init__(self):
        self._task_tokenization_registry: Dict[str, Callable] = {}

    def register(self, *names: str):
        """Decorator for registering tokenization functions."""

        def _register_func(name: str, func: Callable) -> None:
            if name in self._task_tokenization_registry:
                raise ValueError(
                    f'A tokenization function has already been registered with {name=}.'
                )
            self._task_tokenization_registry[name] = func
            return

        def wrapper(func: Callable) -> Callable:
            for name in names:
                _register_func(name, func)
            return func

        return wrapper

    def build(self, cfg: DictConfig, tokenizer: Tokenizer):
        dataset_name = cfg.name
        split = cfg.split
        kwargs = cfg.get('kwargs', {})

        if dataset_name not in self._task_tokenization_registry:
            raise ValueError(
                f'{dataset_name} is not a registered dataset. ' +
                f'Available datasets: {self._task_tokenization_registry.keys()}'
            )

        dataset = datasets.load_dataset(dataset_name, split=split, **kwargs)

        tokenize_function = partial(
            self._task_tokenization_registry[dataset_name], tokenizer=tokenizer)

        columns_to_remove = list(dataset[0].keys())
        dataset = dataset.map(
            tokenize_function,
            batched=False,
            remove_columns=columns_to_remove,
        )
        return dataset

    def build_from_streaming(self, dataset_name: str, tokenizer_name: str,
                             **kwargs: Any):

        tokenize_function = self._task_tokenization_registry[dataset_name]

        dataset = StreamingFinetuningDataset(
            tokenizer_name=tokenizer_name,
            tokenize_function=tokenize_function,
            **kwargs,
        )

        return dataset


dataset_constructor = DatasetConstructor()


@dataset_constructor.register('tatsu-lab/alpaca')
def alpaca_tokenize_function(inp: Dict, tokenizer: Tokenizer):
    """Format the text string and simply tokenize."""
    # `text` is the text the encoder receives (i.e. the prompt)
    # `text_target` is the target output the decoder should produce
    try:
        prompt, response = inp['text'].split('### Response:')
    except:
        print(inp)
        raise
    return tokenizer(
        text=prompt + '### Response:',
        text_target=response,
    )


@dataset_constructor.register('HuggingFaceH4/databricks_dolly_15k')
def dolly_tokenize_function(inp: Dict, tokenizer: Tokenizer):
    """Format the text string and simply tokenize."""
    PROMPT_FORMAT = 'Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n'
    try:
        if inp['input'] != '':
            instruction = inp['instruction'] + '\n' + inp['input']
        else:
            instruction = inp['instruction']
        prompt = PROMPT_FORMAT.format(instruction=instruction)
        response = inp['output']
    except:
        print(inp)
        raise
    return tokenizer(
        text=prompt,
        text_target=response,
    )


@dataset_constructor.register('sam-mosaic/vicuna_alpaca_hc3_chatml')
@dataset_constructor.register('sam-mosaic/full-hh-rlhf-chatml')
def simple_tokenize_function(inp: Dict, tokenizer: Tokenizer):
    """Already split, just tokenize."""
    return tokenizer(
        text=inp['prompt'],
        text_target=inp['response'],
    )


@dataset_constructor.register('laion/OIG')
def oig_tokenize_function(inp: Dict, tokenizer: Tokenizer):
    """Format the text string and simply tokenize."""
    # `text` is the text the encoder receives (i.e. the prompt)
    # `text_target` is the target output the decoder should produce
    try:
        # janky heuristic but I really want to train on this data
        # all data is OTF human: blah\n bot: blah blah
        # so this should "work" for some definition of work
        # too many of the tokens are loss generating but I am okay with that
        prompt, response = inp['text'].split(sep=':', maxsplit=1)
    except:
        print(inp)
        raise
    return tokenizer(
        text=prompt + ':',
        text_target=response,
    )


@dataset_constructor.register('bigscience/P3')
def p3_tokenize_function(inp: Dict, tokenizer: Tokenizer):
    """Format the text string and simply tokenize."""
    # `text` is the text the encoder receives (i.e. the prompt)
    # `text_target` is the target output the decoder should produce
    try:
        prompt = inp['inputs']  # instruction-finetuned prompts
        response = inp['targets']  # sequence of values
    except:
        print(inp)
        raise
    return tokenizer(
        text=prompt + ':',
        text_target=response,
    )


# Muennighoff's P3 and flan datasets share a similar convention
@dataset_constructor.register('Muennighoff/P3', 'Muennighoff/flan')
def muennighoff_tokenize_function(inp: Dict, tokenizer: Tokenizer):
    """Format the text string and simply tokenize."""
    # `text` is the text the encoder receives (i.e. the prompt)
    # `text_target` is the target output the decoder should produce
    # In this dataset, there are only 2 columns: "inputs" and "targets"
    try:
        prompt: str = inp['inputs']
        response: str = inp['targets']
        # Put a space before the response if needed
        transitions = (' ', '\n', '\t')
        if not (prompt.endswith(transitions) or
                response.startswith(transitions)):
            response = ' ' + response
    except:
        print(inp)
        raise
    return tokenizer(
        text=prompt,
        text_target=response,
    )
