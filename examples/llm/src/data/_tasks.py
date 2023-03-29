# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from functools import partial
from typing import Any, Dict, Optional, Union

import datasets
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from examples.common.text_data import StreamingTextDataset

__all__ = ['dataset_constructor']


class DatasetConstructor:

    def __init__(self):
        self._task_tokenization_registry: Dict[str, callable] = {}

    def add(self, dataset_name: str, tokenize_function: callable):
        self._task_tokenization_registry[dataset_name] = tokenize_function

    def build(self, dataset_name: str,
              tokenizer: Union[PreTrainedTokenizer,
                               PreTrainedTokenizerFast], split: str):
        assert dataset_name in self._task_tokenization_registry
        # UPDATE THIS LINE TO LOAD YOUR RAW DATASET
        dataset = datasets.load_dataset(dataset_name, split=split)

        tokenize_function = partial(
            self._task_tokenization_registry[dataset_name], tokenizer=tokenizer)

        columns_to_remove = list(dataset[0].keys())
        dataset = dataset.map(
            tokenize_function,
            batched=False,
            remove_columns=columns_to_remove,
        )
        return dataset

    def build_from_streaming(self,
                             dataset_name: str,
                             tokenizer: Union[PreTrainedTokenizer,
                                              PreTrainedTokenizerFast],
                             split: str,
                             local: str,
                             remote: Optional[str] = None,
                             **kwargs: Dict[str, Any]):

        tokenize_function = partial(
            self._task_tokenization_registry[dataset_name], tokenizer=tokenizer)

        class DynamicStreamingDataset(StreamingTextDataset):

            def __getitem__(self, idx: int) -> Dict[str, Any]:
                sample = super(StreamingTextDataset, self).__getitem__(idx)
                return tokenize_function(sample)

        dataset = DynamicStreamingDataset(
            tokenizer_name='gpt2',  # This will be ignored. Just a placeholder.
            remote=remote,
            local=local,
            split=split,
            max_seq_len=32768,  # Seq length not relevant here, so set high.
            **kwargs,
        )

        return dataset


dataset_constructor = DatasetConstructor()


def alpaca_tokenize_function(inp, tokenizer):
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


def oig_tokenize_function(inp, tokenizer):
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


def p3_tokenize_function(inp, tokenizer):
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


def muennighoff_p3_tokenize_function(inp, tokenizer):
    """Format the text string and simply tokenize."""
    # `text` is the text the encoder receives (i.e. the prompt)
    # `text_target` is the target output the decoder should produce
    # In this dataset, there are only 2 columns: "inputs" and "targets"
    try:
        prompt: str = inp['inputs']
        response: str = inp['targets']
        # Put a space before the response if needed
        transitions = (' ', '\n', ':')
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


dataset_constructor.add('tatsu-lab/alpaca', alpaca_tokenize_function)
dataset_constructor.add('laion/OIG', oig_tokenize_function)
dataset_constructor.add('bigscience/P3', p3_tokenize_function)
dataset_constructor.add('Muennighoff/P3', muennighoff_p3_tokenize_function)
