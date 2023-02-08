# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

"""C4 streaming dataset conversion scripts."""
import os
import platform
from argparse import Action, ArgumentError, ArgumentParser, Namespace
from enum import Enum
from textwrap import dedent
from typing import Dict, Iterable, Optional
import warnings

import datasets as hf_datasets
import numpy as np
from streaming import MDSWriter
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase


class ConcatMode(Enum):
    NO_CONCAT = 'NO_CONCAT'
    CONCAT_TEXT = 'CONCAT_TEXT'
    CONCAT_TOKENS = 'CONCAT_TOKENS'


class EmptyOrNonexistentDirectory(Action):

    def __call__(self, parser, namespace, values, option_string=None):
        prospective_dir = values
        if os.path.isdir(prospective_dir) and len(
                os.listdir(prospective_dir)) > 0:
            raise ArgumentError(self,
                                f'--out_root={prospective_dir} must be empty')
        else:
            setattr(namespace, 'out_root', prospective_dir)


def parse_args() -> Namespace:
    """Parse commandline arguments."""
    parser = ArgumentParser(
        description=
        'Convert C4 into MDS format, optionally concatenating and tokenizing')
    parser.add_argument('--out_root',
                        type=str,
                        required=True,
                        action=EmptyOrNonexistentDirectory)
    parser.add_argument('--compression', type=str, default=None)
    parser.add_argument('--splits',
                        nargs='+',
                        default=['train', 'train_small', 'val'])

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--concat_text',
                       type=int,
                       help='Number of characters to concatenate to')
    group.add_argument(
        '--concat_tokens',
        type=int,
        help='Convert text to tokens and concatenate up to this many tokens')

    parser.add_argument('--tokenizer', type=str, required=False, default=None)
    parser.add_argument('--bos_text', type=str, required=False, default=None)
    parser.add_argument('--eos_text', type=str, required=False, default=None)

    parsed = parser.parse_args()

    if hasattr(parsed, 'concat_text') or hasattr(parsed, 'concat_tokens'):
        # Make sure we have needed concat options
        if (parsed.concat_text is not None and
                isinstance(parsed.concat_text, int) and
                parsed.bos_text is None and parsed.eos_text is None):
            parser.error(
                dedent(
                    'When setting --concat_text, you must specify at least one of --bos_text or --eos_text \
                with which to separate concatenated sequences'))
        if (parsed.concat_tokens is not None and
                isinstance(parsed.concat_tokens, int) and
                parsed.tokenizer is None):
            parser.error(
                'When setting --concat_tokens, you must specify a --tokenizer')

    # now that we have validated them, change BOS/EOS to strings
    if parsed.bos_text is None:
        parsed.bos_text = ''
    if parsed.eos_text is None:
        parsed.eos_text = ''
    return parsed


class ShardedC4(IterableDataset):
    """The class used for iterating through C4 shards when not concatenating.

    We override __iter__ and use the loops in generate_samples rather than
    hf_datasets.map because .map effectively does not work for streaming data
    sets
    """

    def __init__(self,
                 split: str,
                 expected_num_samples: int,
                 max_length: Optional[int] = None,
                 batch_size: int = 512):
        self.dataset = hf_datasets.load_dataset(path='c4',
                                                name='en',
                                                split=split,
                                                streaming=True)
        self.expected_num_samples = expected_num_samples
        self.max_length = max_length
        self.batch_size = batch_size
        self._build_loader()

    def num_shards(self):
        it = self.dataset._ex_iterable  # type: ignore
        return len(it.kwargs['filepaths'])  # type: ignore

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is not None:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            it = self.dataset._ex_iterable  # type: ignore
            shards = it.kwargs['filepaths']  # type: ignore
            assert len(shards) % num_workers == 0
            it.kwargs['filepaths'] = shards[  # type: ignore
                worker_id::num_workers]
        return iter(self.dataset)

    def _build_loader(self) -> None:
        # Multiple workers is only supported on linux machines
        if 'linux' in platform.platform().lower():
            num_workers = min(64, self.num_shards())  # type: ignore
        else:
            num_workers = 0

        # If using multiple workers, configure each worker to prefetch as many samples as it can, up to
        # the aggregate device batch size
        # If not using workers, the torch DataLoader expects the default value for prefetch_factor,
        # which non-intuitively must be 2.
        prefetch_factor = max(1, 2 * self.batch_size //
                              num_workers) if num_workers > 0 else 2

        self.loader = DataLoader(  # type: ignore
            dataset=self.dataset,
            sampler=None,
            batch_size=self.batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
        )

    def generate_samples(self) -> Iterable[Dict[str, bytes]]:
        """Generator over each dataset sample.

        Args:
            samples (IterableDataset): An iterable dataset that is multi-worker compatible

        Yields:
            Sample dicts.
        """
        total_chars = 0
        n_samples = 0
        for batch in self.loader:
            keys = list(batch.keys())
            current_bs = len(batch[keys[0]])
            for idx in range(current_bs):
                total_chars += len(batch['text'][idx].encode('utf-8'))
                yield {
                    key: batch_values[idx].encode('utf-8')
                    for key, batch_values in batch.items()
                }
                n_samples += 1
                if n_samples == self.expected_num_samples:
                    print(f'{total_chars=}')
                    return


class ConcatC4(ShardedC4):

    def __init__(self, bos_text: str, eos_text: str, *args, **kwargs):
        self.bos_text = bos_text
        self.eos_text = eos_text
        super().__init__(*args, **kwargs)

    def generate_samples(self) -> Iterable[Dict[str, bytes]]:
        """Generator over each dataset sample.

        Args:
            samples (IterableDataset): An iterable dataset that is multi-worker compatible

        Yields:
            Sample dicts.
        """

        n_samples = 0
        buffer = ''
        while True:
            for batch in self.loader:
                for sample in batch['text']:
                    buffer = buffer + self.bos_text + sample + self.eos_text
                    while len(buffer) >= self.max_length:
                        concat_sample = buffer[:self.max_length]
                        buffer = buffer[self.max_length:]

                        yield {'text': concat_sample.encode('utf-8')}

                        n_samples += 1
                        if n_samples == self.expected_num_samples:
                            return


class TokenizedC4(ShardedC4):
    """A tokenized, concatenated, iterable dataset.

    To use data created by this class and written to MDS format:

    ```python
        import torch
        from streaming.base import StreamingDataset
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained('your/tokenizer')
        ds = StreamingDataset(local='mds-data-folder', split='val')

        # note, you need to copy the numpy array because the original is non-writeable
        # and torch does not support non-writeable tensors, so you get a scary warning and
        # if you do try to write to the tensor you get undefined behavior
        tokens = torch.from_numpy(np.frombuffer(ds[0]['tokens'], dtype=np.int64).copy())
        print(tokenizer.decode(tokens))
    ```
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, bos_text: str,
                 eos_text: str, *args, **kwargs):
        self.tokenizer = tokenizer
        self.bos_text = bos_text
        self.eos_text = eos_text
        super().__init__(*args, **kwargs)

    def generate_samples(self) -> Iterable[Dict[str, bytes]]:
        """Generator over each dataset sample.

        Args:
            samples (IterableDataset): An iterable dataset that is multi-worker compatible

        Yields:
            Sample dicts.
        """

        bos_tokens = self.tokenizer(self.bos_text,
                                    truncation=False,
                                    padding=False,
                                    add_special_tokens=False)['input_ids']
        if len(bos_tokens) > 1:
            warnings.warn(
                f'You specified --concat_tokens with --bos_text, but your BOS text is not tokenizing to one token\
                , instead we got {bos_tokens}. Quit if this was in error.')

        eos_tokens = self.tokenizer(self.eos_text,
                                    truncation=False,
                                    padding=False,
                                    add_special_tokens=False)['input_ids']
        if len(eos_tokens) > 1:
            warnings.warn(
                f'You specified --concat_tokens with --eos_text, but your EOS text is not tokenizing to one token\
                , instead we got {eos_tokens}. Quit if this was in error.')

        n_samples = 0
        buffer = []
        while True:
            for batch in self.loader:
                encoded = self.tokenizer(batch['text'],
                                         truncation=False,
                                         padding=False)
                for iids in encoded['input_ids']:
                    buffer = buffer + bos_tokens + iids + eos_tokens
                    while len(buffer) >= self.max_length:
                        concat_sample = buffer[:self.max_length]
                        buffer = buffer[self.max_length:]

                        yield {
                            # convert to bytes to store in MDS binary format
                            'tokens': np.asarray(concat_sample).tobytes()
                        }
                        n_samples += 1
                        if n_samples == self.expected_num_samples:
                            return


def build_hf_c4_dataset(split: str, mode: ConcatMode, max_length: Optional[int],
                        bos_text: Optional[str], eos_text: Optional[str],
                        tokenizer: Optional[PreTrainedTokenizerBase],
                        expected_num_samples: int) -> IterableDataset:
    """Collect the samples for this dataset split.

    Args:
        split (str): Split name.
        mode (ConcatMode): NO_CONCAT, CONCAT_TEXT, or CONCAT_TOKENS
        bos_text (str): text to insert at the beginning of each sequence
        eos_text (str): text to insert at the end of each sequence
        tokenizer (PreTrainedTokenizerBase): if mode is CONCAT_TOKENS, the tokenizer to use
        expected_num_samples (int): once we have generated this many samples, exit

    Returns:
        An IterableDataset.
    """
    if mode == ConcatMode.NO_CONCAT:
        dataset = ShardedC4(split, expected_num_samples=expected_num_samples)
    elif mode == ConcatMode.CONCAT_TEXT:
        dataset = ConcatC4(split=split,
                           max_length=max_length,
                           bos_text=bos_text,
                           eos_text=eos_text,
                           expected_num_samples=expected_num_samples)
    else:
        dataset = TokenizedC4(split=split,
                              tokenizer=tokenizer,
                              max_length=max_length,
                              bos_text=bos_text,
                              eos_text=eos_text,
                              expected_num_samples=expected_num_samples)
    return dataset


def _concat_sample_count(sample_count: int,
                         mode: ConcatMode,
                         max_length: Optional[int] = None) -> int:
    emp_chars_per_sample = 2163  # measured for val split
    est_tokens_per_samples = 540  # using est_char_per_token = 4

    if max_length is None and mode != ConcatMode.NO_CONCAT:
        raise ValueError("max_length can't be None when concatenating")

    if mode == ConcatMode.NO_CONCAT:
        return sample_count
    elif mode == ConcatMode.CONCAT_TEXT:
        total_char = sample_count * emp_chars_per_sample
        return total_char // max_length
    elif mode == ConcatMode.CONCAT_TOKENS:
        total_tokens = sample_count * est_tokens_per_samples
        return total_tokens // max_length


def _get_kwargs(args):
    split_samples = (364868892, 327680, 364608)

    if hasattr(args, 'concat_tokens') and args.concat_tokens is not None:
        mode = ConcatMode.CONCAT_TOKENS
        max_length = args.concat_tokens
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        # we will enforce length, so suppress warnings about sequences too long for the model
        tokenizer.model_max_length = int(1e30)
        bos_text = args.bos_text
        eos_text = args.eos_text
        # concatenating makes us lose timestamp and url
        columns = {'tokens': 'bytes'}
        split_samples = [
            _concat_sample_count(ct, mode, max_length) for ct in split_samples
        ]

        if bos_text + eos_text == '':
            test_tokens = tokenizer('test')
            if test_tokens['input_ids'][
                    0] != tokenizer.bos_token_id and test_tokens['input_ids'][
                        -1] != tokenizer.eos_token_id:
                tok_error_msg = 'This tokenizer does not insert an EOS nor BOS token.'
                tok_error_msg += 'Concatenating with this tokenizer will result in sequences being '
                tok_error_msg += 'attached without a separating token. Please use another tokenizer, '
                tok_error_msg += 'such as facebook/opt-125m, or specify EOS/BOS text with e.g. '
                tok_error_msg += '--bos_text=<|endoftext|>.'
                raise ValueError(tok_error_msg)

    elif hasattr(args, 'concat_text') and args.concat_text is not None:
        mode = ConcatMode.CONCAT_TEXT
        max_length = args.concat_text
        tokenizer = None
        bos_text = args.bos_text
        eos_text = args.eos_text
        # concatenating makes us lose timestamp and url
        columns = {'text': 'str'}
        split_samples = [
            _concat_sample_count(ct, mode, max_length) for ct in split_samples
        ]

    else:
        mode = ConcatMode.NO_CONCAT
        max_length = None
        tokenizer = None
        bos_text = None
        eos_text = None
        columns = {'text': 'str', 'timestamp': 'str', 'url': 'str'}

    return mode, max_length, tokenizer, bos_text, eos_text, columns, split_samples


def main(args: Namespace) -> None:
    """Main: create C4 streaming dataset.

    Args:
        args (Namespace): Commandline arguments.
    """
    splits = ('train', 'train', 'validation')
    split_names = ('train', 'train_small', 'val')

    mode, max_length, tokenizer, bos_text, eos_text, columns, split_samples = _get_kwargs(
        args)

    for (split, split_new_name,
         expected_num_samples) in zip(splits, split_names, split_samples):
        # Only generate the splits requested
        if split_new_name not in args.splits:
            continue

        # Get samples
        print(f'Downloading and formatting {split_new_name}...')
        dataset = build_hf_c4_dataset(split=split,
                                      mode=mode,
                                      max_length=max_length,
                                      bos_text=bos_text,
                                      eos_text=eos_text,
                                      tokenizer=tokenizer,
                                      expected_num_samples=expected_num_samples)
        samples = dataset.generate_samples()

        # Write samples
        print(f'Converting {split_new_name} to MDS format...')
        with MDSWriter(dirname=os.path.join(args.out_root, split_new_name),
                       columns=columns,
                       compression=args.compression) as out:
            for sample in tqdm(samples,
                               desc=split_new_name,
                               total=expected_num_samples):
                out.write(sample)


if __name__ == '__main__':
    main(parse_args())
