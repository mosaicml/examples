# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

"""C4 streaming dataset conversion scripts."""
import os
import platform
import warnings
from argparse import Action, ArgumentError, ArgumentParser, Namespace
from enum import Enum
from textwrap import dedent
from typing import Dict, Iterable, Optional

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

    group = parser.add_mutually_exclusive_group(required=False)
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


class NoConcatC4(IterableDataset):
    """The class is a wrapper around HuggingFace's streaming dataset to allow us to modify the iterator over samples

    We override __iter__ to return dicts of {'text': bytes}
    """

    def __init__(self, split: str):
        self.hf_dataset = hf_datasets.load_dataset(path='c4',
                                                   name='en',
                                                   split=split,
                                                   streaming=True)

    def __iter__(self):
        for sample in self.hf_dataset:
            yield {'text': sample['text'].encode('utf-8')}


# class ConcatC4(ShardedC4):

#     def __init__(self, bos_text: str, eos_text: str, *args, **kwargs):
#         self.bos_text = bos_text
#         self.eos_text = eos_text
#         super().__init__(*args, **kwargs)

#     def generate_samples(self) -> Iterable[Dict[str, bytes]]:
#         """Generator over each dataset sample.

#         Args:
#             samples (IterableDataset): An iterable dataset that is multi-worker compatible

#         Yields:
#             Sample dicts.
#         """
#         n_samples = 0
#         buffer = ''
#         while True:
#             for batch in self.loader:
#                 for sample in batch['text']:
#                     n_samples += 1
#                     buffer = buffer + self.bos_text + sample + self.eos_text
#                     while len(buffer) >= self.max_length:
#                         concat_sample = buffer[:self.max_length]
#                         buffer = buffer[self.max_length:]

#                         yield {'text': concat_sample.encode('utf-8')}

#                         if n_samples >= self.expected_num_samples:
#                             return

# class TokenizedC4(ShardedC4):
#     """A tokenized, concatenated, iterable dataset.

#     To use data created by this class and written to MDS format:

#     ```python
#         import torch
#         from streaming.base import StreamingDataset
#         from transformers import AutoTokenizer

#         tokenizer = AutoTokenizer.from_pretrained('your/tokenizer')
#         ds = StreamingDataset(local='mds-data-folder', split='val')

#         # note, you need to copy the numpy array because the original is non-writeable
#         # and torch does not support non-writeable tensors, so you get a scary warning and
#         # if you do try to write to the tensor you get undefined behavior
#         tokens = torch.from_numpy(np.frombuffer(ds[0]['tokens'], dtype=np.int64).copy())
#         print(tokenizer.decode(tokens))
#     ```
#     """

#     def __init__(self, tokenizer: PreTrainedTokenizerBase, bos_text: str,
#                  eos_text: str, *args, **kwargs):
#         self.tokenizer = tokenizer
#         self.bos_text = bos_text
#         self.eos_text = eos_text
#         super().__init__(*args, **kwargs)

#     def generate_samples(self) -> Iterable[Dict[str, bytes]]:
#         """Generator over each dataset sample.

#         Args:
#             samples (IterableDataset): An iterable dataset that is multi-worker compatible

#         Yields:
#             Sample dicts.
#         """
#         bos_tokens = self.tokenizer(self.bos_text,
#                                     truncation=False,
#                                     padding=False,
#                                     add_special_tokens=False)['input_ids']
#         if len(bos_tokens) > 1:
#             warnings.warn(
#                 f'You specified --concat_tokens with --bos_text, but your BOS text is not tokenizing to one token\
#                 , instead we got {bos_tokens}. Quit if this was in error.')

#         eos_tokens = self.tokenizer(self.eos_text,
#                                     truncation=False,
#                                     padding=False,
#                                     add_special_tokens=False)['input_ids']
#         if len(eos_tokens) > 1:
#             warnings.warn(
#                 f'You specified --concat_tokens with --eos_text, but your EOS text is not tokenizing to one token\
#                 , instead we got {eos_tokens}. Quit if this was in error.')

#         n_samples = 0
#         buffer = []
#         while True:
#             for batch in self.loader:
#                 encoded = self.tokenizer(batch['text'],
#                                          truncation=False,
#                                          padding=False)
#                 for iids in encoded['input_ids']:
#                     n_samples += 1
#                     buffer = buffer + bos_tokens + iids + eos_tokens
#                     while len(buffer) >= self.max_length:
#                         concat_sample = buffer[:self.max_length]
#                         buffer = buffer[self.max_length:]

#                         yield {
#                             # convert to bytes to store in MDS binary format
#                             'tokens': np.asarray(concat_sample).tobytes()
#                         }
#                         if n_samples >= self.expected_num_samples:
#                             return


def build_hf_c4_dataset(
        split: str, mode: ConcatMode, max_length: Optional[int],
        bos_text: Optional[str], eos_text: Optional[str],
        tokenizer: Optional[PreTrainedTokenizerBase]) -> IterableDataset:
    """Collect the samples for this dataset split.

    Args:
        split (str): Split name.
        mode (ConcatMode): NO_CONCAT, CONCAT_TEXT, or CONCAT_TOKENS
        bos_text (str): text to insert at the beginning of each sequence
        eos_text (str): text to insert at the end of each sequence
        tokenizer (PreTrainedTokenizerBase): if mode is CONCAT_TOKENS, the tokenizer to use

    Returns:
        An IterableDataset.
    """
    if mode == ConcatMode.NO_CONCAT:
        dataset = NoConcatC4(split)
    # elif mode == ConcatMode.CONCAT_TEXT:
    #     dataset = ConcatC4(split=split,
    #                        max_length=max_length,
    #                        bos_text=bos_text,
    #                        eos_text=eos_text,
    #                        expected_num_samples=expected_num_samples)
    # else:
    #     dataset = TokenizedC4(split=split,
    #                           tokenizer=tokenizer,
    #                           max_length=max_length,
    #                           bos_text=bos_text,
    #                           eos_text=eos_text,
    #                           expected_num_samples=expected_num_samples)
    return dataset


def _get_kwargs(args):
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

        if bos_text + eos_text == '':
            test_tokens = tokenizer('test')
            if test_tokens['input_ids'][
                    0] != tokenizer.bos_token_id and test_tokens['input_ids'][
                        -1] != tokenizer.eos_token_id:
                tok_error_msg = 'This tokenizer does not insert an EOS nor BOS token. '
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

    else:
        mode = ConcatMode.NO_CONCAT
        max_length = None
        tokenizer = None
        bos_text = None
        eos_text = None
        columns = {'text': 'str'}

    return mode, max_length, tokenizer, bos_text, eos_text, columns


def _est_progress_denominator(total_samples: int, mode: ConcatMode,
                              max_length: int):
    emp_chars_per_sample = 2163  # measured for val split
    est_tokens_per_samples = 540  # using est_char_per_token = 4
    if mode == ConcatMode.NO_CONCAT:
        return total_samples
    elif mode == ConcatMode.CONCAT_TEXT:
        return total_samples * emp_chars_per_sample // max_length
    elif mode == ConcatMode.CONCAT_TOKENS:
        return total_samples * est_tokens_per_samples // max_length


def build_dataloader(dataset, batch_size) -> None:
    # Multiple workers is only supported on linux machines
    if 'linux' in platform.platform().lower():
        num_workers = min(64, dataset.hf_dataset.n_shards)  # type: ignore
    else:
        num_workers = 0

    # If using multiple workers, configure each worker to prefetch as many samples as it can, up to
    # the aggregate device batch size
    # If not using workers, the torch DataLoader expects the default value for prefetch_factor,
    # which non-intuitively must be 2.
    prefetch_factor = max(1, 2 * batch_size //
                          num_workers) if num_workers > 0 else 2

    return DataLoader(
        dataset=dataset,
        sampler=None,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )


def generate_samples(loader,
                     truncate_num_samples=None) -> Iterable[Dict[str, bytes]]:
    """Generator over each dataset sample.

    Args:
        samples (IterableDataset): An iterable dataset that is multi-worker compatible

    Yields:
        Sample dicts.
    """
    n_samples = 0
    for batch in loader:
        keys = list(batch.keys())
        current_bs = len(batch[keys[0]])
        for idx in range(current_bs):
            if truncate_num_samples is not None and n_samples == truncate_num_samples:
                return
            n_samples += 1
            yield {k: v[idx] for k, v in batch.items()}


def main(args: Namespace) -> None:
    """Main: create C4 streaming dataset.

    Args:
        args (Namespace): Commandline arguments.
    """
    hf_splits = ('train', 'train', 'validation')
    folder_splits = ('train', 'train_small', 'val')
    expected_counts = (364868892, 655360, 364608)
    truncate_counts = (None, 655360, None)

    mode, max_length, tokenizer, bos_text, eos_text, columns = _get_kwargs(args)

    for (hf_split, folder_split, expected_num_samples,
         truncate_num_samples) in zip(hf_splits, folder_splits, expected_counts,
                                      truncate_counts):
        # Only generate the splits requested
        if folder_split not in args.splits:
            continue

        # Get samples
        print(f'Downloading and formatting {folder_split}...')
        dataset = build_hf_c4_dataset(split=hf_split,
                                      mode=mode,
                                      max_length=max_length,
                                      bos_text=bos_text,
                                      eos_text=eos_text,
                                      tokenizer=tokenizer)
        loader = build_dataloader(dataset=dataset, batch_size=512)
        samples = generate_samples(loader,
                                   truncate_num_samples=truncate_num_samples)

        denominator = truncate_num_samples if truncate_num_samples is not None else _est_progress_denominator(
            expected_num_samples, mode, max_length)

        # Write samples
        print(f'Converting {folder_split} to MDS format...')
        with MDSWriter(dirname=os.path.join(args.out_root, folder_split),
                       columns=columns,
                       compression=args.compression) as out:
            for sample in tqdm(samples, desc=folder_split, total=denominator):
                out.write(sample)


if __name__ == '__main__':
    main(parse_args())
