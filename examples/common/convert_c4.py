# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

"""C4 streaming dataset conversion scripts."""
import os
import platform
from argparse import Action, ArgumentError, ArgumentParser, Namespace
from enum import Enum
from typing import Dict, Iterable, Optional

import datasets as hf_datasets
from streaming import MDSWriter
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase


class ConcatMode(Enum):
    NO_CONCAT = 0
    CONCAT_TEXT = 1
    CONCAT_TOKENS = 2


def parse_args() -> Namespace:
    """Parse commandline arguments."""
    parser = ArgumentParser()
    parser.add_argument('--out_root', type=str, required=True)
    parser.add_argument('--compression', type=str, default=None)
    parser.add_argument('--splits',
                        nargs='+',
                        default=['train', 'train_small', 'val'])

    subparsers = parser.add_subparsers(required=False, help='Concatenation utilities')
    concat_parser = subparsers.add_parser('concat', help='Pre-concatenate before converting to MDS')

    group = concat_parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--text', type=int, help='Number of characters to concatenate to')
    group.add_argument('--tokens', type=int, help='Convert text to tokens and concatenate up to this many tokens')

    concat_parser.add_argument('--tokenizer', type=str, required=False, default=None)
    concat_parser.add_argument('--sep_text', type=str, required=False, default=None)

    parsed = parser.parse_args()

    if hasattr(parsed, "text") or hasattr(parsed, "tokens"):
        # Make sure we have needed concat options
        if (parsed.text is not None and isinstance(parsed.text, int) and
            parsed.sep_text is None):
                parser.error('When setting concat --text, you must specify a '
                        '--sep_text with which to separate concatenated sequences')
        if (parsed.tokens is not None and isinstance(parsed.tokens, int) and
            parsed.tokenizer is None):
                parser.error('When setting concat --tokens, you must specify a '
                        '--tokenizer')
    return parsed



def build_hf_c4_dataset(split: str,
                        mode: ConcatMode,
                        tokenizer: Optional[PreTrainedTokenizerBase]) -> IterableDataset:
    """Collect the samples for this dataset split.

    Args:
        split (str): Split name.

    Returns:
        An IterableDataset.
    """

    class ShardedC4(IterableDataset):

        def __init__(self):
            self.dataset = hf_datasets.load_dataset(path='c4',
                                                    name='en',
                                                    split=split,
                                                    streaming=True)

        def num_shards(self):
            it = self.dataset._ex_iterable  # type: ignore
            return len(it.kwargs['filepaths'])  # type: ignore

        def __iter__(self):
            worker_info = get_worker_info()
            if worker_info:
                num_workers = worker_info.num_workers
                worker_id = worker_info.id
                it = self.dataset._ex_iterable  # type: ignore
                shards = it.kwargs['filepaths']  # type: ignore
                assert len(shards) % num_workers == 0
                it.kwargs['filepaths'] = shards[  # type: ignore
                    worker_id::num_workers]
            return iter(self.dataset)

        def generate_samples(self,
                             expected_num_samples: int) -> Iterable[Dict[str, bytes]]:
            """Generator over each dataset sample.

            Args:
                samples (IterableDataset): An iterable dataset that is multi-worker compatible

            Yields:
                Sample dicts.
            """
            # Multiple workers is only supported on linux machines
            if 'linux' in platform.platform().lower():
                num_workers = min(64, self.dataset.num_shards())  # type: ignore
            else:
                num_workers = 0
            batch_size = 512
            # If using multiple workers, configure each worker to prefetch as many samples as it can, up to
            # the aggregate device batch size
            # If not using workers, the torch DataLoader expects the default value for prefetch_factor,
            # which non-intuitively must be 2.
            prefetch_factor = max(1, 2 * batch_size //
                                num_workers) if num_workers > 0 else 2

            loader = DataLoader(
                dataset=self.dataset,
                sampler=None,
                batch_size=batch_size,
                num_workers=num_workers,
                prefetch_factor=prefetch_factor,
            )

            n_samples = 0
            for batch in loader:
                keys = list(batch.keys())
                current_bs = len(batch[keys[0]])
                for idx in range(current_bs):
                    yield {
                        key: batch_values[idx].encode('utf-8')
                        for key, batch_values in batch.items()
                    }
                    n_samples += 1
                    if n_samples == expected_num_samples:
                        return


        class ConcatC4(IterableDataset):

        def __init__(self, max_length: int, batch_size: int = 1000):
            self.dataset = hf_datasets.load_dataset(path='c4',
                                                    name='en',
                                                    split=split,
                                                    streaming=True)
            self.max_length = max_length
            self.batch_size = batch_size
            self._tokenize_concat()

        def _concat(self):
            """concatenate samples ahead of time

            we could also do this on the fly in generate_samples,
            and that would be more data efficient (we throw out the overflow each batch,
            whereas tokenizing as needed would only throw out overflow once at the end).
            However, doing it this way lets us use all cores, which is
            a massive speedup on larger VMs
            """

            def preprocess(examples: dict):
                concatenated = []
                index = 0
                while index < len(examples['text']):
                    concat = []
                    while len(concat) < self.max_length:
                        if index >= len(examples['text']):
                            break
                        concat += examples['text'][index]
                        index += 1
                    concatenated.append(concat[:self.max_length])
                if len(concatenated[-1]) < self.max_length:
                    # throw away remainder
                    del concatenated[-1]
                for c in concatenated:
                    print(len(c))

                return {'text': concatenated}

            self.dataset.map(
                preprocess,
                batched=True,
                # Arrow errors if you don't remove all other columns
                remove_columns=self.dataset.columns,
                load_from_cache_file=True,
            )

        def __iter__(self):
            return iter(self.dataset)

        def generate_samples(self, expected_num_samples: Optional[int] = None) -> Iterable[Dict[str, bytes]]:
            """Generator over each dataset sample.

            Args:
                samples (IterableDataset): An iterable dataset that is multi-worker compatible

            Yields:
                Sample dicts.
            """
            # Multiple workers is only supported on linux machines
            if 'linux' in platform.platform().lower():
                num_workers = 64  # make configurable
            else:
                num_workers = 0
            batch_size = 512
            # If using multiple workers, configure each worker to prefetch as many samples as it can, up to
            # the aggregate device batch size
            # If not using workers, the torch DataLoader expects the default value for prefetch_factor,
            # which non-intuitively must be 2.
            prefetch_factor = max(1, 2 * batch_size //
                                num_workers) if num_workers > 0 else 2

            loader = DataLoader(
                dataset=self.dataset,
                sampler=None,
                batch_size=batch_size,
                num_workers=num_workers,
                prefetch_factor=prefetch_factor,
            )

            for batch in loader:
                keys = list(batch.keys())
                current_bs = len(batch[keys[0]])
                for idx in range(current_bs):
                    yield {
                        key: batch_values[idx]
                        for key, batch_values in batch.items()
                    }

    class TokenizedC4(IterableDataset):

        def __init__(self, tokenizer: PreTrainedTokenizerBase, max_length: int, batch_size: int = 1000):
            self.dataset = hf_datasets.load_dataset(path='c4',
                                                    name='en',
                                                    split=split,
                                                    streaming=True)
            self.tokenizer = tokenizer
            self.max_length = max_length
            self.batch_size = batch_size
            self._tokenize_concat()

        def _tokenize_concat(self):
            """tokenize and concatenate samples ahead of time

            we could also do this on the fly in generate_samples,
            and that would be more data efficient (we throw out the overflow each batch,
            whereas tokenizing as needed would only throw out overflow once at the end).
            However, doing it this way lets us use all cores, which is
            a massive speedup on larger VMs
            """

            def preprocess(examples: dict):
                batch = self.tokenizer(examples['text'],
                                    padding=False,
                                    truncation=False)
                concatenated = []
                index = 0
                while index < len(batch['input_ids']):
                    concat = []
                    while len(concat) < self.max_length:
                        if index >= len(batch['input_ids']):
                            break
                        concat += batch['input_ids'][index]
                        index += 1
                    concatenated.append(concat[:self.max_length])
                if len(concatenated[-1]) < self.max_length:
                    # throw away remainder
                    del concatenated[-1]
                for c in concatenated:
                    print(len(c))
                return {'tokens': concatenated}

            self.dataset.map(
                preprocess,
                batched=True,
                # Arrow errors if you don't remove all other columns
                remove_columns=self.dataset.columns,
                load_from_cache_file=True,
            )

        def __iter__(self):
            return iter(self.dataset)

        def generate_samples(self, expected_num_samples: Optional[int] = None) -> Iterable[Dict[str, bytes]]:
            """Generator over each dataset sample.

            Args:
                samples (IterableDataset): An iterable dataset that is multi-worker compatible

            Yields:
                Sample dicts.
            """
            # Multiple workers is only supported on linux machines
            if 'linux' in platform.platform().lower():
                num_workers = 64  # make configurable
            else:
                num_workers = 0
            batch_size = 512
            # If using multiple workers, configure each worker to prefetch as many samples as it can, up to
            # the aggregate device batch size
            # If not using workers, the torch DataLoader expects the default value for prefetch_factor,
            # which non-intuitively must be 2.
            prefetch_factor = max(1, 2 * batch_size //
                                num_workers) if num_workers > 0 else 2

            loader = DataLoader(
                dataset=self.dataset,
                sampler=None,
                batch_size=batch_size,
                num_workers=num_workers,
                prefetch_factor=prefetch_factor,
            )

            for batch in loader:
                keys = list(batch.keys())
                current_bs = len(batch[keys[0]])
                for idx in range(current_bs):
                    yield {
                        key: batch_values[idx]
                        for key, batch_values in batch.items()
                    }


    if mode == ConcatMode.NO_CONCAT:
        dataset = ShardedC4()
    elif mode == ConcatMode.CONCAT_TEXT:
        dataset = ConcatC4(max_length=max_length)
    else:
        dataset = TokenizedC4(tokenizer=tokenizer, max_length=max_length)
    return dataset


def main(args: Namespace) -> None:
    """Main: create C4 streaming dataset.

    Args:
        args (Namespace): Commandline arguments.
    """
    splits = ('train', 'train', 'validation')
    split_names = ('train', 'train_small', 'val')
    split_sizes_raw = (364868892, 327680, 364608)

    if args.tokens is not None:
        mode = ConcatMode.CONCAT_TOKENS
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        # concatenating makes us lose timestamp and url
        columns = {'tokens': 'List[int]'}
        split_samples = (None, None, None)

        test_tokens = tokenizer("test")

        if ts['input_ids'][0] != tokenizer.bos_token_id and ts['input_ids'][-1] != tokenizer.eos_token_id:
            warning_msg = 'This tokenizer does not insert an EOS nor BOS token.'
            warning_msg += 'Concatenating with this tokenizer will result in sequences being '
            warning_msg += 'attached without a separating token. Please use another tokenizer, '
            warning_msg += 'such as facebook/opt-125m'
            raise ValueError(warning_msg)

    elif args.text is not None:
        mode = ConcatMode.CONCAT_TEXT
        tokenizer = None
        # concatenating makes us lose timestamp and url
        columns = {'text': 'str'}
        split_samples = tuple([size // args.text for size in split_sizes_raw])
    else:
        mode = ConcatMode.NO_CONCAT
        tokenizer = None
        columns = {'text': 'str', 'timestamp': 'str', 'url': 'str'}
        split_samples = split_sizes_raw

    for (split, split_new_name, expected_num_samples) in zip(
        splits,
        split_names,
        split_samples
    ):
        # Only generate the splits requested
        if split_new_name not in args.splits:
            continue

        # Get samples
        dataset = build_hf_c4_dataset(split=split, mode=mode, tokenizer=tokenizer)
        samples = dataset.generate_samples(
            expected_num_samples=expected_num_samples
        )

        # Write samples
        with MDSWriter(dirname=os.path.join(args.out_root, split_new_name),
                       columns=columns,
                       compression=args.compression) as out:
            for sample in tqdm(samples,
                               desc=split_new_name,
                               total=expected_num_samples):
                out.write(sample)


if __name__ == '__main__':
    print(parse_args())
    # main(parse_args())
