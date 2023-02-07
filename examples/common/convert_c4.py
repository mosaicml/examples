# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

"""C4 streaming dataset conversion scripts test with small subset."""
import io
import os
import platform
from argparse import Action, ArgumentError, ArgumentParser, Namespace
from enum import Enum
from multiprocessing import cpu_count
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


class ShardedC4(IterableDataset):
    def __init__(self):
        self.dataset = hf_datasets.load_dataset(path='c4',
                                                name='en',
                                                split='validation',
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


class ProcessedC4(IterableDataset):
    def __init__(self, max_length: int, batch_size: int = 1000,
                 num_proc: Optional[int] = None,
                 expected_num_samples: Optional[int] = None):
        self.dataset = hf_datasets.load_dataset('stas/c4-en-10k', #path='c4',
                                                # name='en',
                                                # split='validation',
                                                split='train',
                                                streaming=False)
        # next 2 are needed for streaming
        # self.dataset._resolve_features
        if not hasattr(self.dataset, "column_names"):
            self.dataset.column_names = list(self.dataset.features.keys())
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_proc = cpu_count() - 1 if num_proc is None else num_proc
        self.expected_num_samples = expected_num_samples
        self._preprocess()
        self._build_loader()

    def _preprocess(self):
        raise NotImplementedError()

    def __iter__(self):
        return iter(self.dataset)

    def _build_loader(self):
        num_workers = self.num_proc

        if self.expected_num_samples and self.expected_num_samples < 512:
            batch_size = self.expected_num_samples
        else:
            batch_size = 512
        # If using multiple workers, configure each worker to prefetch as many samples as it can, up to
        # the aggregate device batch size
        # If not using workers, the torch DataLoader expects the default value for prefetch_factor,
        # which non-intuitively must be 2.
        prefetch_factor = max(1, 2 * batch_size //
                            num_workers) if num_workers > 0 else 2

        self.loader = DataLoader(
            dataset=self.dataset,
            sampler=None,
            batch_size=batch_size,
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
        for batch in self.loader:
            keys = list(batch.keys())
            current_bs = len(batch[keys[0]])
            for idx in range(current_bs):
                yield {
                    key: batch_values[idx].encode('utf-8')
                    for key, batch_values in batch.items()
                }

class ConcatC4(ProcessedC4):
    def __init__(self, sep_text: str, *args, **kwargs):
        self.sep_text = sep_text
        super().__init__(*args, **kwargs)

    def _preprocess(self):
        """concatenate samples ahead of time

        we could also do this on the fly in generate_samples,
        and that would be more data efficient (we throw out the overflow each batch,
        whereas tokenizing as needed would only throw out overflow once at the end).
        However, doing it this way lets us use all cores, which is
        a massive speedup on larger VMs
        """

        def concat_text(examples: dict):
            concatenated = []
            index = 0
            while index < len(examples['text']):
                concat = []
                while len(concat) < self.max_length:
                    if index >= len(examples['text']):
                        break
                    concat += self.sep_text
                    concat += examples['text'][index]
                    index += 1
                concatenated.append(concat[:self.max_length])
            if len(concatenated[-1]) < self.max_length:
                # throw away remainder
                del concatenated[-1]

            return {'text': ["".join(c) for c in concatenated]}

        self.dataset = self.dataset.map(
            concat_text,
            batched=True,
            num_proc=self.num_proc,
            # Arrow errors if you don't remove all other columns
            remove_columns=self.dataset.column_names,
        )


class TokenizedC4(ProcessedC4):
    """A tokenized, concatenated, iterable dataset

    To use data created by this class and written to MDS format:

    ```python
    import io
    import torch
    from streaming.base import StreamingDataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("your/tokenizer)
    ds = StreamingDataset(local="mds-data-folder", split="val")

    tokens = torch.load(io.BytesIO(ds[1]['tokens']))
    print(tokenizer.decode(tokens))
    ```"""

    def __init__(self, tokenizer: PreTrainedTokenizerBase, *args, **kwargs):
        try:
            import torch
        except ImportError as e:
            print('torch is required to pre-tokenize')
            raise e

        self.tokenizer = tokenizer
        super().__init__(*args, **kwargs)

    def _preprocess(self):
        """tokenize and concatenate samples ahead of time

        we could also do this on the fly in generate_samples,
        and that would be more data efficient (we throw out the overflow each batch,
        whereas tokenizing as needed would only throw out overflow once at the end).
        However, doing it this way lets us use all cores, which is
        a massive speedup on larger VMs
        """

        def tokenize_concat(examples: dict):
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
            return {'tokens': concatenated}

        self.dataset = self.dataset.map(
            tokenize_concat,
            batched=True,
            num_proc=self.num_proc,
            # Arrow errors if you don't remove all other columns
            remove_columns=self.dataset.column_names,
            # for development
            load_from_cache_file=False,
        )

    def generate_samples(self, expected_num_samples: Optional[int] = None) -> Iterable[Dict[str, bytes]]:
        """Generator over each dataset sample.

        Args:
            samples (IterableDataset): An iterable dataset that is multi-worker compatible

        Yields:
            Sample dicts.
        """
        import torch

        # TODO: figure out why rows and columns are transposed
        # if you step through with a debugger, the batch is definitely transposed here
        # but then swapping rows and columns back here does not fix how garbled this data gets
        for batch in self.loader:
            transposed = torch.stack(tuple(batch['tokens'])).transpose(0,1)
            for tokens in transposed:
                tok_bytes = tokens.byte()
                buffer = io.BytesIO()
                torch.save(tok_bytes, buffer)
                yield {
                    'tokens': buffer.getvalue()
                }


class NonexistentDir(Action):
    def __call__(self, parser, namespace, values, option_string=None):
        prospective_dir = values
        if os.path.isdir(prospective_dir) and len(os.listdir(prospective_dir)) > 0:
            raise ArgumentError(self, f'--out_root={prospective_dir} must be empty')
        else:
            setattr(namespace, "out_root", prospective_dir)


def parse_args() -> Namespace:
    """Parse commandline arguments."""
    parser = ArgumentParser()
    parser.add_argument('--out_root', type=str, required=True, action=NonexistentDir)
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
                        max_length: Optional[int] = None,
                        sep_text: Optional[str] = None,
                        tokenizer: Optional[PreTrainedTokenizerBase] = None,
                        expected_num_samples: Optional[int] = None) -> IterableDataset:
    """Collect the samples for this dataset split.

    Args:
        split (str): Split name.
        mode (ConcatMode): NO_CONCAT, CONCAT_TEXT, or CONCAT_TOKENS
        sep_text (str): if mode is CONCAT_TEXT, what string to use to separate samples
        tokenizer (PreTrainedTokenizerBase): if mode is CONCAT_TOKENS, the tokenizer to use
        expected_num_samples (int): _not yet implemented_ once we have generated this many samples, quit

    Returns:
        An IterableDataset.
    """

    if mode == ConcatMode.NO_CONCAT:
        dataset = ShardedC4()
    elif mode == ConcatMode.CONCAT_TEXT:
        dataset = ConcatC4(max_length=max_length, sep_text=sep_text, expected_num_samples=expected_num_samples)
    else:
        dataset = TokenizedC4(tokenizer=tokenizer, max_length=max_length, expected_num_samples=expected_num_samples)
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
        max_length = args.tokens
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        sep_text = None
        # concatenating makes us lose timestamp and url
        columns = {'tokens': 'bytes'}
        split_samples = (None, None, None)

        test_tokens = tokenizer("test")

        if test_tokens['input_ids'][0] != tokenizer.bos_token_id and test_tokens['input_ids'][-1] != tokenizer.eos_token_id:
            warning_msg = 'This tokenizer does not insert an EOS nor BOS token.'
            warning_msg += 'Concatenating with this tokenizer will result in sequences being '
            warning_msg += 'attached without a separating token. Please use another tokenizer, '
            warning_msg += 'such as facebook/opt-125m'
            raise ValueError(warning_msg)

    elif args.text is not None:
        mode = ConcatMode.CONCAT_TEXT
        max_length = args.text
        tokenizer = None
        sep_text = args.sep_text
        # concatenating makes us lose timestamp and url
        columns = {'text': 'str'}
        split_samples = tuple([size // args.text for size in split_sizes_raw])
    else:
        mode = ConcatMode.NO_CONCAT
        max_length = None
        tokenizer = None
        sep_text = None
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
        print('Downloading and formatting dataset...')
        dataset = build_hf_c4_dataset(split=split,
                                      mode=mode,
                                      max_length=max_length,
                                      sep_text=sep_text,
                                      tokenizer=tokenizer,
                                      expected_num_samples=expected_num_samples)
        samples = dataset.generate_samples()
        # Somehow during CONCAT_TOKENS the data is getting garbled...

        # Write samples
        print('Converting to MDS format...')
        with MDSWriter(dirname=os.path.join(args.out_root, split_new_name),
                       columns=columns,
                       compression=args.compression) as out:
            for sample in tqdm(samples,
                               desc=split_new_name,
                               total=expected_num_samples):
                out.write(sample)


if __name__ == '__main__':
    main(parse_args())
