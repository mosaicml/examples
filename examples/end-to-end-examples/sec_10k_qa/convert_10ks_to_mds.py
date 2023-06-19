# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
from argparse import ArgumentParser, Namespace
from typing import Any, Dict, Iterable, List, Optional

import datasets
from composer.utils import (ObjectStore, maybe_create_object_store_from_uri,
                            parse_uri)
# type ignore because we don't want to add foundry to the local dependencies
from llmfoundry.data import ConcatTokensDataset  # type: ignore
from streaming import MDSWriter
from torch.utils.data import DataLoader, Dataset, get_worker_info
from tqdm import tqdm
from transformers import AutoTokenizer


def parse_args() -> Namespace:
    """Parse commandline arguments."""
    parser = ArgumentParser(
        description=
        'Convert dataset into MDS format, optionally concatenating and tokenizing'
    )
    parser.add_argument(
        '--max_workers',
        type=int,
        default=64,
        help='The maximum number of workers to use for MDS writing')
    parser.add_argument('--out_root',
                        type=str,
                        required=True,
                        help='The folder to write output to')
    parser.add_argument('--in_root',
                        type=str,
                        required=True,
                        help='The folder to read input from')
    parser.add_argument(
        '--dataset_subset',
        type=str,
        required=False,
        default='small_full',
        help=
        'The subset of the dataset to use. Options are large_full and small_full.'
    )
    parser.add_argument('--compression',
                        type=str,
                        default='zstd',
                        help='The compression algorithm to use for MDS writing')

    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        '--concat_tokens',
        type=int,
        help='Convert text to tokens and concatenate up to this many tokens')

    parser.add_argument('--tokenizer',
                        type=str,
                        required=False,
                        default=None,
                        help='The name of the tokenizer to use')
    parser.add_argument(
        '--bos_text',
        type=str,
        required=False,
        default=None,
        help=
        'The text to prepend to each example to separate concatenated examples')
    parser.add_argument(
        '--eos_text',
        type=str,
        required=False,
        default=None,
        help=
        'The text to append to each example to separate concatenated examples')
    parser.add_argument(
        '--no_wrap',
        default=False,
        action='store_true',
        help=
        'Whether to let text examples wrap across multiple training examples')

    parsed = parser.parse_args()

    if os.path.isdir(parsed.out_root) and len(
            set(os.listdir(parsed.out_root)).intersection(set(
                parsed.splits))) > 0:
        raise ValueError(
            f'--out_root={parsed.out_root} contains {os.listdir(parsed.out_root)} which cannot overlap with the requested splits {parsed.splits}.'
        )

    # Make sure we have needed concat options
    if (parsed.concat_tokens is not None and
            isinstance(parsed.concat_tokens, int) and parsed.tokenizer is None):
        parser.error(
            'When setting --concat_tokens, you must specify a --tokenizer')

    # now that we have validated them, change BOS/EOS to strings
    if parsed.bos_text is None:
        parsed.bos_text = ''
    if parsed.eos_text is None:
        parsed.eos_text = ''
    return parsed


def build_dataloader(dataset: Dataset, batch_size: int) -> DataLoader:
    return DataLoader(
        dataset=dataset,
        sampler=None,
        batch_size=batch_size,
        num_workers=8,
        prefetch_factor=2,
    )


def generate_samples(
        loader: DataLoader,
        truncate_num_samples: Optional[int] = None
) -> Iterable[Dict[str, bytes]]:
    """Generator over samples of a dataloader.

    Args:
       loader (DataLoader): A dataloader emitting batches like {key: [sample0_bytes, sample1_bytes, sample2_bytes, ...]}
       truncate_num_samples (Optional[int]): An optional # of samples to stop at.

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


class DownloadingIterable:

    def __init__(
        self,
        identifiers: List[str],
        input_folder_prefix: str,
        output_folder: str,
        object_store: ObjectStore,
    ):
        """Iterable that downloads files from an object store before yielding.

        text samples.

        Args:
            identifiers (List[str]): List of identifiers (<doc id>|||<ticker>|||<report_date>) to iterate over
            input_folder_prefix (str): Object store prefix to download from
            output_folder (str): Local folder to write downloaded files to
            object_store (ObjectStore): Object store to download from
        """
        self.identifiers = [
            identifier.split('|||') for identifier in identifiers
        ]
        self.object_store = object_store
        self.input_folder_prefix = input_folder_prefix
        self.output_folder = output_folder

    def __iter__(self):
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1
        identifiers_shard = self.identifiers[worker_id::num_workers]
        print(f'Worker {worker_id} processing {len(identifiers_shard)} files')
        for (_, ticker, report_date) in identifiers_shard:
            os.makedirs(os.path.join(self.output_folder, ticker), exist_ok=True)

            year = report_date.split('-')[0]
            self.object_store.download_object(
                os.path.join(self.input_folder_prefix, ticker,
                             f'sec_{year}_txt.txt'),
                os.path.join(self.output_folder, ticker, f'sec_{year}.txt'),
            )

            with open(
                    os.path.join(self.output_folder, ticker,
                                 f'sec_{year}.txt')) as _txt_file:
                txt = _txt_file.read()

            yield {'text': txt}


def main(
    tokenizer_name: str,
    output_folder: str,
    input_folder: str,
    dataset_subset: str,
    concat_tokens: int,
    eos_text: str,
    bos_text: str,
    no_wrap: bool,
    max_workers: int,
    compression: str,
) -> None:
    """Convert the 10-K dataset into MDS format.

    Args:
        tokenizer_name (str): Name of tokenizer to use.
        output_folder (str): Folder to write output to.
        input_folder (str): Folder to read input from.
        dataset_subset (str): Dataset subset to use.
        concat_tokens (int): Number of tokens to concatenate.
        eos_text (str): Text to append to end of each sample.
        bos_text (str): Text to prepend to beginning of each sample.
        no_wrap (bool): Whether to allow wrapping of text across samples.
        max_workers (int): Max # of workers to use.
        compression (str): Compression to use.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    # we will enforce length, so suppress warnings about sequences too long for the model
    tokenizer.model_max_length = int(1e30)
    columns = {'tokens': 'bytes'}

    object_store = maybe_create_object_store_from_uri(input_folder)
    _, _, folder_prefix = parse_uri(input_folder)

    num_cpus = 1
    detected_cpus = os.cpu_count()
    if detected_cpus is not None:
        num_cpus = max(detected_cpus // 2, 1)

    for split in ['validation', 'test', 'train']:
        print(f'Processing {split}')
        with tempfile.TemporaryDirectory() as tmp_dir:
            sub_prefix = os.path.join(folder_prefix, split)
            sec_filing_data = datasets.load_dataset(
                'JanosAudran/financial-reports-sec',
                dataset_subset,
                num_proc=num_cpus,
                split=split)

            def joined_identifier(example: Dict[str, Any]):
                example[
                    'joined_identifier'] = f"{example['docID']}|||{example['tickers'][0]}|||{example['reportDate']}"
                return example

            sec_filing_data = sec_filing_data.map(joined_identifier,
                                                  num_proc=num_cpus)
            unique_identifiers = sec_filing_data.unique('joined_identifier')

            print(f'Processing {len(unique_identifiers)} documents')

            assert object_store is not None  # pyright
            downloading_iter = DownloadingIterable(unique_identifiers,
                                                   sub_prefix,
                                                   os.path.join(tmp_dir, split),
                                                   object_store)

            # Use the ConcatTokensDataset from LLM-foundry to concatenate sequences of tokens up to the maximum sequence length
            dataset = ConcatTokensDataset(
                hf_dataset=downloading_iter,
                max_length=concat_tokens,
                tokenizer=tokenizer,
                eos_text=eos_text,
                bos_text=bos_text,
                no_wrap=no_wrap,
            )

            # Generate samples
            loader = build_dataloader(dataset=dataset, batch_size=512)
            samples = generate_samples(loader)

            # Write samples in MDS format
            print(f'Converting to MDS format...')
            with MDSWriter(out=os.path.join(output_folder, split),
                           max_workers=max_workers,
                           progress_bar=False,
                           columns=columns,
                           compression=compression) as out:
                for sample in tqdm(samples):
                    out.write(sample)


if __name__ == '__main__':
    args = parse_args()
    main(
        tokenizer_name=args.tokenizer,
        output_folder=args.out_root,
        input_folder=args.in_root,
        dataset_subset=args.dataset_subset,
        concat_tokens=args.concat_tokens,
        eos_text=args.eos_text,
        bos_text=args.bos_text,
        no_wrap=args.no_wrap,
        max_workers=args.max_workers,
        compression=args.compression,
    )
