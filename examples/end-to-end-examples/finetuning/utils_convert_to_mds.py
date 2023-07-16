# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from argparse import ArgumentParser, Namespace
import os
import tempfile
from typing import Dict, Iterable, List, Optional

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
    parser.add_argument('--output_folder',
                        type=str,
                        required=True,
                        help='The folder to write output to')
    parser.add_argument('--train_index_path',
                        type=str,
                        required=True,
                        help='The file that lists all the data paths to read from for training')
    parser.add_argument('--eval_index_path',
                        type=str,
                        required=True,
                        help='The file that lists all the data paths to read from for evaluation')
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
        self.identifiers = identifiers
        self.object_store = object_store
        self.input_folder_prefix = input_folder_prefix
        self.output_folder = output_folder

    def __iter__(self):
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1
        identifiers_shard = self.identifiers[worker_id::num_workers]
        print(f'Worker {worker_id} processing {len(identifiers_shard)} files')
        for shard in identifiers_shard:
            self.object_store.download_object(
                os.path.join(self.input_folder_prefix, shard),
                os.path.join(self.output_folder, shard),
            )

            with open(
                    os.path.join(self.output_folder,
                                 shard)) as _txt_file:
                txt = _txt_file.read()

            yield {'text': txt}


def convert_data(
    tokenizer_name: str,
    output_folder: str,
    train_index_path: str,
    eval_index_path: str,
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

    for split in ['train', 'validate']:
        if split == 'train':
            path = train_index_path
        else:
            path = eval_index_path
        print(f'Processing {path}')
        index_object = maybe_create_object_store_from_uri(path)
        _, _, folder_prefix = parse_uri(path)
        index_object.download_object('index.txt', '.')
        with open('index.txt') as index_file:
            file_list = index_file.read().splitlines()
        
        with tempfile.TemporaryDirectory() as tmp_dir:  
          downloading_iter = DownloadingIterable(file_list,
                                                 folder_prefix,
                                                 os.path.join(tmp_dir, path),
                                                 index_object)
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
    convert_data(
        tokenizer_name=args.tokenizer,
        output_folder=args.output_folder,
        train_index_path=args.train_index_path,
        eval_index_path=args.eval_index_path,
        concat_tokens=args.concat_tokens,
        eos_text=args.eos_text,
        bos_text=args.bos_text,
        no_wrap=args.no_wrap,
        max_workers=args.max_workers,
        compression=args.compression,
    )
