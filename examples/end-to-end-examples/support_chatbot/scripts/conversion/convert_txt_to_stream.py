# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import os
from argparse import ArgumentParser, Namespace
from typing import Dict, Iterable, Optional
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
        required=False,
        help='The maximum number of workers to use for MDS writing')
    
    parser.add_argument(
        '--out_root',
        type=str,
        required=True,
        help='The folder to write output to')
    
    parser.add_argument(
        '--in_root',
        type=str,
        required=True,
        help='The folder to read input from')

    parser.add_argument(
        '--compression',
        type=str,
        default='zstd',
        help='The compression algorithm to use for MDS writing')

    parser.add_argument(
        '--concat_tokens',
        type=int,
        default=8192,
        required=False,
        help='Convert text to tokens and concatenate up to this many tokens')

    parser.add_argument(
        '--tokenizer',
        type=str,
        default='mosaicml/mpt-30b',
        required=False,
        help='The name of the tokenizer to use')
    parser.add_argument(
        '--bos_text',
        type=str,
        default=None,
        required=False,
        help='The text to prepend to each example to separate concatenated examples')
    parser.add_argument(
        '--eos_text',
        type=str,
        default='<|endoftext|>',
        required=False,
        help='The text to append to each example to separate concatenated examples')
    parser.add_argument(
        '--no_wrap',
        action='store_true',
        required=False,
        help='Whether to let text examples wrap across multiple training examples')

    parsed = parser.parse_args()

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

class DatasetIterable:
    def __init__(self, dataset: list[str]):
        self.dataset = list(set(dataset))  # Remove duplicates
        print(f'Total files in the dataset: {len(self.dataset)}')

    def __iter__(self):
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1
        string_shard = self.dataset[worker_id::num_workers]
        print(f'Worker {worker_id} processing {len(string_shard)} files')

        for file in string_shard:
            print(f'Processing file: {file}')
            try:
                with open(file, 'r') as f:
                    for line in f:
                        yield {'text': line.strip()}
            except Exception as e:
                print(f'Error processing file: {file}. Error: {e}')


def main(
    input_folder: str,
    output_folder: str,
    tokenizer_name: str,
    concat_tokens: int,
    eos_text: str,
    bos_text: str,
    no_wrap: bool,
    max_workers: int,
    compression: str) -> None:
    """Convert the generic txt dataset into MDS format.

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

    files = os.listdir(input_folder)
    txt_files = [os.path.join(input_folder, f) for f in files if f.endswith('.txt')]
    num_files = len(txt_files)

    split_dataset = {
        'train': txt_files[:int(num_files * 0.8)],
        'validation': txt_files[int(num_files * 0.8):int(num_files * 0.9)],
        'test': txt_files[int(num_files * 0.9):]
    }


    for split in ['train', 'validation', 'test']:
        print(f'Processing {split}')
        data = split_dataset[split]

        # Use the ConcatTokensDataset from LLM-foundry to concatenate sequences of tokens up to the maximum sequence length
        dataset = ConcatTokensDataset(
            hf_dataset=DatasetIterable(dataset=data),
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
        concat_tokens=args.concat_tokens,
        eos_text=args.eos_text,
        bos_text=args.bos_text,
        no_wrap=args.no_wrap,
        max_workers=args.max_workers,
        compression=args.compression,
    )