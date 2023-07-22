import os
import json
import oci
from argparse import ArgumentParser, Namespace
from typing import Dict, Iterable, Optional
from llmfoundry.data import ConcatTokensDataset  # type: ignore
from streaming import MDSWriter, StreamingDataset
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import datasets

from transformers import AutoTokenizer

DATA_PATH = 'train_data/pipeline_data/pypi_documentation/'
MAX_ENTRIES_PER_FILE = 25000  # Maximum number of entries per file

def parse_args()-> Namespace:
    parser = ArgumentParser(
        description=
        'Convert dataset into MDS format, optionally concatenating and tokenizing'
    )
    parser.add_argument('--out_root', type=str, required=True,
                        help='The destination object name in OCI')

    parser.add_argument(
        '--max_workers',
        type=int,
        default=64,
        help='The maximum number of workers to use for MDS writing')

    parser.add_argument('--compression',
                        type=str,
                        default='zstd',
                        help='The compression algorithm to use for MDS writing')

    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        '--concat_tokens',
        type=int,
        default=2048,
        help='Convert text to tokens and concatenate up to this many tokens')

    parser.add_argument('--tokenizer',
                        type=str,
                        required=False,
                        default='mosaicml/mpt-7b',
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
        default='<|endoftext|>',
        help=
        'The text to append to each example to separate concatenated examples')
    parser.add_argument(
        '--no_wrap',
        default=False,
        action='store_true',
        help=
        'Whether to let text examples wrap across multiple training examples')

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


def generate_samples(
        dataset: Dataset, 
        truncate_num_samples: Optional[int] = None
) -> Iterable[Dict[str, bytes]]:
    """Generator over samples of a dataloader.

    Args:
       dataset (Dataset): A dataset to be converted
       truncate_num_samples (Optional[int]): An optional # of samples to stop at.

    Yields:
        Sample dicts.
    """
    n_samples = 0
    for data in dataset:
        if truncate_num_samples is not None and n_samples == truncate_num_samples:
            return
        n_samples += 1
        yield data

class DatasetIterable:
    def __init__(self, 
                 dataset: datasets.Dataset):
        self.dataset = dataset

    def __iter__(self):
        for item in self.dataset:
            yield {'text': json.dumps(item)}


def convert_to_MDS(
    output_folder: str,
    tokenizer_name: str,
    concat_tokens: int,
    eos_text: str,
    bos_text: str,
    no_wrap: bool,
    folder: str,
    file_path: str,
    max_workers: int,
    compression: str) -> None:
    """Convert the generic txt dataset into MDS format.

    Args:
        tokenizer_name (str): Name of tokenizer to use.
        output_folder (str): Folder to write output to.
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

    data = datasets.load_dataset('json', data_files=file_path, split=None, cache_dir=None)['train']

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
    samples = generate_samples(dataset)

    # Write samples in MDS format
    print(f'Converting to MDS format...')
    with MDSWriter(out=os.path.join(output_folder, 'train' if folder == 'source_code_processed_train' else 'validation'),
                    max_workers=max_workers,
                    progress_bar=False,
                    columns=columns,
                    compression=compression) as out:
        for sample in tqdm(samples):
            out.write(sample)

def main(
    output_folder: str,
    tokenizer_name: str,
    concat_tokens: int,
    eos_text: str,
    bos_text: str,
    no_wrap: bool,
    max_workers: int,
    compression: str) -> None:
    """Convert the MDS dataset into correct concat tokens MDS format.

    Args:
        tokenizer_name (str): Name of tokenizer to use.
        output_folder (str): Folder to write output to.
        dataset_subset (str): Dataset subset to use.
        concat_tokens (int): Number of tokens to concatenate.
        eos_text (str): Text to append to end of each sample.
        bos_text (str): Text to prepend to beginning of each sample.
        no_wrap (bool): Whether to allow wrapping of text across samples.
        max_workers (int): Max # of workers to use.
        compression (str): Compression to use.
    """

    os.makedirs(DATA_PATH, exist_ok=True)
    remote_source_folders = ['source_code_processed_train', 'source_code_processed_val']
    for folder in remote_source_folders:
        remote_dir = os.path.join('oci://mosaicml-internal-datasets/mpt-swe-filtered/', folder)
        local_dir = os.path.join(DATA_PATH, folder)
        os.makedirs(local_dir, exist_ok=True)
        dataset = StreamingDataset(remote=remote_dir, split=None, shuffle=False)

        file_count = 0
        current_file_entries = 0

        # Create the initial file
        file_path = os.path.join(local_dir, f'data_{file_count}.jsonl')
        file = open(file_path, 'w')

        for doc_data in tqdm(dataset, desc='Loading PyPi docstrings', total=len(dataset)):
            text = doc_data['text']
            text = ', '.join(text)
            url = doc_data['url']
            url = ', '.join(url)
            package_name = doc_data['package_name']
            package_name = ', '.join(package_name)

            # Create the entry as a dictionary, then convert to json
            entry = {
                'url': url,
                'package name': package_name,
                'text': text
            }

            # If this entry makes the file exceed the maximum number of entries, create a new file
            if current_file_entries == MAX_ENTRIES_PER_FILE:
                file.close()
                convert_to_MDS(output_folder=output_folder,
                               tokenizer_name=tokenizer_name,
                               concat_tokens=concat_tokens,
                               eos_text=eos_text,
                               bos_text=bos_text,
                               no_wrap=no_wrap,
                               folder=folder,
                               file_path=file_path,
                               max_workers=max_workers,
                               compression=compression)

                os.remove(file_path)
                
                file_count += 1
                file_path = os.path.join(local_dir, f'data_{file_count}.jsonl')
                file = open(file_path, 'w', encoding='utf-8')
                current_file_entries = 0

            # Write the entry to the file
            file.write(json.dumps(entry) + '\n')
            current_file_entries += 1

        # Close the last file
        file.close()
        convert_to_MDS(output_folder=output_folder,
                       tokenizer_name=tokenizer_name,
                       concat_tokens=concat_tokens,
                       eos_text=eos_text,
                       bos_text=bos_text,
                       no_wrap=no_wrap,
                       folder=folder,
                       file_path=file_path,
                       max_workers=max_workers,
                       compression=compression)
        
        os.remove(file_path)
    if os.path.exists(DATA_PATH):
        os.remove(DATA_PATH)

if __name__ == '__main__':
    args = parse_args()
    main(
        tokenizer_name=args.tokenizer,
        output_folder=args.out_root,
        concat_tokens=args.concat_tokens,
        eos_text=args.eos_text,
        bos_text=args.bos_text,
        no_wrap=args.no_wrap,
        max_workers=args.max_workers,
        compression=args.compression,
    )