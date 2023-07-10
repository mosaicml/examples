import os
from typing import Dict, Iterable, Optional
from llmfoundry.data.text_data import StreamingTextDataset
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

def main() -> None:
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

    remote_source_folders = ['source_code_processed_train', 'source_code_processed_val']
    tokenizer = AutoTokenizer.from_pretrained('mosaicml/mpt-7b')
    for folder in remote_source_folders:
        remote_dir = os.path.join('oci://mosaicml-internal-datasets/mpt-swe-filtered/', folder)
        dataset = StreamingTextDataset(remote=remote_dir, split=None, shuffle=False, max_seq_len=2048, tokenizer=tokenizer)
        print(dataset.num_samples)

if __name__ == '__main__':
    main()