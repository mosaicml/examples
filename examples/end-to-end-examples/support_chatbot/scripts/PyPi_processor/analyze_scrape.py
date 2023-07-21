

## scrape_pypi.analyze_scrape
import random
from textwrap import wrap
from torch.utils.data import DataLoader
from streaming import StreamingDataset
from transformers import AutoTokenizer
import sys
import numpy as np
import tqdm
import os
import shutil
import re
from streaming import MDSWriter



if __name__ == "__main__":
    if os.path.exists('local_dir'):
        shutil.rmtree('local_dir')
    # Remote directory (S3 or local filesystem) where dataset is stored
    remote_dir = 'oci://mosaicml-internal-datasets/mpt-swe/docstring_processed_train/'

    # Local directory where dataset is cached during operation
    local_dir = 'local_dir2'
    dataset = StreamingDataset(local=local_dir, remote=remote_dir, split=None, shuffle=False)

    # Create PyTorch DataLoader
    tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')
    dataloader = DataLoader(dataset)
    stats = {
        'samples': 0,
        'prompt_tokens': 0,
        'continuation_tokens': 0
    }
    for batch in tqdm.tqdm(dataloader):
        
        stats['prompt_tokens'] += len(tokenizer(batch['instruction'][0])['input_ids'])
        stats['continuation_tokens'] += len(tokenizer(batch['completion'][0])['input_ids'])
        breakpoint()
        stats['samples'] += 1
        if stats['samples'] % 1000 == 0:
            print(stats)
        
    
    print(stats)