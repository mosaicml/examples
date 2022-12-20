# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

"""Build a StreamingTextDataset dataset and dataloader for training."""

import os
import sys
from itertools import islice
from typing import Any, Dict, Iterator, Optional

import transformers
from omegaconf import DictConfig
from omegaconf import OmegaConf as om
from streaming import StreamingDataset, StreamingDataLoader


class StreamingTextDataset(StreamingDataset):
    """Generic implementation of a streaming text dataset using MosaicML's StreamingDataset.

    Args:
        local (str): Local filesystem directory where dataset is cached
            during operation.
        tokenizer_name (str): The name of the HuggingFace tokenizer to use to
            tokenize samples.
        max_seq_len (int): The max sequence length of each sample.
        group_method (str): How to group text samples into token samples.
            Supports 'truncate' or 'concat'.
        remote (str, optional): Download shards from this remote path or directory. If None, this
            rank and worker's partition of the dataset must all exist locally. Defaults to
            ``None``.
        split (str, optional): Which dataset split to use, if any. Defaults to ``None``.
        shuffle (bool): Whether to iterate over the samples in randomized order. Defaults to
            ``False``.
        predownload (int, optional): Target number of samples ahead to download the shards of while
            iterating. Defaults to ``100_000``.
        download_retry (int): Number of download re-attempts before giving up. Defaults to ``2``.
        download_timeout (float): Number of seconds to wait for a shard to download before raising
            an exception. Defaults to ``60``.
        shuffle_seed (int, optional): Seed for shuffling, or ``None`` for random seed. Defaults to
            ``None``.
        num_canonical_nodes (int, optional): Canonical number of nodes for shuffling with resumption.
            Defaults to ``None``, which is interpreted as the number of nodes of the initial run.
        batch_size (int, optional): Batch size of its DataLoader, which affects how the dataset is
            partitioned over the workers. Defaults to ``None``.
    """

    def __init__(self,
                 local,
                 tokenizer_name: str,
                 max_seq_len: int,
                 group_method: str = 'truncate',
                 remote: Optional[str] = None,
                 split: str = None,
                 shuffle: bool = False,
                 predownload: Optional[int] = 100_000,
                 download_retry: int = 2,
                 download_timeout: float = 120,
                 shuffle_seed: Optional[int] = None,
                 num_canonical_nodes: Optional[int] = None,
                 batch_size: Optional[int] = None):
        # Validation
        if split not in ['train', 'val']:
            raise ValueError(
                f"split='{split}' must be one of ['train', 'val'].")
        if group_method not in ['truncate', 'concat']:
            raise ValueError(
                f"group_method='{group_method}' must be one of ['truncate', 'concat']."
            )

        # Build Dataset
        super().__init__(local=local,
                         remote=remote,
                         split=split,
                         shuffle=shuffle,
                         predownload=predownload,
                         keep_zip=False,
                         download_retry=download_retry,
                         download_timeout=download_timeout,
                         validate_hash=None,
                         shuffle_seed=shuffle_seed,
                         num_canonical_nodes=num_canonical_nodes,
                         batch_size=batch_size)

        self.tokenizer_name = tokenizer_name
        self.max_seq_len = max_seq_len
        self.group_method = group_method

        # Build tokenizer
        os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.tokenizer_name)
        if self.tokenizer.pad_token is None:
            # Some tokenizers (e.g. GPT2 tokenizer) have no padding token which causes bugs
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # suppress warnings when using group_method='concat' and no truncation
        self.tokenizer.model_max_length = int(1e30)

    # How to tokenize a text sample to a token sample
    def _tokenize(self, text_sample):
        if self.group_method == 'truncate':
            truncation = True
            padding = 'max_length'
            max_length = self.max_seq_len
        elif self.group_method == 'concat':
            truncation = False
            padding = False
            max_length = None
        else:
            raise ValueError(f"Got unknown group_method='{self.group_method}'.")
        return self.tokenizer(text_sample['text'],
                              truncation=truncation,
                              padding=padding,
                              max_length=max_length)

    # How to process a sample
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        text_sample = super().__getitem__(idx)
        token_sample = self._tokenize(text_sample)
        return token_sample

    # Define iterable over samples
    # Usually this can be left alone and inherited directly from super()
    # class StreamingDataset, but concatenating samples is custom behavior.
    # If group_method=='truncate', we simply return the token sample.
    # If group_method=='concat', then we keep fetching token samples until we
    # fill up max_seq_len.
    def __iter__(self) -> Iterator[Any]:
        if self.group_method == 'truncate':
            iterator = super().__iter__()
            yield from iterator

        elif self.group_method == 'concat':
            buffer = {}
            while True:
                iterator = super().__iter__()
                for sample in iterator:

                    for k, v in sample.items():
                        buffer[k] = buffer.get(k, []) + v
                    while len(buffer['input_ids']) >= self.max_seq_len:
                        concat_sample = {}
                        for k, v in buffer.items():
                            concat_sample[k] = v[:self.max_seq_len]
                            buffer[k] = v[self.max_seq_len:]
                        yield concat_sample
        else:
            raise ValueError(f"Got unknown group_method='{self.group_method}'.")

    # Define length
    # Usually this can be left alone and inherited directly from super() class
    # Dataset, but concatenating samples is custom behavior.
    # If group_method=='truncate', we simply return the # samples.
    # If group_method=='concat', we repeat forever, and have no defined length.
    def __len__(self) -> Optional[int]:
        if self.group_method == 'truncate':
            return super().__len__()
        elif self.group_method == 'concat':
            return None
        else:
            raise ValueError(f"Got unknown group_method='{self.group_method}'.")


def build_text_dataloader(cfg: DictConfig, device_batch_size: int):
    dataset = StreamingTextDataset(
        local=cfg.dataset.local,
        tokenizer_name=cfg.dataset.tokenizer_name,
        max_seq_len=cfg.dataset.max_seq_len,
        group_method=cfg.dataset.group_method,
        remote=cfg.dataset.remote,
        split=cfg.dataset.split,
        shuffle=cfg.dataset.shuffle,
        predownload=cfg.dataset.prefetch,
        download_timeout=cfg.timeout,
        # shuffle_seed=cfg.seed,                                TODO
        # num_canonical_nodes=cfg.dataset.num_canonical_nodes,  TODO
        batch_size=device_batch_size,
    )

    collate_fn = transformers.DataCollatorForLanguageModeling(
        tokenizer=dataset.tokenizer,
        mlm=False
    )

    return StreamingDataLoader(
        dataset,
        collate_fn=collate_fn,
        batch_size=device_batch_size,
        drop_last=cfg.drop_last,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        prefetch_factor=cfg.prefetch_factor,
        persistent_workers=cfg.persistent_workers,
        timeout=cfg.timeout,
    )


# Helpful to test if your dataloader is working locally
# Run `python data.py [remote] [local, optional]` and verify that batches are printed out
if __name__ == '__main__':
    remote = sys.argv[1]
    ds_name = 'c4' if 'c4' in remote else 'the_pile' if 'pile' in remote else 'dataset'
    if len(sys.argv) > 2:
        local = sys.argv[2]
    else:
        local = f'/tmp/{ds_name}' if 's3' in remote else remote
    print(f'Reading val split of {ds_name} dataset from {remote} -> {local}')

    #     dataset:
    #   remote: *data_remote
    #   local: *data_local
    #   split: val
    #   shuffle: false
    #   prefetch: 1000
    #   tokenizer_name: *tokenizer_name
    #   max_seq_len: *max_seq_len
    #   group_method: truncate
    # drop_last: false
    # num_workers: 8
    # pin_memory: true
    # prefetch_factor: 2
    # persistent_workers: true
    # timeout: 60

    cfg = {
        'dataset': {
            'remote': remote,
            'local': local,
            'split': 'val',
            'shuffle': True,
            'prefetch': 1000,
            'tokenizer_name': 'gpt2',
            'max_seq_len': 32,
            'group_method': 'truncate'
        },
        'drop_last': False,
        'num_workers': 2,
        'pin_memory': True,
        'prefetch_factor': 2,
        'persistent_workers': True,
        'timeout': 60,
    }
    cfg = om.create(cfg)
    device_batch_size = 2

    loader = build_text_dataloader(cfg, device_batch_size)
    tokenizer = loader.dataset.tokenizer  # type: ignore
    for batch_ix, batch in enumerate(islice(loader, 5)):
        print('\n')
        print('#' * 20, f'Batch {batch_ix}', '#' * 20)
        for k, v in batch.items():
            print(k, v.shape, v.dtype)
        for sample_ix, token_sample in enumerate(batch['input_ids']):
            print('-' * 20, f' Sample {sample_ix} ', '-' * 20)
            print(tokenizer.decode(token_sample))
