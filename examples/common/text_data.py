# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

"""Build a StreamingTextDataset dataset and dataloader for training."""

import os
import sys
import textwrap
import warnings
from itertools import islice
from typing import Any, Dict, Iterator, Optional

import numpy as np
import torch
import transformers
from omegaconf import DictConfig
from omegaconf import OmegaConf as om
from streaming import StreamingDataset
from torch.utils.data import DataLoader


class StreamingTextDataset(StreamingDataset):
    """Generic text dataset using MosaicML's StreamingDataset.

    Args:
        local (str): Local dataset directory where shards are cached by split.
        tokenizer_name (str): The name of the HuggingFace tokenizer to use to
            tokenize samples.
        max_seq_len (int): The max sequence length of each sample.
        group_method (str): (Deprecated, use ``--concat_text`` or ``--concat_tokens`` option in
            create_c4.py) How to group text samples into token samples. Supports 'truncate' or 'concat'.
        remote (str, optional): Download shards from this remote path or directory. If None, this
            rank and worker's partition of the dataset must all exist locally. Defaults to ``None``.
        split (str, optional): Which dataset split to use, if any. Defaults to ``None``.
        shuffle (bool): Whether to iterate over the samples in randomized order. Defaults to ``False``.
        predownload (int, optional): Target number of samples ahead to download the shards of while
            iterating. Defaults to ``100_000``.
        keep_zip (bool, optional): Whether to keep or delete the compressed file when
            decompressing downloaded shards. If set to None, keep if remote is local. Defaults to
            ``None``.
        download_retry (int): Number of download re-attempts before giving up. Defaults to ``2``.
        download_timeout (float): Number of seconds to wait for a shard to download before raising
            an exception. Defaults to ``60``.
        validate_hash (str, optional): Optional hash or checksum algorithm to use to validate
            shards. Defaults to ``None``.
        shuffle_seed (int): Seed for Deterministic data shuffling. Defaults to ``9176``.
        num_canonical_nodes (int, optional): Canonical number of nodes for shuffling with resumption.
            Defaults to ``None``, which is interpreted as the number of nodes of the initial run.
        batch_size (int, optional): Batch size of its DataLoader, which affects how the dataset is
            partitioned over the workers. Defaults to ``None``.
    """

    def __init__(self,
                 local: str,
                 tokenizer_name: str,
                 max_seq_len: int,
                 group_method: str = 'truncate',
                 remote: Optional[str] = None,
                 split: Optional[str] = None,
                 shuffle: bool = False,
                 predownload: Optional[int] = 100_000,
                 keep_zip: Optional[bool] = None,
                 download_retry: int = 2,
                 download_timeout: float = 60,
                 validate_hash: Optional[str] = None,
                 shuffle_seed: int = 9176,
                 num_canonical_nodes: Optional[int] = None,
                 batch_size: Optional[int] = None):

        # Validation
        warning_msg = textwrap.dedent("""\
            The group_method argument is deprecated and will be removed in the future.
             Using it makes the dataloader have no length and prevents deterministic resumption.
             It is also no longer needed, because concatenation can happen when we create
             the dataset in create_c4.py.
            """)
        warnings.warn(DeprecationWarning(warning_msg))

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
                         keep_zip=keep_zip,
                         download_retry=download_retry,
                         download_timeout=download_timeout,
                         validate_hash=validate_hash,
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
        if self.tokenizer._pad_token is None:
            # Some tokenizers (e.g. GPT2 tokenizer) have no padding token which causes bugs
            self.tokenizer._pad_token = self.tokenizer._eos_token
        # suppress warnings when using group_method='concat' and no truncation
        self.tokenizer.model_max_length = int(1e30)

        self.add_special_tokens = None

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

        # The first time we tokenize we need to figure out the special token situation,
        # as pre-concatenated text will already have special tokens
        if self.add_special_tokens is None:
            if self.group_method == 'concat':
                # True is the default value so this will not change any behavior
                self.add_special_tokens = True
            else:
                # if the 2nd token is BOS or 2nd-to-last-is EOS, the special tokens
                # were added as part of pre-concatenating the text, so don't add them again
                tokens = self.tokenizer(text_sample['text'],
                                        truncation=truncation,
                                        padding=padding,
                                        max_length=max_length,
                                        add_special_tokens=True)
                if tokens['input_ids'][
                        1] == self.tokenizer.bos_token_id or tokens[
                            'input_ids'][-2] == self.tokenizer.eos_token_id:
                    self.add_special_tokens = False
                else:
                    # Normal text without special tokens added
                    self.add_special_tokens = True

        return self.tokenizer(text_sample['text'],
                              truncation=truncation,
                              padding=padding,
                              max_length=max_length,
                              add_special_tokens=self.add_special_tokens)

    def _read_binary_tokenized_sample(self, sample):
        return torch.from_numpy(
            np.frombuffer(sample['tokens'], dtype=np.int64).copy())

    # How to process a sample
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = super().__getitem__(idx)
        if sample.get('text', None) is not None:
            token_sample = self._tokenize(sample)
        elif sample.get('tokens', None) is not None:
            token_sample = self._read_binary_tokenized_sample(sample)
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
    assert cfg.name == 'text', f'Tried to build text dataloader with cfg.name={cfg.name}'
    dataset = StreamingTextDataset(
        local=cfg.dataset.local,
        tokenizer_name=cfg.dataset.tokenizer_name,
        max_seq_len=cfg.dataset.max_seq_len,
        group_method=cfg.dataset.group_method,
        remote=cfg.dataset.get('remote', None),
        split=cfg.dataset.get('split', None),
        shuffle=cfg.dataset.get('shuffle', False),
        predownload=cfg.dataset.get('predownload', 100_000),
        keep_zip=cfg.dataset.get('keep_zip', False),
        download_retry=cfg.dataset.get('download_retry', 2),
        download_timeout=cfg.dataset.get('download_timeout', 60),
        validate_hash=cfg.dataset.get('validate_hash', None),
        shuffle_seed=cfg.dataset.get('shuffle_seed', None),
        num_canonical_nodes=cfg.dataset.get('num_canonical_nodes', 128),
        batch_size=device_batch_size)

    mlm_probability = cfg.dataset.get('mlm_probability', None)
    collate_fn = transformers.DataCollatorForLanguageModeling(
        tokenizer=dataset.tokenizer,
        mlm=mlm_probability is not None,
        mlm_probability=mlm_probability)

    return DataLoader(
        dataset,
        collate_fn=collate_fn,
        batch_size=device_batch_size,
        drop_last=cfg.drop_last,
        num_workers=cfg.num_workers,
        pin_memory=cfg.get('pin_memory', True),
        prefetch_factor=cfg.get('prefetch_factor', 2),
        persistent_workers=cfg.get('persistent_workers', True),
        timeout=cfg.get('timeout', 0),
    )


# Helpful to test if your dataloader is working locally
# Run `python data.py [remote] [local, optional]` and verify that batches are printed out
if __name__ == '__main__':

    if len(sys.argv) > 2:
        local, remote = sys.argv[1:3]
        print(f'Reading val split from {local} <- streamed from <- {remote}')
    else:
        local = sys.argv[1]
        remote = None
        print(f'Reading val split from {local}')

    cfg = {
        'name': 'text',
        'dataset': {
            'local': local,
            'remote': remote,
            'split': 'val',
            'shuffle': False,
            'tokenizer_name': 'facebook/opt-125m',
            'max_seq_len': 32,
            'group_method': 'truncate',
            'keep_zip': True,  # in case we need compressed files after testing
        },
        'drop_last': False,
        'num_workers': 4,
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
