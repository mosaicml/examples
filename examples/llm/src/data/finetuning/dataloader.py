# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

import logging

import torch
import transformers
from composer.utils import dist
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from examples.llm.src.data.finetuning._tasks import dataset_constructor
from examples.llm.src.data.finetuning.collator import Seq2SeqFinetuningCollator

log = logging.getLogger(__name__)

# HuggingFace hardcodes the ignore index to -100
_HF_IGNORE_INDEX = -100


def build_finetuning_dataloader(cfg: DictConfig,
                                device_batch_size: int) -> DataLoader:
    """Builds a dataloader for training or evaluating.

    Args:
        cfg (Mapping): The config for your dataset/dataloader. The config needs
            to define the following:
            - cfg.dataset.name (e.g. "tatsu-lab/alpaca"; must be
                registered in `dataset_constructor` -- see `./_tasks.py` for details)
            - cfg.dataset.split (e.g. "train" or "validation")
            - cfg.dataset.tokenizer_name (e.g. "gpt2")
            - cfg.dataset.max_seq_length (e.g., 512)
            - cfg.dataset.decoder_only_format (should be True for a GPT model)
            - cfg.dataset.local (local location if using a streaming dataset)
            - cfg.dataset.remote (remote location if using a streaming dataset)
            - cfg.drop_last (e.g. False)
            - cfg.shuffle (e.g. True for training, False for validation)
            - cfg.num_workers (e.g. 8)
            - cfg.pin_memory (e.g. True)
            - cfg.prefetch_factor (e.g. 2)
            - cfg.persistent_workers (e.g. True)
            - cfg.timeout (e.g. 30)

        device_batch_size (int): The batch size that should be loaded on each device.

    Returns:
        A pytorch dataloader
    """
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        cfg.dataset.tokenizer_name,
        model_max_length=cfg.dataset.max_seq_length,
        padding_side='left' if cfg.dataset.decoder_only_format else
        'right')  #type: ignore (thirdparty)

    if tokenizer.pad_token is None:  # type: ignore
        tokenizer.pad_token = tokenizer.eos_token

    # custom for P3
    if cfg.dataset.get('local') is not None:
        dataset = dataset_constructor.build_from_streaming(
            cfg.dataset.name,
            tokenizer,
            cfg.dataset.split,
            local=cfg.dataset.local,
            remote=cfg.dataset.get('remote'),
            shuffle=cfg.shuffle,
            predownload=cfg.dataset.get('predownload', 100_000),
            keep_zip=cfg.dataset.get('keep_zip', False),
            download_retry=cfg.dataset.get('download_retry', 2),
            download_timeout=cfg.dataset.get('download_timeout', 60),
            validate_hash=cfg.dataset.get('validate_hash'),
            shuffle_seed=cfg.dataset.get('shuffle_seed'),
            num_canonical_nodes=cfg.dataset.get('num_canonical_nodes'),
            batch_size=device_batch_size)

        return DataLoader(
            dataset,
            collate_fn=Seq2SeqFinetuningCollator(
                tokenizer,
                max_seq_length=cfg.dataset.max_seq_length,
                decoder_only_format=cfg.dataset.decoder_only_format,
                allow_pad_trimming=cfg.dataset.get('allow_pad_trimming', False),
            ),
            batch_size=device_batch_size,
            drop_last=cfg.drop_last,
            num_workers=cfg.num_workers,
            pin_memory=cfg.get('pin_memory', True),
            prefetch_factor=cfg.get('prefetch_factor', 2),
            persistent_workers=cfg.get('persistent_workers', False),
            timeout=cfg.get('timeout', 0),
        )

    else:
        if cfg.dataset.get('remote') is not None:
            raise ValueError(
                f'{cfg.dataset.remote=} but cfg.dataset.local is None.')
        dataset = dataset_constructor.build(cfg.dataset.name, tokenizer,
                                            cfg.dataset.split)

        return DataLoader(
            dataset,
            collate_fn=Seq2SeqFinetuningCollator(
                tokenizer,
                max_seq_length=cfg.dataset.max_seq_length,
                decoder_only_format=cfg.dataset.decoder_only_format,
                allow_pad_trimming=cfg.dataset.get('allow_pad_trimming', False),
            ),
            batch_size=device_batch_size,
            sampler=dist.get_sampler(dataset,
                                     drop_last=cfg.drop_last,
                                     shuffle=cfg.shuffle),
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            prefetch_factor=cfg.prefetch_factor,
            persistent_workers=cfg.persistent_workers,
            timeout=cfg.timeout,
        )


if __name__ == '__main__':
    from omegaconf import OmegaConf as om
    cfg = om.create({
        'dataset': {
            'name': 'tatsu-lab/alpaca',
            'split': 'train',
            'tokenizer_name': 'gpt2',
            'max_seq_length': 2048,
            'decoder_only_format': True,
            'separator_text': False,
            'allow_pad_trimming': False,
        },
        'drop_last': False,
        'shuffle': True,
        'num_workers': 0,
        'pin_memory': False,
        'prefetch_factor': 2,
        'persistent_workers': False,
        'timeout': 0
    })

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        cfg.dataset.tokenizer_name)

    device_batch_size = 2

    dataloader = build_finetuning_dataloader(cfg, device_batch_size)

    import torch
    for i, batch in enumerate(dataloader):
        print(f'-----Batch {i}-----')
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                print(k, v.shape)
            else:
                print(k, v)
        for j in range(device_batch_size):
            print(f'--- Sample {j} ---')
            if cfg.dataset.decoder_only_format:
                print(
                    '\033[93m{}\033[00m\n'.format('INPUT IDS:'),
                    tokenizer.decode(
                        batch['input_ids'][j, batch['attention_mask'][j] == 1],
                        skip_special_tokens=False))
                print(
                    '\033[92m{}\033[00m\n'.format('CONTEXT:  '),
                    tokenizer.decode(
                        batch['input_ids'][j,
                                           batch['bidirectional_mask'][j] == 1],
                        skip_special_tokens=False))
                print(
                    '\033[91m{}\033[00m\n'.format('TARGET:   '),
                    tokenizer.decode(batch['input_ids'][
                        j, batch['labels'][j] != _HF_IGNORE_INDEX],
                                     skip_special_tokens=False))
            else:
                print(
                    '\033[92m{}\033[00m\n'.format('CONTEXT:  '),
                    tokenizer.decode(
                        batch['input_ids'][j, batch['attention_mask'][j] == 1],
                        skip_special_tokens=False))
                print(
                    '\033[91m{}\033[00m\n'.format('TARGET:   '),
                    tokenizer.decode(batch['labels'][
                        j, batch['decoder_attention_mask'][j] == 1],
                                     skip_special_tokens=False))
        print('   ')
        if i >= 5:
            break
