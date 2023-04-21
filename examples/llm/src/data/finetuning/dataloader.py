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
from examples.llm.src.data.packing import BinPackWrapper

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
            - cfg.dataset.local (local location if using a streaming dataset, optional)
            - cfg.dataset.remote (remote location if using a streaming dataset, optional)
            - cfg.dataset.split (e.g. "train" or "validation")
            - cfg.dataset.tokenizer_name (e.g. "gpt2")
            - cfg.dataset.max_seq_len (e.g., 512)
            - cfg.dataset.decoder_only_format (should be True for a GPT model)
            - cfg.dataset.allow_pad_trimming (default is False)
            - cfg.dataset.packing_ratio (set to >1.0 to activate packing)
            - cfg.dataset.shuffle (e.g. True for training, False for validation)

            - cfg.drop_last (e.g. False)
            - cfg.num_workers (e.g. 8)
            - cfg.pin_memory (e.g. True)
            - cfg.prefetch_factor (e.g. 2)
            - cfg.persistent_workers (e.g. True)
            - cfg.timeout (e.g. 30)

        device_batch_size (int): The batch size that should be loaded on each device.

    Returns:
        A pytorch dataloader

    Note:
        You can run the script inside `src/data/packing.py` to quickly test the padding/waste
        rate for different `cfg.dataset.packing_ratio` choices, given a starting workload YAML.
    """
    if cfg.dataset.get('local') is not None:
        dataset = dataset_constructor.build_from_streaming(
            cfg.dataset.name,
            cfg.dataset.tokenizer_name,
            local=cfg.dataset.local,
            remote=cfg.dataset.get('remote', None),
            split=cfg.dataset.get('split'),
            shuffle=cfg.dataset.get('shuffle', False),
            predownload=cfg.dataset.get('predownload', 100_000),
            keep_zip=cfg.dataset.get('keep_zip', None),
            download_retry=cfg.dataset.get('download_retry', 2),
            download_timeout=cfg.dataset.get('download_timeout', 60),
            validate_hash=cfg.dataset.get('validate_hash', None),
            shuffle_seed=cfg.dataset.get('shuffle_seed', 9176),
            num_canonical_nodes=cfg.dataset.get('num_canonical_nodes', 128),
            batch_size=device_batch_size)

        collate_fn = Seq2SeqFinetuningCollator(
            dataset.tokenizer,
            max_seq_len=cfg.dataset.max_seq_len,
            decoder_only_format=cfg.dataset.decoder_only_format,
            allow_pad_trimming=cfg.dataset.get('allow_pad_trimming', False),
        )

        if cfg.dataset.get('packing_ratio'):
            n_examples_to_pack = int(device_batch_size *
                                     cfg.dataset.packing_ratio)
            if n_examples_to_pack < device_batch_size:
                raise ValueError('packing_ratio must be >= 1, if supplied')
            if not cfg.dataset.decoder_only_format:
                raise NotImplementedError(
                    'On-the-fly packing is currently only supported for decoder-only formats.'
                )
            collate_fn = BinPackWrapper(
                collator=collate_fn,
                target_batch_size=device_batch_size,
                max_seq_len=cfg.dataset.max_seq_len,
                pad_token_id=dataset.tokenizer.pad_token_id,
                padding_side=dataset.tokenizer.padding_side,
                max_leftover_bins_to_keep=cfg.dataset.get(
                    'max_leftover_bins_to_keep'),
            )
            device_batch_size = n_examples_to_pack
        elif cfg.dataset.get('max_leftover_bins_to_keep') is not None:
            raise ValueError(
                'cfg.dataset.max_leftover_bins_to_keep has been defined, ' +\
                'but cfg.dataset.packing_ratio has not been set. Please set ' +\
                'the latter to turn on packing or remove the former from the config.')

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

    else:
        if cfg.dataset.get('remote') is not None:
            raise ValueError(
                f'{cfg.dataset.remote=} but cfg.dataset.local is None.')

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            cfg.dataset.tokenizer_name)

        if tokenizer.pad_token is None:  # type: ignore
            tokenizer.pad_token = tokenizer.eos_token

        dataset = dataset_constructor.build(cfg.dataset.name, tokenizer,
                                            cfg.dataset.split)

        collate_fn = Seq2SeqFinetuningCollator(
            tokenizer,
            max_seq_len=cfg.dataset.max_seq_len,
            decoder_only_format=cfg.dataset.decoder_only_format,
            allow_pad_trimming=cfg.dataset.get('allow_pad_trimming', False),
        )

        if cfg.dataset.get('packing_ratio'):
            n_examples_to_pack = int(device_batch_size *
                                     cfg.dataset.packing_ratio)
            if n_examples_to_pack < device_batch_size:
                raise ValueError('packing_ratio must be >= 1, if supplied')
            if not cfg.dataset.decoder_only_format:
                raise NotImplementedError(
                    'On-the-fly packing is currently only supported for decoder-only formats.'
                )
            collate_fn = BinPackWrapper(
                collator=collate_fn,
                target_batch_size=device_batch_size,
                max_seq_len=cfg.dataset.max_seq_len,
                pad_token_id=tokenizer.pad_token_id,
                padding_side=tokenizer.padding_side,
                max_leftover_bins_to_keep=cfg.dataset.get(
                    'max_leftover_bins_to_keep'),
            )
            device_batch_size = n_examples_to_pack
        elif cfg.dataset.get('max_leftover_bins_to_keep') is not None:
            raise ValueError(
                'cfg.dataset.max_leftover_bins_to_keep has been defined, ' +\
                'but cfg.dataset.packing_ratio has not been set. Please set ' +\
                'the latter to turn on packing or remove the former from the config.')

        return DataLoader(
            dataset,
            collate_fn=collate_fn,
            batch_size=device_batch_size,
            sampler=dist.get_sampler(dataset,
                                     drop_last=cfg.drop_last,
                                     shuffle=cfg.dataset.shuffle),
            num_workers=cfg.num_workers,
            pin_memory=cfg.get('pin_memory', True),
            prefetch_factor=cfg.get('prefetch_factor', 2),
            persistent_workers=cfg.get('persistent_workers', True),
            timeout=cfg.get('timeout', 0),
        )


if __name__ == '__main__':
    import torch
    from omegaconf import OmegaConf as om
    cfg = om.create({
        'dataset': {
            'name': 'tatsu-lab/alpaca',
            'split': 'train',
            'packing_ratio': 18.0,
            'tokenizer_name': 'gpt2',
            'max_seq_len': 2048,
            'decoder_only_format': True,
            'separator_text': False,
            'allow_pad_trimming': False,
            'num_canonical_nodes': 472,
            'shuffle': True,
        },
        'drop_last': False,
        'num_workers': 0,
        'pin_memory': False,
        'prefetch_factor': 2,
        'persistent_workers': False,
        'timeout': 0
    })

    device_batch_size = 8

    dataloader = build_finetuning_dataloader(cfg, device_batch_size)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        cfg.dataset.tokenizer_name)

    packing = cfg.dataset.get('packing_ratio') is not None

    for i, batch in enumerate(dataloader):
        if i >= 100:
            break
        if i >= 1:
            if not packing:
                break
            continue
        print(f'-----Batch {i}-----')
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                print(k, v.shape)
            else:
                print(k, v)
        for j in range(device_batch_size):
            print(f'--- Sample {j} ---')
            if cfg.dataset.decoder_only_format:
                if packing:
                    for subseq in range(int(batch['sequence_id'][j].max()) + 1):
                        is_subseq = batch['sequence_id'][j] == subseq
                        print(
                            '\033[93m{}\033[00m\n'.format('INPUT IDS:'),
                            tokenizer.decode(batch['input_ids'][
                                j,
                                torch.logical_and(
                                    is_subseq, batch['attention_mask'][j] ==
                                    1)],
                                             skip_special_tokens=False))
                        print(
                            '\033[92m{}\033[00m\n'.format('CONTEXT:  '),
                            tokenizer.decode(batch['input_ids'][
                                j,
                                torch.logical_and(
                                    is_subseq, batch['bidirectional_mask'][j] ==
                                    1)],
                                             skip_special_tokens=False))
                        print(
                            '\033[91m{}\033[00m\n'.format('TARGET:   '),
                            tokenizer.decode(batch['input_ids'][
                                j,
                                torch.logical_and(
                                    is_subseq,
                                    batch['labels'][j] != _HF_IGNORE_INDEX)],
                                             skip_special_tokens=False))
                else:
                    print(
                        '\033[93m{}\033[00m\n'.format('INPUT IDS:'),
                        tokenizer.decode(
                            batch['input_ids'][j,
                                               batch['attention_mask'][j] == 1],
                            skip_special_tokens=False))
                    print(
                        '\033[92m{}\033[00m\n'.format('CONTEXT:  '),
                        tokenizer.decode(batch['input_ids'][
                            j, batch['bidirectional_mask'][j] == 1],
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
    if packing:
        print(f'Padding = {100*(1-dataloader.collate_fn.efficiency):5.2f}%')
        print(f'Waste   = {100*dataloader.collate_fn.waste:5.2f}%')
