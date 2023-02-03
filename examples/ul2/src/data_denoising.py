# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

"""Streaming dataloader for (mixture of) denoising task(s)."""

import logging
import random
import sys
from itertools import islice
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import torch
from omegaconf import DictConfig
from omegaconf import OmegaConf as om
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from examples.common.text_data import StreamingTextDataset
from examples.ul2.src import utils

__all__ = ['MixtureOfDenoisersCollator', 'build_text_denoising_dataloader']

log = logging.getLogger(__name__)

# HuggingFace hardcodes the ignore index to -100
_HF_IGNORE_INDEX = -100

Tokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]

# Required signature of any `prefix_function` (see below)
PREFIX_FUNCTION = Callable[[float, Optional[float], Tokenizer], Sequence[int]]


def ul2_prefix_function(
    mask_ratio: float,
    mean_length: Optional[float],
    tokenizer: Tokenizer,
) -> Sequence[int]:
    """Generates prefixes based on UL2 paper.

    See: http://arxiv.org/abs/2205.05131
    """
    if mean_length is None:
        # This is the case for "sequence to sequence"
        prefix = '[S2S]'
    elif mean_length >= 12 or mask_ratio >= 0.3:
        # UL2 tags this corruption rate "extreme"
        prefix = '[NLG]'
    else:
        # UL2 tags this corruption rate as "regular"
        prefix = '[NLU]'
    return tokenizer(prefix, add_special_tokens=False).input_ids


class MixtureOfDenoisersCollator:
    """Data collator for mixture of span-corruption denoisers, as in UL2.

    TODO(Alex): Complete docstring.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        max_seq_length: int,
        decoder_only_format: bool = False,
        span_mean_lengths_and_ratios: Optional[Union[List[List[float]],
                                                     List[float]]] = None,
        sequence_mask_ratios: Optional[Union[List[float], float]] = None,
        prefix_function: Optional[PREFIX_FUNCTION] = ul2_prefix_function,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.decoder_only_format = decoder_only_format

        # Prepare the tokenizer for denoising tasks
        n_sentinels = 100
        utils.adapt_tokenizer_for_denoising(self.tokenizer,
                                            num_sentinel_tokens=n_sentinels)
        sentinels = ''.join([f'<extra_id_{i}>' for i in range(n_sentinels)])
        self.sentinel_token_ids = np.array(tokenizer(sentinels).input_ids)

        # Populate the noisers so we can learn to denoise them!
        self._noisers = []

        # Add "noisers" for the span corruption denoising task
        self.span_mean_lengths_and_ratios = span_mean_lengths_and_ratios
        if self.span_mean_lengths_and_ratios is None:
            self.span_mean_lengths_and_ratios = []
        elif isinstance(self.span_mean_lengths_and_ratios[0], (int, float)):
            if not len(self.span_mean_lengths_and_ratios) == 2:
                raise ValueError('`span_mean_lengths_and_ratios` must be a '
                                 'pair of [mean_length, mask_ratio] or a list '
                                 'of such pairs.')
            self.span_mean_lengths_and_ratios = [
                self.span_mean_lengths_and_ratios
            ]
        # Each mean_length / mask_ratio combo becomes one of the span
        # corruption denoising tasks
        for span_mean_length, span_mask_ratio in self.span_mean_lengths_and_ratios:
            if span_mean_length < 0:
                raise ValueError('All span mean lengths must be positive.')
            if not 0 < span_mask_ratio < 1.0:
                raise ValueError(
                    'All span masking ratios must be between 0.0 and 1.0.')

            if prefix_function is not None:
                prefix_tokens = prefix_function(span_mask_ratio,
                                                span_mean_length,
                                                self.tokenizer)
            else:
                prefix_tokens = None

            kwargs = {
                'mean_span_length': span_mean_length,
                'mask_ratio': span_mask_ratio,
                'prefix_tokens': prefix_tokens,
            }
            self._noisers.append(kwargs)

        # Add "noisers" for the sequential denoising task
        self.sequence_mask_ratios = sequence_mask_ratios
        if self.sequence_mask_ratios is None:
            self.sequence_mask_ratios = []
        elif isinstance(self.sequence_mask_ratios, float):
            self.sequence_mask_ratios = [self.sequence_mask_ratios]
        for sequence_mask_ratio in self.sequence_mask_ratios:
            if not 0 < sequence_mask_ratio < 0.5:
                raise ValueError(
                    'All sequence masking ratios must be between 0.0 and 0.5.')

            if prefix_function is not None:
                prefix_tokens = prefix_function(sequence_mask_ratio, None,
                                                self.tokenizer)
            else:
                prefix_tokens = None

            kwargs = {
                'mean_span_length': None,
                'mask_ratio': sequence_mask_ratio,
                'prefix_tokens': prefix_tokens,
            }
            self._noisers.append(kwargs)

        if not self._noisers:
            raise ValueError(
                'No denoising tasks were included. Make sure to set '
                '`span_mean_lengths_and_ratios` and/or `sequence_mask_ratios`.')

    def __call__(self, examples: List[Dict[str,
                                           Any]]) -> Dict[str, torch.Tensor]:
        """Batch examples processed by the span corrupter."""
        processed_examples = []
        for example in examples:
            # Randomly pick a "noiser" to apply to this example
            noiser = random.choice(self._noisers)
            # Apply it
            processed_examples.append(
                noise_token_sequence(
                    example,
                    mask_ratio=noiser['mask_ratio'],
                    mean_span_length=noiser['mean_span_length'],
                    prefix_tokens=noiser['prefix_tokens'],
                    max_seq_length=self.max_seq_length,
                    tokenizer=self.tokenizer,
                    sentinel_token_ids=self.sentinel_token_ids,
                    decoder_only_format=self.decoder_only_format))
        batch = self.tokenizer.pad(processed_examples)

        # Truncate portions of the inputs that are purely padding
        # (up to a multiple of 8)
        multiple_of = 8
        n_examples_per_length = batch['attention_mask'].sum(0)
        keep_tokens = torch.sum(n_examples_per_length > 0)
        keep_tokens = int(multiple_of * torch.ceil(keep_tokens / multiple_of))

        # Note: EncDec formatting will always produce a right-padded batch
        if self.tokenizer.padding_side == 'left' and self.decoder_only_format:
            batch['input_ids'] = batch['input_ids'][:, -keep_tokens:]
            batch['attention_mask'] = batch['attention_mask'][:, -keep_tokens:]
        else:
            batch['input_ids'] = batch['input_ids'][:, :keep_tokens]
            batch['attention_mask'] = batch['attention_mask'][:, :keep_tokens]

        if self.decoder_only_format:
            if self.tokenizer.padding_side == 'left':
                batch['labels'] = batch['labels'][:, -keep_tokens:]
                batch['bidirectional_mask'] = batch[
                    'bidirectional_mask'][:, -keep_tokens:]
            else:
                batch['labels'] = batch['labels'][:, :keep_tokens]
                batch['bidirectional_mask'] = batch[
                    'bidirectional_mask'][:, :keep_tokens]

        else:
            # Truncate portions of the decoder inputs that are purely padding
            n_examples_per_length = batch['decoder_attention_mask'].sum(0)
            keep_tokens = torch.sum(n_examples_per_length > 0)
            keep_tokens = int(multiple_of *
                              torch.ceil(keep_tokens / multiple_of))

            batch['labels'] = batch['labels'][:, :keep_tokens]
            batch['decoder_attention_mask'] = batch[
                'decoder_attention_mask'][:, :keep_tokens]

        # This slicing can produce non-contiguous tensors, so use .contiguous
        # to prevent related problems
        batch = {k: v.contiguous() for k, v in batch.items()}

        return batch


def build_text_denoising_dataloader(cfg: DictConfig,
                                    device_batch_size: int) -> DataLoader:
    assert cfg.name == 'text_denoising', f'Tried to build_denoising text dataloader with cfg.name={cfg.name}'
    dataset = StreamingTextDataset(
        local=cfg.dataset.local,
        tokenizer_name=cfg.dataset.tokenizer_name,
        max_seq_len=cfg.dataset.max_seq_len,
        group_method=cfg.dataset.group_method,
        remote=cfg.dataset.get('remote'),
        split=cfg.dataset.get('split'),
        shuffle=cfg.dataset.get('shuffle', False),
        predownload=cfg.dataset.get('predownload', 100_000),
        keep_zip=cfg.dataset.get('keep_zip', False),
        download_retry=cfg.dataset.get('download_retry', 2),
        download_timeout=cfg.dataset.get('download_timeout', 60),
        validate_hash=cfg.dataset.get('validate_hash'),
        shuffle_seed=cfg.dataset.get('shuffle_seed'),
        num_canonical_nodes=cfg.dataset.get('num_canonical_nodes'),
        batch_size=device_batch_size)

    collate_fn = MixtureOfDenoisersCollator(
        tokenizer=dataset.tokenizer,
        max_seq_length=cfg.dataset.max_seq_len,
        decoder_only_format=cfg.mixture_of_denoisers.decoder_only_format,
        span_mean_lengths_and_ratios=cfg.mixture_of_denoisers.get(
            'span_mean_lengths_and_ratios'),
        sequence_mask_ratios=cfg.mixture_of_denoisers.get(
            'sequence_mask_ratios'),
        prefix_function=cfg.mixture_of_denoisers.get('prefix_function',
                                                     ul2_prefix_function))

    return DataLoader(
        dataset,
        collate_fn=collate_fn,
        batch_size=device_batch_size,
        drop_last=cfg.drop_last,
        num_workers=cfg.num_workers,
        pin_memory=cfg.get('pin_memory', True),
        prefetch_factor=cfg.get('prefetch_factor', 2),
        persistent_workers=cfg.get('persistent_workers', False),
        timeout=cfg.get('timeout', 0),
    )


def noise_token_sequence(
    example: Mapping[str, Any],
    mask_ratio: float,
    mean_span_length: Optional[float],
    prefix_tokens: Optional[Sequence[int]],
    max_seq_length: int,
    tokenizer: Tokenizer,
    sentinel_token_ids: Sequence[int],
    decoder_only_format: bool,
) -> Dict[str, torch.Tensor]:
    """Span corruption applicable to all UL2 denoising tasks."""
    # Extract the raw text tokens (trim if we need to)
    length = sum(example['attention_mask'])
    if length > max_seq_length:
        length = max_seq_length
    if tokenizer.padding_side == 'left':
        tokens = example['input_ids'][-length:]
    else:
        tokens = example['input_ids'][:length]

    prefix_tokens = prefix_tokens or []

    # mean_span_length==None is a special case for "sequential" denoising
    # (where a single span at the end of the sequence is masked)
    if mean_span_length is None:
        # This ensures that exactly 1 span will be produced and that
        # trimming to max_seq_length won't cut off the sentinel and <EOS>
        min_span_length = np.maximum(
            1, length + len(prefix_tokens) + 2 - max_seq_length)
        max_span_length = np.maximum(
            min_span_length, np.minimum(length - 1, 2 * mask_ratio * length))
        mean_span_length = np.floor(
            np.random.uniform(low=min_span_length, high=max_span_length))
        mask_ratio = mean_span_length / length
        use_sentinels = False
    else:
        use_sentinels = True

    # Generate the mask
    # Note: this function can be used for all the UL2 noising functions
    mask = _sample_mask_array(length, mask_ratio, mean_span_length)
    # The sequence should always be unmasked at the beginning
    assert mask[0] == 0

    # Generate the input/label sequences given the raw tokens and the mask
    tokens_inputs = _apply_mask(tokens, mask, use_sentinels,
                                tokenizer.eos_token_id, sentinel_token_ids)
    tokens_labels = _apply_mask(tokens, 1 - mask, use_sentinels,
                                tokenizer.eos_token_id, sentinel_token_ids)

    # Tag the inputs with any prefix
    if prefix_tokens:
        tokens_inputs = np.concatenate([prefix_tokens, tokens_inputs])

    # Trim if necessary
    if len(tokens_inputs) > max_seq_length:
        tokens_inputs = tokens_inputs[:max_seq_length]
    if len(tokens_labels) > max_seq_length:
        tokens_labels = tokens_labels[:max_seq_length]

    tokens_inputs = torch.LongTensor(tokens_inputs)
    tokens_labels = torch.LongTensor(tokens_labels)

    if decoder_only_format:
        return _format_tokens_for_decoder_only(tokens_inputs, tokens_labels,
                                               max_seq_length * 2,
                                               tokenizer.pad_token_id,
                                               tokenizer.padding_side)
    return _format_tokens_for_encoder_decoder(tokens_inputs, tokens_labels,
                                              max_seq_length,
                                              tokenizer.pad_token_id)


def _sample_mask_array(length: int, mask_ratio: float,
                       mean_span_length: float) -> np.ndarray:
    """Samples a span corruption mask."""
    if mask_ratio == 0.0:
        return np.zeros(length)
    # This first block computes the number of noise/non-noise spans and the
    # total tokens in each. Extra steps are taken to handle edge cases that
    # cause degeneracy.
    starting_length = length
    length = np.maximum(length, 2)
    num_noise_tokens = int(np.round(mask_ratio * float(length)))
    num_noise_tokens = np.minimum(np.maximum(num_noise_tokens, 1), length - 1)
    num_spans = int(np.round(float(num_noise_tokens) / mean_span_length))
    num_noise_spans = np.maximum(num_spans, 1)
    num_nonnoise_tokens = length - num_noise_tokens

    # Sample the noise/non-noise span lengths and interleave them to
    # generate the mask array.
    # Note: We always start with a non-noise span.
    def _sample_span_lengths(total_tokens: int, num_spans: int) -> np.ndarray:
        """Samples lengths of num_spans segments.

        Note: the combined length of segments equals total_tokens.
        """
        span_markers = np.less(np.arange(total_tokens - 1), num_spans -
                               1)[np.random.permutation(total_tokens - 1)]
        span_start_indicator = np.concatenate([[0], span_markers])
        span_id = np.cumsum(span_start_indicator).reshape(-1, 1)
        spans = np.arange(num_spans).reshape(1, -1)
        span_lengths = np.sum(span_id == spans, axis=0)
        return span_lengths

    noise_span_lengths = _sample_span_lengths(num_noise_tokens, num_noise_spans)
    nonnoise_span_lengths = _sample_span_lengths(num_nonnoise_tokens,
                                                 num_noise_spans)
    interleaved_span_lengths = np.reshape(
        np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1),
        [num_noise_spans * 2])

    span_starts = np.cumsum(interleaved_span_lengths)[:-1]
    span_start_indicator = np.zeros(length)
    span_start_indicator[span_starts] = 1
    span_id = np.cumsum(span_start_indicator)
    is_noise = np.equal(np.mod(span_id, 2), 1)

    mask = is_noise[:starting_length]

    return mask


def _apply_mask(tokens: Sequence[int], mask: np.ndarray, use_sentinels: bool,
                eos_token_id: int,
                sentinel_token_ids: np.ndarray) -> np.ndarray:
    """Remove or replace masked portions from token sequence."""
    if not use_sentinels:
        # The logic is simple if we don't use sentinel tokens
        noised_tokens = np.array(tokens)[np.logical_not(mask)]

        # Ensure there's an end-of-sentence token at the end
        if noised_tokens[-1] != eos_token_id:
            noised_tokens = np.concatenate([noised_tokens, [eos_token_id]])

        return noised_tokens

    # Masking at previous token
    prev_token_mask = np.concatenate([[0], mask[:-1]])

    # Decompose mask into start-of-span mask and non-start-of-span mask
    start_of_noise_span_token = np.logical_and(mask,
                                               np.logical_not(prev_token_mask))
    nonstart_noise_span_token = np.logical_and(mask, prev_token_mask)

    # Replace tokens at the start of each noise span with its corresponding
    # sentinel token
    tokens = np.where(
        start_of_noise_span_token,
        sentinel_token_ids[np.cumsum(start_of_noise_span_token) - 1], tokens)

    # Remove masked tokens (but preserving the sentinel tokens)
    noised_tokens = tokens[np.logical_not(nonstart_noise_span_token)]

    # Ensure there's an end-of-sentence token at the end
    if noised_tokens[-1] != eos_token_id:
        noised_tokens = np.concatenate([noised_tokens, [eos_token_id]])
    return noised_tokens


def _format_tokens_for_encoder_decoder(
    tokens_inputs: torch.LongTensor,
    tokens_labels: torch.LongTensor,
    max_seq_length: int,
    pad_token_id: int,
) -> Dict[str, torch.Tensor]:
    """Package the input/label sequence for an EncDec model."""
    example = {}
    # Re-populate with an empty, padded example
    example['input_ids'] = torch.full((max_seq_length,),
                                      pad_token_id,
                                      dtype=torch.int32)
    example['labels'] = torch.full((max_seq_length,),
                                   _HF_IGNORE_INDEX,
                                   dtype=torch.int32)
    example['attention_mask'] = torch.zeros_like(example['input_ids'])
    example['decoder_attention_mask'] = torch.zeros_like(example['labels'])

    # Fill in with processed results (Note: EncDec format is right-padded)
    example['input_ids'][:len(tokens_inputs)] = tokens_inputs
    example['labels'][:len(tokens_labels)] = tokens_labels
    example['attention_mask'][:len(tokens_inputs)] = 1
    example['decoder_attention_mask'][:len(tokens_labels)] = 1
    return example


def _format_tokens_for_decoder_only(
    tokens_inputs: torch.LongTensor,
    tokens_labels: torch.LongTensor,
    max_seq_length: int,
    pad_token_id: int,
    padding_side: str,
) -> Dict[str, torch.Tensor]:
    """Package the input/label sequence for an decoder-only model."""
    example = {}
    # Re-populate with an empty, padded example
    example['input_ids'] = torch.full((max_seq_length,),
                                      pad_token_id,
                                      dtype=torch.int32)
    example['labels'] = torch.full((max_seq_length,),
                                   _HF_IGNORE_INDEX,
                                   dtype=torch.int32)
    example['attention_mask'] = torch.full((max_seq_length,),
                                           0,
                                           dtype=torch.bool)
    example['bidirectional_mask'] = torch.full((max_seq_length,),
                                               0,
                                               dtype=torch.bool)

    n_input = len(tokens_inputs)
    n_label = len(tokens_labels)
    n_concat = n_input + n_label
    assert n_concat <= max_seq_length

    tokens_concat = torch.concat([tokens_inputs, tokens_labels], dim=0)

    # Fill in with the processed results
    if padding_side == 'left':
        example['input_ids'][-n_concat:] = tokens_concat
        # `labels` copies `input_ids` but with -100 at
        # non-loss-generating tokens. `labels` will be shifted in the
        # model code when computing loss.
        example['labels'][-n_concat:] = tokens_concat
        example['labels'][-n_concat:-n_label] = _HF_IGNORE_INDEX
        example['attention_mask'][-n_concat:] = 1
        example['bidirectional_mask'][-n_concat:-n_label] = 1
    else:
        example['input_ids'][:n_concat] = tokens_concat
        # See above comment regarding `labels`
        example['labels'][:n_concat] = tokens_concat
        example['labels'][:n_input] = _HF_IGNORE_INDEX
        example['attention_mask'][:n_concat] = 1
        example['bidirectional_mask'][:n_input] = 1
    return example


# Helpful to test if your dataloader is working locally
# Run `python data.py [remote] [local, optional]` and verify that batches
# are printed out
if __name__ == '__main__':
    remote = sys.argv[1]
    if len(sys.argv) > 2:
        local = sys.argv[2]
    else:
        local = remote
    print(f'Reading val split from {remote} -> {local}')

    cfg = {
        'name': 'text_denoising',
        'dataset': {
            'local': local,
            'remote': remote,
            'split': 'val',
            'shuffle': False,
            'tokenizer_name': 't5-base',
            'max_seq_len': 128,
            'group_method': 'concat',
            'keep_zip': True,  # in case we need compressed files after testing
        },
        'drop_last': False,
        'num_workers': 4,
    }
    cfg = om.create(cfg)
    device_batch_size = 2

    loader = build_text_denoising_dataloader(cfg, device_batch_size)
    tokenizer = loader.dataset.tokenizer
    for batch_ix, batch in enumerate(islice(loader, 5)):
        print('\n')
        print('#' * 20, f'Batch {batch_ix}', '#' * 20)
        for k, v in batch.items():
            print(k, v.shape, v.dtype)
        for sample_ix, token_sample in enumerate(batch['input_ids']):
            labels = batch['labels'][sample_ix]
            attn_inputs = batch['attention_mask'][sample_ix].to(torch.bool)
            attn_labels = batch['decoder_attention_mask'][sample_ix].to(
                torch.bool)
            print('-' * 20, f' Sample {sample_ix} ', '-' * 20)
            print('Input:  ', tokenizer.decode(token_sample[attn_inputs]))
            print('Target: ', tokenizer.decode(labels[attn_labels]))
