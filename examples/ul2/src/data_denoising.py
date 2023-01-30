# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Streaming dataloader for (mixture of) denoising task(s)."""

import logging
import random
import sys
from itertools import islice
from typing import Any, Dict, List, Mapping, Optional, Union

import numpy as np
import torch
from omegaconf import DictConfig
from omegaconf import OmegaConf as om
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from examples.common.text_data import StreamingTextDataset
from examples.ul2.src import utils

log = logging.getLogger(__name__)


class MixtureOfDenoisersCollator:
    """Data collator for mixture of span-corruption denoisers, as in UL2.

    TODO(Alex): Complete docstring.
    """

    def __init__(
        self,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        max_seq_length: int,
        decoder_only_format: bool = False,
        span_mean_lengths_and_ratios: Optional[Union[List[List[float]],
                                                     List[float]]] = None,
        sequence_mask_ratios: Optional[Union[List[float], float]] = None,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.decoder_only_format = decoder_only_format

        # Prepare the tokenizer for denoising tasks
        n_sentinels = 100
        utils.adapt_tokenizer_for_denoising(self.tokenizer,
                                            num_sentinel_tokens=n_sentinels)
        self.sentinel_token_ids = np.array(
            tokenizer(''.join([f'<extra_id_{i}>' for i in range(n_sentinels)
                              ])).input_ids)

        self._denoiser_tags = [
            '[NLU]',  # "Regular" span corruption
            '[NLG]',  # "Extreme" span corruption
            '[S2S]',  # Sequential denoising
        ]
        self._denoiser_tag_token_ids = {
        }  # To be prepended to `input_ids` when corrupting
        for tag in self._denoiser_tags:
            self._denoiser_tag_token_ids[tag] = self.tokenizer(
                tag, add_special_tokens=False).input_ids

        self._noisers = []

        # Add "noisers" for the span corruption denoising task
        self.span_mean_lengths_and_ratios = span_mean_lengths_and_ratios
        if self.span_mean_lengths_and_ratios is None:
            self.span_mean_lengths_and_ratios = []
        elif isinstance(self.span_mean_lengths_and_ratios[0], (int, float)):
            assert len(
                self.span_mean_lengths_and_ratios
            ) == 2, '`span_mean_lengths_and_ratios` must be a pair of [mean_length, mask_ratio] or a list of such pairs.'
            self.span_mean_lengths_and_ratios = [
                self.span_mean_lengths_and_ratios
            ]
        for span_mean_length, span_mask_ratio in self.span_mean_lengths_and_ratios:
            assert span_mean_length > 0, 'All span mean lengths must be positive.'
            assert 0 < span_mask_ratio < 1.0, 'All span masking ratios must be between 0.0 and 1.0.'

            # This mean_length / mask_ratio combo becomes one of the span corruption denoising tasks
            if span_mean_length >= 12 or span_mask_ratio >= 0.3:
                prefix = '[NLG]'  # UL2 considers this corruption rate "extreme" and tags it accordingly
            else:
                prefix = '[NLU]'  # UL2 considers this corruption rate "regular" and tags it accordingly
            noiser = (self.noise_token_sequence, {
                'mean_span_length': span_mean_length,
                'mask_ratio': span_mask_ratio,
                'prefix': prefix
            })
            self._noisers.append(noiser)

        # Add "noisers" for the sequential denoising task
        self.sequence_mask_ratios = sequence_mask_ratios
        if self.sequence_mask_ratios is None:
            self.sequence_mask_ratios = []
        elif isinstance(self.sequence_mask_ratios, float):
            self.sequence_mask_ratios = [self.sequence_mask_ratios]
        for sequence_mask_ratio in self.sequence_mask_ratios:
            assert 0 < sequence_mask_ratio < 0.5, 'All sequence masking ratios must be between 0.0 and 0.5.'

            # This mask_ratio becomes one of the sequential denoising tasks
            noiser = (self.noise_token_sequence, {
                'mean_span_length': None,
                'mask_ratio': span_mask_ratio,
                'prefix': '[S2S'
            })
            self._noisers.append(noiser)

        if not self._noisers:
            raise ValueError(
                'No denoising tasks were included. Make sure to set `span_mean_lengths_and_ratios` and/or `sequence_mask_ratios`.'
            )

    @staticmethod
    def _sample_mask_array(length: int, mask_ratio: float,
                           mean_span_length: float) -> np.ndarray:
        if mask_ratio == 0.0:
            return np.zeros(length)
        # This first block computes the number of noise/non-noise spans and the total tokens in each.
        # Extra steps are taken to handle edge cases that cause degeneracy.
        starting_length = length
        length = np.maximum(length, 2)
        num_noise_tokens = int(np.round(mask_ratio * float(length)))
        num_noise_tokens = np.minimum(np.maximum(num_noise_tokens, 1),
                                      length - 1)
        num_spans = int(np.round(float(num_noise_tokens) / mean_span_length))
        num_noise_spans = np.maximum(num_spans, 1)
        num_nonnoise_tokens = length - num_noise_tokens

        # Sample the noise/non-noise span lengths and interleave them to generate the mask array.
        # Note: We always start with a non-noise span.
        def _sample_span_lengths(total_tokens: int, num_spans: int):
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

        noise_span_lengths = _sample_span_lengths(num_noise_tokens,
                                                  num_noise_spans)
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

    def apply_mask(self, tokens: List, mask: np.ndarray,
                   use_sentinels: bool) -> np.ndarray:
        if not use_sentinels:
            # The logic is simple if we do not mark replaced spans with sentinel tokens
            noised_tokens = np.array(tokens)[np.logical_not(mask)]

            # Ensure there's an end-of-sentence token at the end
            if noised_tokens[-1] != self.tokenizer.eos_token_id:
                noised_tokens = np.concatenate(
                    [noised_tokens, [self.tokenizer.eos_token_id]])

            return noised_tokens

        # Masking at previous token
        prev_token_mask = np.concatenate([[0], mask[:-1]])

        # Decompose mask into start-of-span mask and non-start-of-span mask
        start_of_noise_span_token = np.logical_and(
            mask, np.logical_not(prev_token_mask))
        nonstart_noise_span_token = np.logical_and(mask, prev_token_mask)

        # Replace tokens at the start of each noise span with its corresponding sentinel token
        tokens = np.where(
            start_of_noise_span_token,
            self.sentinel_token_ids[np.cumsum(start_of_noise_span_token) - 1],
            tokens)

        # Remove masked tokens (but preserving the sentinel tokens)
        noised_tokens = tokens[np.logical_not(nonstart_noise_span_token)]

        # Ensure there's an end-of-sentence token at the end
        if noised_tokens[-1] != self.tokenizer.eos_token_id:
            noised_tokens = np.concatenate(
                [noised_tokens, [self.tokenizer.eos_token_id]])
        return noised_tokens

    def noise_token_sequence(self, example: Mapping[str,
                                                    Any], mask_ratio: float,
                             mean_span_length: Optional[float],
                             prefix: Optional[str]) -> Dict[str, torch.Tensor]:
        """Span corruption applicable to all UL2 denoising tasks."""
        # Extract the raw text tokens (trim if we need to)
        length = sum(example['attention_mask'])
        if length > self.max_seq_length:
            length = self.max_seq_length
        if self.tokenizer.padding_side == 'left':
            tokens = example['input_ids'][-length:]
        else:
            tokens = example['input_ids'][:length]

        # Figure out if there are any prefix tokens to handle
        if prefix is not None:
            if prefix not in self._denoiser_tag_token_ids:
                raise KeyError(
                    f"Prefix {prefix} is not valid. Must be one of: {', '.join(self._denoiser_tag_token_ids.keys())}"
                )
            prefix_tokens = self._denoiser_tag_token_ids[prefix]
        else:
            prefix_tokens = []

        # mean_span_length==None is a special case for "sequential" denoising (where a single span at the end of the sequence is masked)
        if mean_span_length is None:
            # This ensures that exactly 1 span will be produced and that trimming to max_seq_length will not cut off the sentinel and <EOS>
            min_span_length = np.maximum(
                1, length + len(prefix_tokens) + 2 - self.max_seq_length)
            max_span_length = np.maximum(
                min_span_length, np.minimum(length - 1,
                                            2 * mask_ratio * length))
            mean_span_length = np.floor(
                np.random.uniform(low=min_span_length, high=max_span_length))
            mask_ratio = mean_span_length / length
            use_sentinels = False
        else:
            use_sentinels = True

        # Generate the mask
        mask = self._sample_mask_array(
            length, mask_ratio, mean_span_length
        )  # This function can be used for all the UL2 noising functions
        assert mask[
            0] == 0  # The sequence should always be unmasked at the beginning

        # Generate the input/label sequences given the raw tokens and the mask
        tokens_inputs = self.apply_mask(tokens, mask, use_sentinels)
        tokens_labels = self.apply_mask(tokens, 1 - mask, use_sentinels)

        # Tag the inputs with any prefix
        if prefix is not None:
            tokens_inputs = np.concatenate([prefix_tokens, tokens_inputs])

        # Trim if necessary
        if len(tokens_inputs) > self.max_seq_length:
            tokens_inputs = tokens_inputs[:self.max_seq_length]
        if len(tokens_labels) > self.max_seq_length:
            tokens_labels = tokens_labels[:self.max_seq_length]

        tokens_inputs = torch.LongTensor(tokens_inputs)
        tokens_labels = torch.LongTensor(tokens_labels)

        if self.decoder_only_format:
            return self._populate_decoder_only(tokens_inputs, tokens_labels)
        else:
            return self._populate_encoder_decoder(tokens_inputs, tokens_labels)

    def _populate_encoder_decoder(
            self, tokens_inputs: torch.LongTensor,
            tokens_labels: torch.LongTensor) -> Dict[str, torch.Tensor]:
        example = {}
        # Re-populate with an empty, padded example
        example['input_ids'] = torch.full((self.max_seq_length,),
                                          self.tokenizer.pad_token_id,
                                          dtype=torch.int32)
        example['labels'] = torch.full(
            (self.max_seq_length,), -100, dtype=torch.int32
        )  # Note: The HF code is hardcoded to use -100 as the ignore index
        example['attention_mask'] = torch.zeros_like(example['input_ids'])
        example['decoder_attention_mask'] = torch.zeros_like(example['labels'])

        # Fill in with the processed results (Note: EncDec format will be right-padded)
        example['input_ids'][:len(tokens_inputs)] = tokens_inputs
        example['labels'][:len(tokens_labels)] = tokens_labels
        example['attention_mask'][:len(tokens_inputs)] = 1
        example['decoder_attention_mask'][:len(tokens_labels)] = 1
        return example

    def _populate_decoder_only(
            self, tokens_inputs: torch.LongTensor,
            tokens_labels: torch.LongTensor) -> Dict[str, torch.Tensor]:
        example = {}
        # Re-populate with an empty, padded example
        example['input_ids'] = torch.full((self.max_seq_length * 2,),
                                          self.tokenizer.pad_token_id,
                                          dtype=torch.int32)
        example['labels'] = torch.full(
            (self.max_seq_length * 2,), -100, dtype=torch.int32
        )  # Note: -100 is often hardcoded as the ignore index
        example['attention_mask'] = torch.full((self.max_seq_length * 2,),
                                               0,
                                               dtype=torch.bool)
        example['bidirectional_mask'] = torch.full((self.max_seq_length * 2,),
                                                   0,
                                                   dtype=torch.bool)

        n_input = len(tokens_inputs)
        n_label = len(tokens_labels)
        n_concat = n_input + n_label
        assert n_concat <= self.max_seq_length * 2

        tokens_concat = torch.concat([tokens_inputs, tokens_labels], dim=0)

        # Fill in with the processed results
        if self.tokenizer.padding_side == 'left':
            example['input_ids'][-n_concat:] = tokens_concat
            # (Here `labels` copies `input_ids` but with -100 at non-loss-generating tokens. `labels` will be shifted in the model code when computing loss.)
            example['labels'][-n_concat:] = tokens_concat
            example['labels'][-n_concat:-n_label] = -100
            example['attention_mask'][-n_concat:] = 1
            example['bidirectional_mask'][-n_concat:-n_label] = 1
        else:
            example['input_ids'][:n_concat] = tokens_concat
            # (Here `labels` copies `input_ids` but with -100 at non-loss-generating tokens. `labels` will be shifted in the model code when computing loss.)
            example['labels'][:n_concat] = tokens_concat
            example['labels'][:n_input] = -100
            example['attention_mask'][:n_concat] = 1
            example['bidirectional_mask'][:n_input] = 1
        return example

    def __call__(self, examples: List[Dict[str,
                                           Any]]) -> Dict[str, torch.Tensor]:
        """Batch examples processed by the span corrupter."""
        processed_examples = []
        for example in examples:
            noiser_fcn, noiser_kwargs = random.choice(self._noisers)
            processed_examples.append(noiser_fcn(example, **noiser_kwargs))
        batch = self.tokenizer.pad(processed_examples)

        # Truncate portions of the inputs that are purely padding (up to a multiple of 8)
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

        # This slicing can produce non-contiguous tensors, so use .contiguous to prevent related problems
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
        remote=cfg.dataset.get('remote', None),
        split=cfg.dataset.get('split', None),
        shuffle=cfg.dataset.get('shuffle', False),
        predownload=cfg.dataset.get('predownload', 100_000),
        keep_zip=cfg.dataset.get('keep_zip', False),
        download_retry=cfg.dataset.get('download_retry', 2),
        download_timeout=cfg.dataset.get('download_timeout', 60),
        validate_hash=cfg.dataset.get('validate_hash', None),
        shuffle_seed=cfg.dataset.get('shuffle_seed', None),
        num_canonical_nodes=cfg.dataset.get('num_canonical_nodes', None),
        batch_size=device_batch_size)

    collate_fn = MixtureOfDenoisersCollator(
        tokenizer=dataset.tokenizer,
        max_seq_length=cfg.dataset.max_seq_len,
        decoder_only_format=cfg.mixture_of_denoisers.get(
            'decoder_only_format', False),
        span_mean_lengths_and_ratios=cfg.mixture_of_denoisers.get(
            'span_mean_lengths_and_ratios', None),
        sequence_mask_ratios=cfg.mixture_of_denoisers.get(
            'sequence_mask_ratios', None),
    )

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


# Helpful to test if your dataloader is working locally
# Run `python data.py [remote] [local, optional]` and verify that batches are printed out
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
