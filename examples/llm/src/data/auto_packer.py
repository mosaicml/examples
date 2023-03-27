# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

from typing import Callable, Dict, List, Literal, Tuple

import torch


class PackWrapper:

    def __init__(self,
                 collator: Callable,
                 target_batch_size: int,
                 max_seq_len: int,
                 pad_token_id: int,
                 padding_side: Literal['left', 'right'] = 'right'):
        self.base_collator = collator
        self.out_size = int(target_batch_size)
        self.max_seq_len = int(max_seq_len)
        self.pad_token_id = int(pad_token_id)
        assert self.out_size > 0
        assert self.max_seq_len > 0
        assert self.pad_token_id > 0

        self.padding_side = padding_side

        self.n_packed_tokens = 0
        self.n_total_tokens = 0

    @property
    def waste(self):
        return 1 - (self.n_packed_tokens / self.n_total_tokens)

    def __call__(
            self,
            examples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        batch = self.base_collator(examples)

        assert 'attention_mask' in batch
        assert 'input_ids' in batch

        for key in batch.keys():
            assert key in [
                'input_ids',
                'labels',
                'attention_mask',
                'bidirectional_mask',
            ]

        # Cut everything down to size
        sizes_and_examples = [
            extract_trim_batch_idx(batch, idx)
            for idx in range(batch['attention_mask'].shape[0])
        ]

        # Pack greedily
        packed_examples, n_packed_tokens, n_total_tokens = greedy_pack(
            sizes_and_examples,
            out_size=self.out_size,
            max_seq_len=self.max_seq_len)
        self.n_packed_tokens += n_packed_tokens
        self.n_total_tokens += n_total_tokens

        # Re-pad to max_seq_len and batch
        batch = repad(packed_examples,
                      max_seq_len=self.max_seq_len,
                      pad_token_id=self.pad_token_id,
                      padding_side=self.padding_side)
        return batch


def extract_trim_batch_idx(batch: Dict[str, torch.Tensor], idx: int):
    example = {k: v[idx] for k, v in batch.items()}

    keep = example['attention_mask'] == 1
    size = keep.sum()
    trim_example = {k: v[keep] for k, v in example.items()}
    trim_example['sequence_id'] = torch.zeros_like(trim_example['input_ids'])

    return size, trim_example


def greedy_pack(sizes_and_examples: List[Tuple[int, Dict[str, torch.Tensor]]],
                out_size: int, max_seq_len: int):
    sorted_sizes_and_examples = sorted(sizes_and_examples, key=lambda x: x[0])
    assert len(sorted_sizes_and_examples) >= out_size
    if len(sorted_sizes_and_examples) == out_size:
        return sorted_sizes_and_examples

    def combine_in_place(example: Dict[str, torch.Tensor],
                         add_on: Dict[str, torch.Tensor]):
        if 'labels' in add_on:
            # Prevents the last token in example from being trained to
            # predict the first token in add_on, which would make no sense.
            add_on['labels'][0] = -100

        for k in example.keys():
            if k == 'sequence_id':
                example[k] = torch.cat(
                    [example[k], add_on[k] + 1 + torch.max(example[k])])
            else:
                example[k] = torch.cat([example[k], add_on[k]])
        return example

    packed_sizes_and_examples = []
    n_sorted = len(sorted_sizes_and_examples)
    n_packed = 0
    while n_sorted + n_packed > out_size and n_sorted > 1:
        shortest_length = sorted_sizes_and_examples[0][0]
        longest_length = sorted_sizes_and_examples[-1][0]
        if shortest_length + longest_length <= max_seq_len:
            # Remove shortest examples from the list
            shortest_length, shortest_example = sorted_sizes_and_examples.pop(0)
            n_sorted -= 1
            # And pack it into the longest example
            longest_length, longest_example = sorted_sizes_and_examples.pop(-1)
            longest_example = combine_in_place(longest_example,
                                               shortest_example)
            longest_length += shortest_length
            assert longest_length <= max_seq_len
            sorted_sizes_and_examples.append((longest_length, longest_example))
        else:
            # Can't pack any more into the longest, so remove it from this list
            packed_sizes_and_examples.append(sorted_sizes_and_examples.pop(-1))
            n_sorted -= 1
            n_packed += 1

    combined = sorted(packed_sizes_and_examples + sorted_sizes_and_examples,
                      key=lambda x: x[0])
    packed_examples = [example for _, example in combined][-out_size:]

    tokens = [tokens for tokens, _ in combined]
    n_total_tokens = sum(tokens)
    n_packed_tokens = sum(tokens[-out_size:])

    return packed_examples, n_packed_tokens, n_total_tokens


def repad(packed_examples: List[Dict[str, torch.Tensor]], max_seq_len: int,
          pad_token_id: int,
          padding_side: Literal['left', 'right']) -> Dict[str, torch.Tensor]:

    def pad_tensor(tensor: torch.Tensor, pad_value: int):
        if len(tensor) == max_seq_len:
            return tensor
        t = torch.full((max_seq_len,),
                       pad_value,
                       dtype=tensor.dtype,
                       device=tensor.device)
        if padding_side == 'left':
            t[-len(tensor):] = tensor
        elif padding_side == 'right':
            t[:len(tensor)] = tensor
        else:
            raise ValueError(f'Unknown {padding_side=}')
        return t

    pad_vals = {
        'input_ids': pad_token_id,
        'labels': -100,
        'attention_mask': 0,
        'bidirectional_mask': 0,
        'sequence_id': -1,
    }
    keys = packed_examples[0].keys()
    batch = {}
    for key in keys:
        batch[key] = torch.stack([
            pad_tensor(example[key], pad_vals[key])
            for example in packed_examples
        ])
    return batch
