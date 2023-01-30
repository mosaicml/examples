# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

from typing import Union

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast


def adapt_tokenizer_for_denoising(tokenizer: Union[PreTrainedTokenizer,
                                                   PreTrainedTokenizerFast],
                                  num_sentinel_tokens: int = 100):
    """Adds sentinel tokens and padding token (if missing).

    Expands the tokenizer vocabulary to include sentinel tokens
    used in mixture-of-denoiser tasks as well as a padding token.

    All added tokens are added as special tokens. No tokens are
    added if sentinel tokens and padding token already exist.
    """
    # Add sentinel tokens (e.g., <extra_id_0>, <extra_id_1>, and so on). Has no effect if these are already in the vocab.
    assert 0 <= num_sentinel_tokens <= 1000
    sentinels_to_add = [f'<extra_id_{i}>' for i in range(num_sentinel_tokens)]
    tokenizer.add_tokens(sentinels_to_add, special_tokens=True)

    # If the padding token has not been set, add <pad> and use it
    if tokenizer.pad_token is None:
        tokenizer.add_tokens('<pad>', special_tokens=True)
        tokenizer.pad_token = '<pad>'
        assert tokenizer.pad_token_id is not None
