# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

from transformers import BertConfig as TransformersBertConfig


class BertConfig(TransformersBertConfig):

    def __init__(
        self,
        alibi_starting_size=512,
        **kwargs,
    ):
        self.alibi_starting_size = alibi_starting_size
        super().__init__(**kwargs)
