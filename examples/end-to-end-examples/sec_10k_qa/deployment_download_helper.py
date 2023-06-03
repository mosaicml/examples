# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

import os

from composer.utils import get_file


def download_model(remote_uri: str):
    files = [
        'config.json',
        'pytorch_model.bin',
        'special_tokens_map.json',
        'tokenizer_config.json',
        'tokenizer.json',
        'generation_config.json',
        'merges.txt',
        'vocab.json',
    ]
    for file in files:
        get_file(
            path=os.path.join(remote_uri, file),
            destination=os.path.join('/downloaded_hf_checkpoint/', file),
        )
