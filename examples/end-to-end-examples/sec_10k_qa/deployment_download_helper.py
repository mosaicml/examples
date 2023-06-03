# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

import os

from composer.utils import maybe_create_object_store_from_uri, parse_uri

LOCAL_BASE_FOLDER = '/downloaded_hf_checkpoint/'


def download_model(remote_uri: str):
    object_store = maybe_create_object_store_from_uri(remote_uri)
    _, _, remote_base_key = parse_uri(remote_uri)

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
    os.makedirs(LOCAL_BASE_FOLDER, exist_ok=True)
    for file in files:
        object_store.download_object(
            object_name=os.path.join(remote_base_key, file),
            filename=os.path.join(LOCAL_BASE_FOLDER, file),
        )
