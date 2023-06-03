# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

import os

from composer.utils import maybe_create_object_store_from_uri, parse_uri


def download_model(remote_uri: str):
    # files = [
    #     'config.json',
    #     'pytorch_model.bin',
    #     'special_tokens_map.json',
    #     'tokenizer_config.json',
    #     'tokenizer.json',
    #     'generation_config.json',
    #     'merges.txt',
    #     'vocab.json',
    # ]
    os.makedirs('/downloaded_hf_checkpoint/', exist_ok=True)
    # for file in files:
    #     get_file(
    #         path=os.path.join(remote_uri, file),
    #         destination=os.path.join('/downloaded_hf_checkpoint/', file),
    #     )

    object_store = maybe_create_object_store_from_uri(remote_uri)
    _, _, key = parse_uri(remote_uri)
    object_store.download_object(
        object_name=key,
        file_name='/downloaded_hf_checkpoint/config.json',
    )
    # get_file('oci://mosaicml-internal-checkpoints/daniel/checkpoints/sec-finetune-neo-125-3-8UEv7I/ep1-ba367-rank0/config.json', '/downloaded_hf_checkpoint/config.json')
