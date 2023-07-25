# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

import os

from composer.utils import maybe_create_object_store_from_uri, parse_uri

# This api exists in llm-foundry repo
from scripts.inference.convert_hf_mpt_to_ft import convert_mpt_to_ft  # isort: skip # yapf: disable # type: ignore

LOCAL_BASE_FOLDER = '/downloaded_hf_checkpoint/'
# should match with LOCAL_CHECKPOINT_DIR in mpt_ft_handler.py
LOCAL_FT_FOLDER = '/tmp/mpt'


def download_and_convert(remote_uri: str, gpus: int = 1):
    """Helper function used to download the model at startup time of an.

    inference deployment.

    It is specifically written for MPT, and the file list would need to be adapted to use with
    a different model.

    Args:
        remote_uri (str): Object store prefix of the folder containing the model files.
        gpus (int): Number of gpus to use for inference
    """
    object_store = maybe_create_object_store_from_uri(remote_uri)
    assert object_store is not None  # pyright
    _, _, remote_base_key = parse_uri(remote_uri)

    # These files are hardcoded for MPT, and would need to be changed for a different model
    files = [
        'adapt_tokenizer.py', 'attention.py', 'blocks.py', 'config.json',
        'configuration_mpt.py', 'custom_embedding.py', 'flash_attn_triton.py', 'fc.py',
        'generation_config.json', 'hf_prefixlm_converter.py',
        'meta_init_context.py', 'modeling_mpt.py', 'norm.py',
        'param_init_fns.py', 'pytorch_model.bin', 'special_tokens_map.json',
        'tokenizer.json', 'tokenizer_config.json'
    ]
    os.makedirs(LOCAL_BASE_FOLDER, exist_ok=True)
    for file in files:
        object_store.download_object(
            object_name=os.path.join(remote_base_key, file),
            filename=os.path.join(LOCAL_BASE_FOLDER, file),
        )
    convert_mpt_to_ft(model_name_or_path=LOCAL_BASE_FOLDER,
                      output_dir=LOCAL_FT_FOLDER,
                      infer_gpu_num=gpus)