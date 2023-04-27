import os
import re
from typing import List, Optional

import torch
import torch.distributed as dist
import yaml
from pydantic import BaseModel

# local checkpoint params
LOCAL_CHECKPOINT_PATH = "/mosaicml/"
S3_MODEL_NAME = "s3_model"

# model config params
MODEL_CONFIG_FILE = "/mnt/model/model_config.yaml"


class HFModelConfig(BaseModel):
    """Typed dictionary for nested hugging face model configs"""
    task: str
    model_dtype: str = 'fp16'
    autocast_dtype: Optional[str] = None


class CustomModelConfig(BaseModel):
    """Typed dictionary for nested custom model configs"""
    model_handler: str


class ModelConfig(BaseModel):
    """Typed dictionary for model configs"""
    checkpoint_path: str
    gpu_num: int
    hf_model: Optional[HFModelConfig] = None
    custom_model: Optional[CustomModelConfig] = None


def get_world_size() -> int:
    if dist.is_initialized():
        return dist.get_world_size()
    else:
        cuda_visible_devices = get_cuda_visible_devices()

    return len(cuda_visible_devices)


def get_cuda_visible_devices() -> List[int]:
    cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES")
    if cuda_visible_devices is not None:
        cuda_visible_devices = list(map(int, cuda_visible_devices.split(",")))

    return cuda_visible_devices or []


def print_rank_n(*values, rank: int = 0) -> None:
    # print on only process with specified rank
    if dist.is_initialized():
        if dist.get_rank() == rank:
            print(*values)
    else:
        print(*values)


def get_base_ds_config(dtype: str):
    return {
        "replace_with_kernel_inject": True,
        "dtype": get_torch_dtype(dtype),
        "replace_method": "auto",
        "enable_cuda_graph": False,
        "tensor_parallel": {
            "tp_size": get_world_size()
        },
    }


def get_torch_dtype(dtype_str: str) -> torch.dtype:
    if dtype_str == "bf16":
        return torch.bfloat16
    elif dtype_str == "fp16":
        return torch.float16
    elif dtype_str == "int8":
        return torch.int8
    elif dtype_str == "fp32":
        return torch.float32


def is_s3_path(checkpoint_path) -> bool:
    return checkpoint_path.startswith("s3://")


def camel_case_to_snake_case(text: str) -> str:
    """Converts camelCase 🐪 to snake_case 🐍

    Args:
        text: camel case string to convert

    Returns:
        text in snake case

    Example
    >>> camel_case_to_snake_case("fooBar")
    "foo_bar"
    """
    text = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', text)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', text).lower()


def load_model_config() -> ModelConfig:
    with open(MODEL_CONFIG_FILE, "r") as stream:
        try:
            model_config = yaml.safe_load(stream)

            translated_config = {}
            for key, val in model_config.items():
                if key in {'hfModel', 'customModel'}:
                    translated_model_type = camel_case_to_snake_case(key)
                    translated_config[translated_model_type] = {}

                    for model_key, model_val in val.items():
                        translated_key = camel_case_to_snake_case(model_key)
                        translated_config[translated_model_type][
                            translated_key] = model_val
                else:
                    translated_key = camel_case_to_snake_case(key)
                    translated_config[translated_key] = val

            return ModelConfig.parse_obj(translated_config)
        except yaml.YAMLError as err:
            raise RuntimeError(
                f"Failed to read model config with error: {err}")
