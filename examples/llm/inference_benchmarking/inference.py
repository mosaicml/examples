# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

import sys
import time
import deepspeed
import torch
from composer.utils import reproducibility
from omegaconf import DictConfig
from omegaconf import OmegaConf as om

import gc
import math
import os
import time

import torch
import torch.distributed as dist

from transformers import AutoModelForCausalLM, AutoTokenizer



local_rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = torch.cuda.device_count()
print("World size " + world_size)
rank = local_rank


def print_rank0(*msg):
    if rank != 0:
        return
    print(*msg)





def get_world_size() -> int:
    if dist.is_initialized():
        return dist.get_world_size()
    else:
        return 1


# balanced_low_0 - because it allows a larger batch size with multiple GPUs

# model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)


# t_ready = time.time()


# ### Generate

# print_rank0(f"*** Starting to generate {num_tokens} tokens with bs={args.batch_size}")

# input_sentences = [
#     "DeepSpeed is a machine learning framework",
#     "He is working on",
#     "He has a",
#     "He got all",
#     "Everyone is happy and I can",
#     "The new movie that got Oscar this year",
#     "In the far far distance from our galaxy,",
#     "Peace is the only way",
# ]

# print_rank0(f"Generate args {generate_kwargs}")
# inputs = input_sentences[: args.batch_size]


def generate(model, inputs, generate_kwargs):
    """returns a list of zipped inputs, outputs and number of new tokens"""
    
    input_tokens = tokenizer.batch_encode_plus(inputs, return_tensors="pt", padding=True)
    for t in input_tokens:
        if torch.is_tensor(input_tokens[t]):
            input_tokens[t] = input_tokens[t].to(model.device)

    outputs = model.generate(**input_tokens, **generate_kwargs)

    input_tokens_lengths = [x.shape[0] for x in input_tokens.input_ids]
    output_tokens_lengths = [x.shape[0] for x in outputs]

    total_new_tokens = [o - i for i, o in zip(input_tokens_lengths, output_tokens_lengths)]
    outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return zip(inputs, outputs, total_new_tokens)


def benchmark(model, inputs, generate_kwargs):

    ### Benchmark

    if True:
        # clear cache / free memory
        torch.cuda.empty_cache()
        gc.collect()

        print_rank0("*** Running benchmark")
        # warm up
        for i in range(1):
            _ = generate(model, inputs, generate_kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # benchmark
        t0 = time.time()
        cycles = 5
        total_new_tokens_generated = 0
        for i in range(cycles):
            generated = generate(model, inputs, generate_kwargs)
            total_new_tokens_generated += sum(new_tokens for _, _, new_tokens in generated)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        latency = time.time() - t0
        throughput = latency / (total_new_tokens_generated)
        print_rank0(
            f"""
        *** Performance stats:
        msec/token: {throughput*1000:.2f}
        bs={len(inputs)}
        num_tokens={total_new_tokens_generated}
        latency={latency:.2f} (sec)
        """
        )



def make_deepspeed(model, dtype):
    return deepspeed.init_inference(model,
                                    mp_size=world_size,
                                    dtype=dtype,
                                    replace_with_kernel_inject=True)

if __name__ == "__main__":
    # Arguments: model, dtype, device, inference_engine
    yaml_path, args_list = sys.argv[1], sys.argv[2:]
    with open(yaml_path) as f:
        yaml_cfg = om.load(f)
    cli_cfg = om.from_cli(args_list)
    cfg = DictConfig(om.merge(yaml_cfg, cli_cfg))

    reproducibility.seed_all(cfg.get('seed', 1234))


    print(cfg.model)
    kwargs = {
        "device_map": "auto"
    }
    if get_world_size() > 1:
        kwargs["device_map"] = "balanced_low_0"

    if cfg.dtype == 'float32':
        dtype = torch.float32
    elif cfg.dtype == 'float16':
        dtype = torch.float16
    elif cfg.dtype  == "int8":
        dtype = torch.int8
        kwargs["load_in_8bit"] = True
    
    kwargs["torch_dtype"] = dtype


    model = AutoModelForCausalLM.from_pretrained(cfg.model.model_name, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    if cfg.inference_platform == 'deepspeed':
        model = make_deepspeed(model, kwargs["torch_dtype"])

    input_sentences = list(cfg.input_sentences)
    batch_size = cfg.batch_size
    num_tokens = cfg.num_tokens

    if batch_size > len(input_sentences):
        # dynamically extend to support larger bs by repetition
        input_sentences *= math.ceil(batch_size / len(input_sentences))
    inputs = input_sentences[: batch_size]

   
    generate_kwargs = dict(max_new_tokens=num_tokens, do_sample=False)

    benchmark(model, inputs, generate_kwargs)