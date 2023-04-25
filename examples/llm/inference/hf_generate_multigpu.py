# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0
import os
import random
import time
import warnings
from argparse import ArgumentParser, ArgumentTypeError, Namespace

import torch
from composer.core import Precision
from composer.trainer.dist_strategy import prepare_fsdp_module
from composer.utils import dist, get_device
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from examples.llm import MosaicGPT, MosaicGPTConfig


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')


def parse_args() -> Namespace:
    """Parse commandline arguments."""
    parser = ArgumentParser(
        description='Load a HF CausalLM Model and use it to generate text.')
    parser.add_argument('-n', '--name_or_path', type=str, required=True)
    parser.add_argument(
        '-p',
        '--prompts',
        type=str,
        default='the_great_gatsby_epi.txt',
        # default=[
        #     'My name is',
        #     'This is an explanation of deep learning to a five year old. Deep learning is',
        # ]
    )
    parser.add_argument('--max_seq_len', type=int, default=81920)
    parser.add_argument('--max_new_tokens', type=int, default=2048)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_k', type=int, default=50)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--do_sample',
                        type=str2bool,
                        nargs='?',
                        const=True,
                        default=True)
    parser.add_argument('--use_cache',
                        type=str2bool,
                        nargs='?',
                        const=True,
                        default=True)
    parser.add_argument('--eos_token_id', type=str, default=None)
    parser.add_argument('--pad_token_id', type=str, default=None)
    parser.add_argument('--dtype',
                        type=str,
                        choices=['fp32', 'fp16', 'bf16'],
                        default='bf16')
    parser.add_argument('--autocast',
                        type=str2bool,
                        nargs='?',
                        const=True,
                        default=False)
    parser.add_argument('--warmup',
                        type=str2bool,
                        nargs='?',
                        const=True,
                        default=True)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def maybe_synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main(args: Namespace) -> None:
    prompts = []
    prompt = args.prompts
    if not os.path.isfile(prompt):
        raise FileNotFoundError(f'{prompt=} does not match any file.')
    with open(prompt, 'r') as f:
        prompts.append(''.join(f.readlines()))
    # prompts = ['This is a book about a dog named Samwise.']

    dist.initialize_dist(get_device(None), timeout=1800)

    AutoConfig.register('mosaic_gpt', MosaicGPTConfig)
    AutoModelForCausalLM.register(MosaicGPTConfig, MosaicGPT)

    print('Loading HF model...')
    # config = AutoConfig.from_pretrained(args.name_or_path,
    #                                     max_seq_len=args.max_seq_len)
    # model = AutoModelForCausalLM.from_config(config)
    model = AutoModelForCausalLM.from_pretrained(args.name_or_path,
                                                 max_seq_len=args.max_seq_len)
    model.eval()
    print(f'n_params={sum(p.numel() for p in model.parameters())}')

    print('\nLoading HF tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(args.name_or_path)
    if tokenizer.pad_token_id is None:
        warnings.warn(
            'pad_token_id is not set for the tokenizer. Using eos_token_id as pad_token_id.'
        )
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    generate_kwargs = {
        'max_new_tokens': args.max_new_tokens,
        'temperature': args.temperature,
        'top_p': args.top_p,
        'top_k': args.top_k,
        'use_cache': args.use_cache,
        'do_sample': args.do_sample,
        'eos_token_id': args.eos_token_id or tokenizer.eos_token_id,
        'pad_token_id': args.pad_token_id or tokenizer.pad_token_id,
        'synced_gpus': True,
    }
    print(f'\nGenerate kwargs:\n{generate_kwargs}')

    device = torch.cuda.current_device(
    )  #'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = {
        'fp16': torch.float16,
        'bf16': torch.bfloat16,
        'fp32': torch.float32,
    }[args.dtype]
    precision = {
        'fp16': Precision('amp_fp16'),
        'bf16': Precision('amp_bf16'),
        'fp32': Precision('fp32'),
    }[args.dtype]
    print(f'\nMoving model and inputs to GPU and dtype={dtype}... FSDP Go!')
    model.to('cuda', dtype)

    fsdp_config = {
        'sharding_strategy': 'FULL_SHARD',
        'mixed_precision': 'PURE',
        'activation_checkpointing': True,
        'activation_cpu_offload': False,
        'verbose': False,
    }

    prepare_fsdp_module(model,
                        optimizers=None,
                        fsdp_config=fsdp_config,
                        precision=precision)

    print(f'\nTokenizing prompts...')
    maybe_synchronize()
    encode_start = time.time()
    encoded_inp = tokenizer(prompts, return_tensors='pt', padding=True)
    for key, value in encoded_inp.items():
        encoded_inp[key] = value.to(device)
    maybe_synchronize()
    encode_end = time.time()
    input_tokens = torch.sum(encoded_inp['input_ids'] != tokenizer.pad_token_id,
                             axis=1).numpy(force=True)  # type: ignore

    # Autocast
    if args.autocast:
        print(f'Using autocast amp_{args.dtype}...')
    else:
        print('NOT using autocast...')

    # dummy forward call needed for FSDP to work consistently
    dummy_input = torch.tensor([[0]], dtype=torch.long, device=device)
    # dummy_input = device.tensor_to_device(dummy_input)
    with torch.no_grad():
        _ = model.forward(input_ids=dummy_input)

    # Warmup
    if args.warmup:
        print('Warming up...')
        with torch.no_grad():
            with torch.autocast('cuda', dtype, enabled=args.autocast):
                encoded_gen = model.generate(
                    input_ids=encoded_inp['input_ids'],
                    attention_mask=encoded_inp['attention_mask'],
                    **generate_kwargs,
                )

    # Seed randomness
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Run HF generate
    print('Generating responses...')
    maybe_synchronize()
    gen_start = time.time()
    with torch.no_grad():
        with torch.autocast('cuda', dtype, enabled=args.autocast):
            encoded_gen = model.generate(
                input_ids=encoded_inp['input_ids'],
                attention_mask=encoded_inp['attention_mask'],
                **generate_kwargs,
            )
    maybe_synchronize()
    gen_end = time.time()

    decode_start = time.time()
    decoded_gen = tokenizer.batch_decode(encoded_gen, skip_special_tokens=True)
    maybe_synchronize()
    decode_end = time.time()
    gen_tokens = torch.sum(encoded_gen != tokenizer.pad_token_id,
                           axis=1).numpy(force=True)  # type: ignore

    # Print generations
    delimiter = '#' * 100
    for prompt, gen in zip(prompts, decoded_gen):
        continuation = gen[len(prompt):]
        print(delimiter)
        print('\033[92m' + prompt + '\033[0m' + continuation)
    print(delimiter)

    # Print timing info
    bs = len(prompts)
    output_tokens = gen_tokens - input_tokens
    total_input_tokens = input_tokens.sum()
    total_output_tokens = output_tokens.sum()
    encode_latency = 1000 * (encode_end - encode_start)
    gen_latency = 1000 * (gen_end - gen_start)
    decode_latency = 1000 * (decode_end - decode_start)
    total_latency = encode_latency + gen_latency + decode_latency

    latency_per_output_token = total_latency / total_output_tokens
    output_tok_per_sec = 1000 / latency_per_output_token
    print(f'{bs=}, {input_tokens=}, {output_tokens=}')
    print(f'{total_input_tokens=}, {total_output_tokens=}')
    print(
        f'{encode_latency=:.2f}ms, {gen_latency=:.2f}ms, {decode_latency=:.2f}ms, {total_latency=:.2f}ms'
    )
    print(f'{latency_per_output_token=:.2f}ms/tok')
    print(f'{output_tok_per_sec=:.2f}tok/sec')


if __name__ == '__main__':
    main(parse_args())
