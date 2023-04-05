# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0
import time
import warnings
from argparse import ArgumentParser, Namespace
from contextlib import nullcontext

import torch
from composer.core import get_precision_context
from composer.utils import get_device, reproducibility
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from examples.llm import MosaicGPT, MosaicGPTConfig


def parse_args() -> Namespace:
    """Parse commandline arguments."""
    parser = ArgumentParser(
        description='Load a HF CausalLM Model and use it to generate text.')
    parser.add_argument('-n', '--name_or_path', type=str, required=True)
    parser.add_argument(
        '-p',
        '--prompts',
        nargs='+',
        default=[
            'My name is',
            'This is an explanation of deep learning to a five year old. Deep learning is',
        ])
    parser.add_argument('--max_new_tokens', type=int, default=100)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_k', type=int, default=50)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--dtype',
                        type=str,
                        choices=['fp32', 'fp16', 'bf16'],
                        default='bf16')
    parser.add_argument('--autocast', action='store_true')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def main(args: Namespace) -> None:
    reproducibility.seed_all(args.seed)

    AutoConfig.register('mosaic_gpt', MosaicGPTConfig)
    AutoModelForCausalLM.register(MosaicGPTConfig, MosaicGPT)

    print('Loading HF model...')
    model = AutoModelForCausalLM.from_pretrained(args.name_or_path)
    model.eval()
    print(f'n_params={sum(p.numel() for p in model.parameters())}')

    print('\nLoading HF tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(args.name_or_path)
    if tokenizer.pad_token_id is None:
        warnings.warn(
            'pad_token_id is not set for the tokenizer. Using eos_token_id as pad_token_id.'
        )
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'left'

    generate_kwargs = {
        'max_new_tokens': args.max_new_tokens,
        'temperature': args.temperature,
        'top_p': args.top_p,
        'top_k': args.top_k,
        'use_cache': True,
        'do_sample': True,
    }
    print(f'\nGenerate kwargs:\n{generate_kwargs}')

    device = get_device(args.device)._device
    dtype = {
        'fp16': torch.float16,
        'bf16': torch.bfloat16,
        'fp32': torch.float32,
    }[args.dtype]
    print(f'\nMoving model and inputs to device={device} and dtype={dtype}...')
    model.to(device, dtype)
    encoded_inp = tokenizer(args.prompts, return_tensors='pt', padding=True)
    input_tokens = torch.sum(encoded_inp['input_ids'] != tokenizer.pad_token_id,
                             axis=1).numpy(force=True)

    for key, value in encoded_inp.items():
        encoded_inp[key] = value.to(device)

    if args.autocast and args.dtype != 'fp32':
        print(f'\nUsing autocast...')
        context_manager = lambda: get_precision_context(f'amp_{args.dtype}')
    else:
        print(f'\nNOT using autocast...')
        context_manager = nullcontext

    # Run HF generate
    start = time.time()
    with torch.no_grad():
        with context_manager():
            encoded_gen = model.generate(
                input_ids=encoded_inp['input_ids'],
                attention_mask=encoded_inp['attention_mask'],
                **generate_kwargs,
            )
    decoded_gen = tokenizer.batch_decode(encoded_gen, skip_special_tokens=True)
    end = time.time()
    gen_tokens = torch.sum(encoded_gen != tokenizer.pad_token_id,
                           axis=1).numpy(force=True)

    # Print generations
    delimiter = '#' * 100
    for prompt, gen in zip(args.prompts, decoded_gen):
        continuation = gen[len(prompt):]
        print(delimiter)
        print('\033[92m' + prompt + '\033[0m' + continuation)
    print(delimiter)

    # Print timing info
    bs = len(args.prompts)
    output_tokens = gen_tokens - input_tokens
    total_input_tokens, total_output_tokens = input_tokens.sum(
    ), output_tokens.sum()
    latency = end - start
    output_tok_per_sec = total_output_tokens / latency
    ms_per_output_tok = 1000 / output_tok_per_sec
    print(f'{bs=}, {input_tokens=}, {output_tokens=}')
    print(f'{total_input_tokens=}, {total_output_tokens=}')
    print(
        f'{latency=:.2f}s, {output_tok_per_sec=:.2f}, {ms_per_output_tok=:.2f}ms'
    )


if __name__ == '__main__':
    main(parse_args())
