# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0
import warnings

import torch
from composer.core import get_precision_context
from composer.utils import get_device, reproducibility
from argparse import ArgumentParser, Namespace, Action
from examples.llm import MosaicGPTConfig, MosaicGPT
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from contextlib import nullcontext
import time

class ParseKwargs(Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value = value.split('=')
            getattr(namespace, self.dest)[key] = value

def parse_args() -> Namespace:
    """Parse commandline arguments."""
    parser = ArgumentParser(
        description=
        'Load a HF CausalLM Model and use it to generate text.'
    )
    parser.add_argument('-n', '--name_or_path', type=str, required=True)
    parser.add_argument('-k', '--kwargs', nargs='*', action=ParseKwargs, default={})
    parser.add_argument('-p', '--prompts', nargs='*', action=ParseKwargs, default={})
    parser.add_argument('--dtype', type=str, choices=['fp32', 'fp16', 'bf16'], default='bf16')
    parser.add_argument('--autocast', action='store_true')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def main(args: Namespace) -> None:
    reproducibility.seed_all(args.seed)

    AutoConfig.register('mosaic_gpt', MosaicGPTConfig)
    AutoModelForCausalLM.register(MosaicGPTConfig, MosaicGPT)
    
    print ('Loading HF model...')
    model = AutoModelForCausalLM.from_pretrained(args.name_or_path)
    model.eval()
    print (f'n_params={sum(p.numel() for p in model.parameters())}')

    print ('\nLoading HF tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(args.name_or_path)
    if tokenizer.pad_token_id is None:
        warnings.warn(
            'pad_token_id is not set for the tokenizer. Using eos_token_id as pad_token_id.'
        )
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'left'

    generate_kwargs = {
        'max_new_tokens': 100,
        'use_cache': True,
        'do_sample': True,
        'top_p': 0.95,
        'eos_token_id': tokenizer.eos_token_id,
    }
    generate_kwargs.update(args.kwargs)
    print (f'\nGenerate kwargs:\n{generate_kwargs}')

    prompts = [
        'My name is',
        'This is an explanation of deep learning to a five year old. Deep learning is',
    ]

    device = get_device(args.device)._device
    dtype = {
        'fp16': torch.float16,
        'bf16': torch.bfloat16,
        'fp32': torch.float32,
    }[args.dtype]
    print (f'\nMoving model and inputs to device={device} and dtype={dtype}...')
    model.to(device, dtype)
    encoded_inp = tokenizer(prompts, return_tensors='pt', padding=True)
    for key, value in encoded_inp.items():
        encoded_inp[key] = value.to(device)

    if args.autocast and args.dtype != 'fp32':
        print (f'\nUsing autocast...')
        context_manager = get_precision_context(f'amp_{args.dtype}')
    else:
        print (f'\nNOT using autocast...')
        context_manager = nullcontext

    start = time.time()
    with torch.no_grad():
        with context_manager:
            generation = model.generate(
                input_ids=encoded_inp['input_ids'],
                attention_mask=encoded_inp['attention_mask'],
                **generate_kwargs,
            )
    decoded_out = tokenizer.batch_decode(generation, skip_special_tokens=True)
    end = time.time()

    delimiter = '#'*100
    for prompt, gen in zip(prompts, decoded_out):
        gen = gen[len(prompt):]
        print (delimiter)
        print ('\033[92m' + prompt + '\033[0m' + gen)
    print (delimiter)

if __name__ == '__main__':
    main(parse_args())