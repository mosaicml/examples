# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0
import itertools
import os
import random
import time
import warnings
from argparse import ArgumentParser, ArgumentTypeError, Namespace

import torch
from composer.core import Precision
from composer.models import HuggingFaceModel
from composer.trainer.dist_strategy import prepare_fsdp_module
from composer.utils import dist, get_device
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from examples.common.scheduled_gc_callback import gc_cuda
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
        '--prompt_files',
        nargs='+',
    )
    parser.add_argument('--max_seq_len', type=int, default=81920)
    parser.add_argument('--max_new_tokens', type=int, default=2048)
    parser.add_argument('--temperature', type=float, nargs='+', default=[1.0])
    parser.add_argument('--top_k', type=int, nargs='+', default=[50])
    parser.add_argument('--top_p', type=float, nargs='+', default=[1.0])
    parser.add_argument('--repetition_penalty', type=float, default=1.0)
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
    parser.add_argument('--seed', type=int, nargs='+', default=[42])
    parser.add_argument('--profile_gen_timing',
                        type=str2bool,
                        nargs='?',
                        const=True,
                        default=False)
    parser.add_argument('--skip_prompt_printout', action='store_true')
    return parser.parse_args()


def maybe_synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main(args: Namespace) -> None:
    prompts = []
    for prompt_file in args.prompt_files:
        if not os.path.isfile(prompt_file):
            raise FileNotFoundError(f'{prompt_file=} does not match any file.')
        with open(prompt_file, 'r') as f:
            prompts.append(''.join(f.readlines()))

    if args.profile_gen_timing:
        max_new_tokens = [1, 1, 128, 256]
        prompts = prompts[:1] * len(max_new_tokens)
    else:
        max_new_tokens = []

    # Seed randomness
    random.seed(args.seed[0])
    torch.manual_seed(args.seed[0])

    dist.initialize_dist(get_device(None), timeout=1800)

    AutoConfig.register('mosaic_gpt', MosaicGPTConfig)
    AutoModelForCausalLM.register(MosaicGPTConfig, MosaicGPT)

    print('Loading HF model...')
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

    model = HuggingFaceModel(model,
                             tokenizer,
                             use_logits=True,
                             shift_labels=True)

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
        'mixed_precision': 'DEFAULT',
        'activation_checkpointing': False,
        'limit_all_gathers': True,
        'activation_cpu_offload': False,
        'verbose': False,
    }

    prepare_fsdp_module(model,
                        optimizers=None,
                        fsdp_config=fsdp_config,
                        precision=precision)

    # Autocast
    if args.autocast:
        print(f'Using autocast amp_{args.dtype}...')
    else:
        print('NOT using autocast...')

    # dummy forward call needed for FSDP to work consistently
    print('Dummy input...')
    dummy_input = torch.tensor([[0]], dtype=torch.long, device=device)
    with torch.no_grad():
        _ = model.model(input_ids=dummy_input)

    # # Warmup
    # if args.warmup:
    #     print('Warming up...')
    #     with torch.no_grad():
    #         with torch.autocast('cuda', dtype, enabled=args.autocast):
    #             encoded_gen = model.model.generate(
    #                 input_ids=encoded_inp['input_ids'],
    #                 attention_mask=encoded_inp['attention_mask'],
    #                 **generate_kwargs,
    #             )

    for idx, prompt in enumerate(prompts):
        for temp, top_k, top_p, seed in itertools.product(
                args.temperature, args.top_k, args.top_p, args.seed):
            print('!@!' * 33)

            # Seed randomness
            random.seed(seed)
            torch.manual_seed(seed)

            print(f'\nSeed: {seed}')

            generate_kwargs = {
                'max_new_tokens': args.max_new_tokens,
                'temperature': temp,
                'top_p': top_p,
                'top_k': top_k,
                'repetition_penalty': args.repetition_penalty,
                'use_cache': args.use_cache,
                'do_sample': args.do_sample,
                'eos_token_id': args.eos_token_id or tokenizer.eos_token_id,
                'pad_token_id': args.pad_token_id or tokenizer.pad_token_id,
                'synced_gpus': True,
            }
            print(f'\nGenerate kwargs:\n{generate_kwargs}')

            if args.profile_gen_timing:
                generate_kwargs['max_new_tokens'] = max_new_tokens[idx]

            print(f'\n\nTokenizing prompt ({idx+1} / {len(prompts)})...')
            maybe_synchronize()
            encode_start = time.time()
            encoded_inp = tokenizer([prompt], return_tensors='pt', padding=True)
            for key, value in encoded_inp.items():
                encoded_inp[key] = value.to(device)
            maybe_synchronize()
            encode_end = time.time()
            input_tokens = torch.sum(
                encoded_inp['input_ids'] != tokenizer.pad_token_id,
                axis=1).numpy(force=True)  # type: ignore

            # Run HF generate
            print(f'Generating responses ({idx+1} / {len(prompts)})...')
            maybe_synchronize()
            gen_start = time.time()
            with torch.no_grad():
                with torch.autocast('cuda', dtype, enabled=args.autocast):
                    encoded_gen = model.model.generate(
                        input_ids=encoded_inp['input_ids'],
                        attention_mask=encoded_inp['attention_mask'],
                        **generate_kwargs,
                    )
            maybe_synchronize()
            gen_end = time.time()

            decode_start = time.time()
            decoded_gen = tokenizer.batch_decode(encoded_gen,
                                                 skip_special_tokens=True)[0]
            maybe_synchronize()
            decode_end = time.time()
            gen_tokens = torch.sum(encoded_gen != tokenizer.pad_token_id,
                                   axis=1).numpy(force=True)  # type: ignore

            # Print generations
            if not args.profile_gen_timing:
                delimiter = '#' * 100
                continuation = decoded_gen[len(prompt):]
                print(delimiter)
                if args.skip_prompt_printout:
                    print(continuation)
                else:
                    print('\033[92m' + prompt + '\033[0m' + continuation)
                print(delimiter)

            # Print timing info
            output_tokens = gen_tokens - input_tokens
            total_input_tokens = input_tokens.sum()
            total_output_tokens = output_tokens.sum()
            encode_latency = 1000 * (encode_end - encode_start)
            gen_latency = 1000 * (gen_end - gen_start)
            decode_latency = 1000 * (decode_end - decode_start)
            total_latency = encode_latency + gen_latency + decode_latency

            latency_per_output_token = total_latency / total_output_tokens
            output_tok_per_sec = 1000 / latency_per_output_token
            print(f'{input_tokens=}, {output_tokens=}')
            print(f'{total_input_tokens=}, {total_output_tokens=}')
            print(
                f'{encode_latency=:.2f}ms, {gen_latency=:.2f}ms, {decode_latency=:.2f}ms, {total_latency=:.2f}ms'
            )
            print(f'{latency_per_output_token=:.2f}ms/tok')
            print(f'{output_tok_per_sec=:.2f}tok/sec')

            gc_cuda()


if __name__ == '__main__':
    main(parse_args())
