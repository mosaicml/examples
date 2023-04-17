# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

import contextlib
import sys
import time

import deepspeed
import numpy as np
import torch
# You can use this to load the model weights
from composer.core import get_precision_context
from omegaconf import OmegaConf as om
from transformers import AutoTokenizer

from examples.llm.src import COMPOSER_MODEL_REGISTRY


def compare_precision(precision, param_dtype):
    if ((precision == 'amp_bf16' and param_dtype != torch.bfloat16) or
        (precision == 'amp_fp16' and param_dtype != torch.float16) or
        (precision == 'fp32' and param_dtype != torch.float32)):
        raise ValueError(
            f'Precision type is: {precision} but model dtype is: {param_dtype}. '
            f"The expected precision and model precision don't match.")


def main(config):
    inference_config = {
        'replace_with_kernel_inject': True,
        'dtype': torch.bfloat16,
        'replace_method': 'auto',
        'enable_cuda_graph': False,
        'tensor_parallel': {
            'tp_size': 0
        },
    }

    model = COMPOSER_MODEL_REGISTRY[config.model.name](config.model,
                                                       config.tokenizer)
    model.eval()

    tokenizer = model.tokenizer

    if config.use_deepspeed:
        model = deepspeed.init_inference(model, config=inference_config)

        # Checking if deepspeed casts dtypes correctly
        for n, p in model.named_parameters():
            compare_precision(config.precision, p.dtype)
            break
    else:
        model.to(torch.cuda.current_device())

    n_params = sum(p.numel() for p in model.parameters())
    print('n_params is: ', n_params)

    print('Run name, latency (s), tokens per second')
    print('=' * 75)

    stats = []
    for batch_size in config.batch_sizes:
        for input_length in config.input_lengths:
            for output_length in config.output_lengths:
                times = []

                batch = torch.randint(
                    0,
                    config.model.vocab_size - 1,
                    size=(
                        batch_size,
                        input_length)).to(f'cuda:{torch.cuda.current_device()}')

                # Make sure we are not generating a fake batch with a EOS token
                eos_mask = batch == tokenizer.eos_token_id
                batch = batch.masked_fill(eos_mask,
                                          config.tokenizer.non_eos_token_id)

                batch = batch.to(torch.long)

                attention_mask = torch.logical_not(
                    torch.eq(batch, tokenizer.eos_token_id))

                torch.cuda.synchronize()

                for i in range(config.num_runs + 1):
                    start_time = time.time()
                    with torch.no_grad():
                        precision_context = contextlib.nullcontext()
                        if config.use_precision_context:
                            precision_context = get_precision_context(
                                config.precision)

                        with precision_context:
                            model.generate(batch,
                                           max_new_tokens=output_length,
                                           use_cache=True,
                                           attention_mask=attention_mask)

                    # We noticed there sometimes might be a small bit of startup time
                    # so we only start to benchmark after some number of batches
                    if i >= config.num_warmup_batches:
                        times.append(time.time() - start_time)

                    torch.cuda.synchronize()

                num_output_tokens = output_length * batch_size
                mean_time = np.mean(times)
                tokens_per_second = num_output_tokens / float(mean_time)

                result = (
                    f'{config.benchmark_name}_{batch_size}_{input_length}_{output_length}',
                    f'{mean_time:.3f}', f'{tokens_per_second:.3f}')

                run_name, latency, tokens_per_second = result

                print(f'{run_name}, {latency}, {tokens_per_second}')

                stats.append(result)

    print('=' * 75)
    print('name, latency (s), tokens / s')
    for val in stats:
        print(val)


if __name__ == '__main__':
    yaml_path, args_list = sys.argv[1], sys.argv[2:]
    with open(yaml_path) as f:
        yaml_config = om.load(f)
    cli_config = om.from_cli(args_list)
    config = om.merge(yaml_config, cli_config)
    main(config)
