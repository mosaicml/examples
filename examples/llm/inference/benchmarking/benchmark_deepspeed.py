# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

import sys
import time

import deepspeed
import numpy as np
import torch
from transformers import AutoTokenizer
import transformers
# You can use this to load the model weights
from composer.core import get_precision_context
from omegaconf import OmegaConf as om

from examples.llm.src import COMPOSER_MODEL_REGISTRY


def main(config):
    inf_config = {
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

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

    inference_model = deepspeed.init_inference(model, config=inf_config)

    # Checking if deepspeed casts dtypes correctly
    for n, p in inference_model.named_parameters():
        print('n is: ', n, p)
        print('dtype is: ', p.dtype)
        break

    stats = []
    for batch_size in config.batch_sizes:
        for input_length in config.input_lengths:
            for output_length in config.output_lengths:
                times = []
                eos_token = tokenizer.eos_token
                # Make sure we are not generating a fake batch with a EOS token 
                while True:
                    batch = torch.randint(
                        0, config.model.vocab_size-1, size=(
                            batch_size,
                            input_length)).to(f'cuda:{torch.cuda.current_device()}')
                    if tokenizer.convert_tokens_to_ids(eos_token) not in batch:
                        break
                batch = batch.to(torch.long)
                for i in range(config.num_runs + 1):
                    start_time = time.time()
                    with torch.no_grad():
                        if config.use_precision_context:
                            with get_precision_context(config.precision):
                                inference_model.generate(
                                    batch,
                                    max_new_tokens=output_length,
                                    use_cache=True)
                        else:
                            inference_model.generate(
                                batch,
                                max_new_tokens=output_length,
                                use_cache=True)
                    # We noticed there sometimes might be a small bit of startup time
                    # so we only start to benchmark after the first iteration
                    if i > 0:
                        times.append(time.time() - start_time)

                num_output_tokens = output_length * batch_size
                mean_time = np.mean(times)
                tokens_per_second = num_output_tokens / float(mean_time)

                resu = (
                    f'{config.benchmark_name}_{batch_size}_{input_length}_{output_length}',
                    f'{mean_time:.3f}', f'{tokens_per_second:.3f}')

                run_name, latency, tokens_per_second = resu

                print(f'{run_name}, {latency}, {tokens_per_second}')

                stats.append(resu)

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
