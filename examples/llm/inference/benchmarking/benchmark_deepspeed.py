import os
import sys
import time
import torch
import transformers
import deepspeed
import numpy as np
from omegaconf import OmegaConf as om

# You can use this to load the model weights
from composer import Trainer
from composer.core import get_precision_context

# from composer.utils import dist, get_device
from examples.llm.src import COMPOSER_MODEL_REGISTRY


def main(config):
    tokenizer_name = "gpt2"

    inf_config = {
        "replace_with_kernel_inject": True,
        "dtype": torch.float16,
        "replace_method": "auto",
        "enable_cuda_graph": False,
        "tensor_parallel": {"tp_size": 1},
    }

    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)

    model = COMPOSER_MODEL_REGISTRY[config.model.name](config.model, config.tokenizer)
    model.eval()

    print ("model is: ", model)
    inference_model = deepspeed.init_inference(model, config=inf_config)
    print ("inference model is: ", inference_model)

    # Checking if deepspeed casts dtypes correctly
    for n, p in inference_model.named_parameters():
        print("n is: ", n, p)
        print ("dtype is: ", p.dtype)
        break
    
    stats = []
    for batch_size in config.batch_sizes:
        for input_length in config.input_lengths:
            for output_length in config.output_lengths:
                times = []
                batch = torch.ones((batch_size, input_length)).cuda() * 17
                batch = batch.to(torch.long)
                for i in range(config.num_runs + 1):
                    start_time = time.time()
                    out = inference_model.generate(batch, max_new_tokens=output_length, use_cache=True)
                    # We noticed there sometimes might be a small bit of startup time 
                    # so we only benchmark after the first iteration
                    if i > 0:
                        times.append(time.time() - start_time)
                
                num_output_tokens = output_length * batch_size
                mean_time = np.mean(times)
                tokens_per_second = num_output_tokens / float(mean_time)

                resu = (f'{config.benchmark_name}_{batch_size}_{input_length}_{output_length}', f"{tokens_per_second:.3f}")

                run_name, tokens_per_second = resu

                print (f"{run_name}, {tokens_per_second}")

                stats.append(resu)

if __name__ == "__main__":
    yaml_path, args_list = sys.argv[1], sys.argv[2:]
    with open(yaml_path) as f:
        yaml_config = om.load(f)
    cli_config = om.from_cli(args_list)
    config = om.merge(yaml_config, cli_config)
    main(config)