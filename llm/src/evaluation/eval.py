# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import os
import time
from typing import Any, Dict, List
import sys
from omegaconf import OmegaConf as om
import lm_eval
import pandas as pd
import torch
from lm_eval import models as lm_eval_models
from lm_eval import tasks as lm_eval_tasks
from lm_eval.evaluator import evaluate
from src.evaluation.model_loading import load_model

RESULTS_DIR = f'{os.path.dirname(__file__)}/eval_reports'
PARTIAL_EVAL_SAMPLE_SIZE = 40

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device(
    'cpu')
JsonResults = Dict[int, Dict[str, Any]]


def get_lm_eval_model(model_config) -> lm_eval.base.LM:
    """Load a trained ComposerGPT model for evaluation.

    In more detail, this loads a ComposerGPT model from a checkpoint & yaml
    and then passes it into the lm_eval.models.ComposerLLM wrapper class so
    that we can interface nicely with lm_eval task evaluation.

    Args:
        model_ctor_args (Dict[str, str]): Constructor args for the model.
        model_type (str): The name of the model from
        llm.llm_evaluation.model_loading.MODEL_LOADERS

    Returns:
        lm (lm_eval.base.LM): An lm_eval wrapper around our constructed model
    """
    model_components = load_model(**model_config)

    model_components.update({'batch_size': 4, 'device': DEVICE.type})
    lm = lm_eval_models.get_model('composer_llm').create_from_arg_string(
        '', model_components)
    return lm


def evaluate_model_on_tasks(model: lm_eval.base.LM, task_configs: List[dict]) -> JsonResults:
    """Returns the sum of two decimal numbers in binary digits.

    Args:
        model (lm_eval.base.LM): An lm_eval LM model.
        tasks (List[str]): List of task names as defined in
            lm_eval.tasks.TASK_REGISTRY
        num_fewshots (List[int]): Number of few shots to used for in-context
            learning for each task. Runs each task separately for each number
            in this list.
        partial_eval_mode (bool): To quickly sanity check model performance,
            run eval on only the first `PARTIAL_EVAL_SAMPLE_SIZE` examples in
            the task's test set.

    Returns:
        results (JsonResults): Results of the task including ppl, acc,
            acc_norm (if multiple choice), task name, and few shot samples used
    """
    results = {}
    for task_conf in task_configs:
        label, num_fewshots = task_conf.get("label"), task_conf.get("num_fewshot")
        task_dict = lm_eval_tasks.get_task_dict([label])
        for num in num_fewshots:
            a = time.time()
            results[num] = evaluate(
                lm=model,
                task_dict=task_dict,
                num_fewshot=num
            )
            b = time.time()
            results[num]['time'] = b-a


    return results


def log_results_to_tsv(results: JsonResults, outfile: str) -> None:
    """Logs the task results to a pandas tsv.

    Args:
        results (JsonResults): Model's performance on the tasks.
        outfile (str): Output file
    """
    dumped = json.dumps(results, indent=2)
    print(f'Logging results to {outfile}')
    print(dumped)
    df = pd.DataFrame(columns=[
        'task', 'ppl', 'accuracy', 'length-normalized_accuracy', 'num_fewshot', "time"
    ])

    for num in results:
        fewshot_exp_results = results[num]['results']
        for task_name in fewshot_exp_results:
            accuracy = fewshot_exp_results[task_name].get('acc', 'n/a')
            accuracy_norm = fewshot_exp_results[task_name].get(
                'acc_norm', 'n/a')
            ppl = fewshot_exp_results[task_name].get('ppl', 'n/a')
            df = pd.concat([
                pd.DataFrame([[task_name, ppl, accuracy, accuracy_norm, num, results[num]['time']]],
                             columns=df.columns), df
            ],
                           ignore_index=True)

    with open(outfile, 'w') as f:
        df.to_csv(f, sep='\t', index=None)
    
    print(df)


if __name__ == '__main__':
    """Example usage from examples/llm directory:

    python -m src.evaluation.eval --experiment_name opt_1.3b \
        --model_type pretrained_hf \
        --model_args checkpoint=facebook/opt-1.3b \
        --tasks lambada_cloze \
        --num_fewshot 10

    python -m src.evaluation.eval --experiment_name gpt_neo_125m \
        --model_type pretrained_hf \
        --model_args checkpoint=EleutherAI/gpt-neo-125M \
        --tasks lambada \
        --num_fewshot 0 1


    For composer checkpoints you can either pass in an S3 URI or a local path to a .pt file
    Your config must also be available locally

    python -m src.evaluation.eval --experiment_name gpt_neo_125m \
        --model_type composer_checkpoint \
        --model_args checkpoint=s3://mosaicml-internal-checkpoints-shared/jeremy/gpt-eval/gpt-neo-125m-5rbfZF/checkpoints/ep0-ba80000-rank0.pt \
            config=yamls/mosaic_gpt/125m.yaml \
        --tasks lambada \
        --num_fewshot 0 1 10
    """
    yaml_path, args_list = sys.argv[1], sys.argv[2:]
    with open(yaml_path) as f:
        yaml_cfg = om.load(f)
    cli_cfg = om.from_cli(args_list)
    cfg = om.merge(yaml_cfg, cli_cfg)

    model = get_lm_eval_model(cfg.get("model"))
    results = evaluate_model_on_tasks(model, cfg.get("icl_tasks"))
    outfile = f"{RESULTS_DIR}/{cfg.get('outfile')}"
    log_results_to_tsv(results, outfile)
