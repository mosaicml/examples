# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import os
import time
from typing import Any, Dict, List

import lm_eval
import pandas as pd
import torch
from lm_eval import models as lm_eval_models
from lm_eval import tasks as lm_eval_tasks
from lm_eval.evaluator import evaluate
from model_loading import load_composer_checkpoint, load_huggingface_checkpoint

RESULTS_DIR = f'{os.path.dirname(__file__)}/eval_reports'
PARTIAL_EVAL_SAMPLE_SIZE = 40

JsonResults = Dict[int, Dict[str, Any]]


def get_lm_eval_model(checkpoint_type: str,
                      checkpoint_args: Dict[str, str]) -> lm_eval.base.LM:
    """Load a trained model for evaluation.

    In more detail, this loads a ComposerGPT model from a checkpoint & yaml
    and then passes it into the lm_eval.models.ComposerLLM wrapper class so
    that we can interface nicely with lm_eval task evaluation.

    Args:
        checkpoint_type (str): The type of checkpoint being loaded. Either 'huggingface' or 'composer'
        checkpoint_args (Dict[str, str]): Any args needed for loading the checkpoint.

    Returns:
        lm (lm_eval.base.LM): An lm_eval wrapper around our constructed model
    """
    if checkpoint_type == 'huggingface':
        model_components = load_huggingface_checkpoint(**checkpoint_args)
    elif checkpoint_type == 'composer':
        model_components = load_composer_checkpoint(**checkpoint_args)
    else:
        raise ValueError(
            f'Not sure how to init {model_type=}. Must be one of "huggingface" or "composer"'
        )
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_components.update({'batch_size': 4, 'device': device})
    lm = lm_eval_models.get_model('composer_llm').create_from_arg_string(
        '', model_components)
    return lm


def evaluate_model_on_tasks(model: lm_eval.base.LM, tasks: List[str],
                            num_fewshots: List[int],
                            partial_eval_mode: bool) -> JsonResults:
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
    task_dict = lm_eval_tasks.get_task_dict(tasks)
    results = {}
    for num in num_fewshots:
        results[num] = evaluate(
            lm=model,
            task_dict=task_dict,
            num_fewshot=num,
            limit=PARTIAL_EVAL_SAMPLE_SIZE if partial_eval_mode else None)

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
        'task', 'ppl', 'accuracy', 'length-normalized_accuracy', 'num_fewshot'
    ])

    for num in results:
        fewshot_exp_results = results[num]['results']
        for task_name in fewshot_exp_results:
            accuracy = fewshot_exp_results[task_name].get('acc', 'n/a')
            accuracy_norm = fewshot_exp_results[task_name].get(
                'acc_norm', 'n/a')
            ppl = fewshot_exp_results[task_name].get('ppl', 'n/a')
            df = pd.concat([
                pd.DataFrame([[task_name, ppl, accuracy, accuracy_norm, num]],
                             columns=df.columns), df
            ],
                           ignore_index=True)

    with open(outfile, 'w') as f:
        df.to_csv(f, sep='\t', index=None)


if __name__ == '__main__':
    """Example usage:

    python eval.py --experiment_name opt_1.3b \
        --checkpoint_type huggingface \
        --checkpoint_args pretrained_model_name_or_path=facebook/opt-1.3b \
        --tasks lambada_cloze \
        --num_fewshot 10

    python eval.py --experiment_name gpt_neo_125m \
        --checkpoint_type huggingface \
        --checkpoint_args pretrained_model_name_or_path=EleutherAI/gpt-neo-125M \
        --tasks lambada_openai \
        --num_fewshot 0 1


    For composer checkpoints you can either pass in an S3 URI or a local path to a .pt file
    Your config must also be available locally

    python eval.py --experiment_name gpt_neo_125m \
        --checkpoint_type composer_checkpoint \
        --checkpoint_args checkpoint=s3://mosaicml-internal-checkpoints-shared/jeremy/gpt-eval/gpt-neo-125m-5rbfZF/checkpoints/ep0-ba80000-rank0.pt \
            config=yamls/mosaic_gpt/125m.yaml \
        --tasks lambada_openai \
        --num_fewshot 0 1 10
    """
    parser = argparse.ArgumentParser(
        description='Run the EleutherAI eval harness on ComposerGPT models')
    parser.add_argument('--experiment_name', type=str)
    parser.add_argument('--checkpoint_type', type=str)
    parser.add_argument('--checkpoint_args', nargs='+', type=str)
    parser.add_argument('--tasks', metavar='task_names', type=str, nargs='+')
    parser.add_argument('--num_fewshots', type=int, nargs='+')
    parser.add_argument('--partial_eval_mode',
                        action='store_true',
                        default=False)

    args = parser.parse_args()
    checkpoint_args = {}
    for arg in args.checkpoint_args:
        k, v = tuple(arg.split('='))
        checkpoint_args[k] = v

    model = get_lm_eval_model(args.checkpoint_type, checkpoint_args)
    results = evaluate_model_on_tasks(model, args.tasks, args.num_fewshots,
                                      args.partial_eval_mode)
    outfile = f"{RESULTS_DIR}/{args.experiment_name}_{'-'.join(args.tasks)}_{hash(time.time())%10_000}.tsv"
    log_results_to_tsv(results, outfile)
