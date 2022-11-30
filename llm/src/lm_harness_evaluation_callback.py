import argparse
import fnmatch
import json
import logging
import pprint

from abc import ABC
from datetime import datetime as dt
from typing import Any

from composer import Callback, State, Logger
import lm_eval.models
from lm_eval import evaluator, tasks

import torch
import torchmetrics
import transformers
import wandb


logging.getLogger("openai").setLevel(logging.WARNING)


class MultiChoice:
    def __init__(self, choices):
        self.choices = choices

    # Simple wildcard support (linux filename patterns)
    def __contains__(self, values):
        for value in values.split(","):
            if len(fnmatch.filter(self.choices, value)) == 0:
                return False

        return True

    def __iter__(self):
        for choice in self.choices:
            yield choice


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model_args", default="")
    parser.add_argument("--tasks", default=None, choices=MultiChoice(tasks.ALL_TASKS))
    parser.add_argument("--provide_description", action="store_true")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--decontamination_ngrams_path", default=None)
    parser.add_argument("--description_dict_path", default=None)
    parser.add_argument("--check_integrity", action="store_true")

    return parser.parse_args()


# Returns a list containing all values of the source_list that
# match at least one of the patterns
def pattern_match(patterns, source_list):
    task_names = set()
    for pattern in patterns:
        for matching in fnmatch.filter(source_list, pattern):
            task_names.add(matching)
    return list(task_names)


def main(args: argparse.Namespace):
    assert not args.provide_description  # not implemented

    if args.limit:
        print(
            "WARNING: --limit SHOULD ONLY BE USED FOR TESTING. REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT."
        )

    if args.tasks is None:
        task_names = tasks.ALL_TASKS
    else:
        task_names = pattern_match(args.tasks.split(","), tasks.ALL_TASKS)

    print(f"Selected Tasks: {task_names}")

    description_dict = {}
    if args.description_dict_path:
        with open(args.description_dict_path, "r") as f:
            description_dict = json.load(f)

    results = evaluator.simple_evaluate(
        model=args.model,
        model_args=args.model_args,
        tasks=task_names,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        device=args.device,
        no_cache=args.no_cache,
        limit=args.limit,
        description_dict=description_dict,
        decontamination_ngrams_path=args.decontamination_ngrams_path,
        check_integrity=args.check_integrity,
    )

    results_without_model = {k: v for k, v in results.items() if k not in {"model", "device"}}
    print(f"results_without_model.keys(): {results_without_model.keys()}")
    print(json.dumps(results_without_model, indent=2))
    wandb.log(results_without_model)


class HFTokenizer(ABC):
    tokenizer_name: str
    max_seq_len: int
    group_method: str
    tokenizer: transformers.AutoTokenizer

    def __init__(self, tokenizer_name: str, max_seq_len: int):
        # super().__init__(tokenizer_name, max_seq_len)
        self.tokenizer_name = tokenizer_name
        self.max_seq_len = max_seq_len

        # Build tokenizer
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.tokenizer_name)
        if self.tokenizer.pad_token is None:
            # Some tokenizers (e.g. GPT2 tokenizer) have no padding token which causes bugs
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # suppress warnings when using group_method='concat' and no truncation
        self.tokenizer.model_max_length = int(1e30)

    def __call__(self, *args, **kwargs):
        return self.tokenizer(*args, **kwargs)

    def encode(self, x):
        return self.tokenizer.encode(x)

    def decode(self, x):
        return self.tokenizer.decode(x)

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size

    @property
    def pad_token_id(self):
        return self.tokenizer.pad_token_id

    @property
    def bos_token_id(self):
        return self.tokenizer.bos_token_id


class LMEvalHarnessReducerMetric(torchmetrics.Metric):
    """Somewhat janky distributed reducer hacked together w/ torchmetrics.

    Just concats responses and sampled indices from eval harness. compute simply
    returns those tensors for use elsewhere.
    """
    def __init__(self):
        super().__init__()
        self.add_state("responses", default=torch.Tensor([]), dist_reduce_fx="cat")
        self.add_state("sampled_indices", default=torch.Tensor([]), dist_reduce_fx="cat")

    def update(self, preds, target):
        self.responses = torch.cat([self.responses, torch.Tensor(preds)])
        self.sampled_indices = torch.cat([self.sampled_indices, torch.Tensor(target)])
        print(f"metric updating... new shape: {self.responses.shape}")

    def compute(self):
        return self.responses, self.sampled_indices


class EvaluationCallback(Callback):
    def __init__(self, every_n_batches=1024):
        super().__init__()
        self.every_n_batches = every_n_batches
        self.metric = LMEvalHarnessReducerMetric()
        self.simple_evaluate_args = None
        self.simple_evaluate_inference = None

    def before_train_batch(self, state: State, logger: Logger):
        print(f"callback observed batch: {state.timestamp.batch}")
        if not (int(state.timestamp.batch) + 1) % self.every_n_batches:  # kick off forked lm evaluation harness
            self.start_time = dt.now()
            model = lm_eval.models.get_model(
                "composer_llm"
            ).create_from_arg_string(
                "",
                {
                    "model": state.model.model,
                    "tokenizer": HFTokenizer(
                        tokenizer_name="gpt2",
                        max_seq_len=2048,
                    ),
                    "precision": state.precision,
                    "device": torch.device('cuda').type,
                    # "batch_size": 64,
                }
            )

            task_names = pattern_match(["lambada"], tasks.ALL_TASKS)
            print(f"Selected Tasks: {task_names}")

            self.simple_evaluate_args = evaluator.simple_evaluate_args(
                model=model,
                model_args="",
                tasks=task_names,
                num_fewshot=0,
                batch_size=None,
                device=None,
                no_cache=True,
                limit=None,
                description_dict={},
                decontamination_ngrams_path=None,
                check_integrity=False,
            )

            inference = evaluator.evaluate_inference(
                **self.simple_evaluate_args
            )
            self.simple_evaluate_inference = inference
            self.metric.update(inference["resps"], inference["sampled_indices"])

            # results = main(
            #     argparse.Namespace(
            #         model=model,
            #         model_args="",
            #         tasks="lambada",
            #         provide_description=False,
            #         num_fewshot=0,
            #         batch_size=None,  # N/A b/c model is defined
            #         device=None,  # N/A b/c model is defined
            #         limit=None,
            #         no_cache=True,
            #         decontamination_ngrams_path=None,
            #         description_dict_path=None,
            #         check_integrity=False,
            #     )
            # )

            # results_without_model = {k: v for k, v in results.items() if k not in {"model", "device"}}
            # print(f"results_without_model.keys(): {results_without_model.keys()}")
            # print(json.dumps(results_without_model, indent=2))
            # wandb.log(results_without_model)

            # logger.info(f"ran evaluation in: {(dt.now() - start).total_seconds():.03f}")

    def after_train_bach(self, state: State, logger: Logger):
        if not (int(state.timestamp.batch) + 1) % self.every_n_batches:
            assert False
            resps, sampled_indices = self.metric.compute()

            results = evaluator.evaluate_metrics(
                **self.simple_evaluate_args,
                **self.simple_evaluate_inference,
                resps=resps,
                sampled_indices=sampled_indices,
            )

            # results_without_model = {k: v for k, v in results.items() if k not in {"model", "device"}}
            # print(f"results_without_model.keys(): {results_without_model.keys()}")

            print(f"eval results: {pprint.pformat(json.dumps(results['results'], indent=2))}")
            wandb.log(results["results"])

            logger.info(f"ran evaluation in: {(dt.now() - self.start_time).total_seconds():.03f}")


if __name__ == "__main__":
    main(parse_args())
