import argparse
import json
import logging
import fnmatch
from datetime import datetime as dt

from abc import ABC
from typing import Any

from composer import Callback, State, Logger
import lm_eval.models
from lm_eval import evaluator, tasks

import torch
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

    wandb.log(results)

    dumped = json.dumps(
        {
            **results,
            "model": None,  # because ComposerLLM object is not JSON serializable
        },
        indent=2,
    )
    print(dumped)


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


class EvaluationCallback(Callback):
    def __init__(self, every_n_batches=1024):
        super().__init__()
        self.every_n_batches = every_n_batches

    def before_train_batch(self, state: State, logger: Logger):
        if not int(state.timestamp.batch) % self.every_n_batches:  # kick off forked lm evaluation harness
            start = dt.now()
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
                }
            )
            main(
                argparse.Namespace(
                    model=model,
                    model_args="",
                    tasks="lambada",
                    provide_description=False,
                    num_fewshot=0,
                    batch_size=None,
                    device=None,
                    limit=None,
                    no_cache=True,
                    decontamination_ngrams_path=None,
                    description_dict_path=None,
                    check_integrity=False,
                )
            )
            logger.info(f"ran evaluation in: {(dt.now() - s).total_seconds():.03f}")


if __name__ == "__main__":
    main(parse_args())
