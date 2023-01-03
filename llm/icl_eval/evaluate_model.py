import os
from src.tokenizer import TOKENIZER_REGISTRY

import pytest
import transformers

from composer.core import Evaluator
from composer.datasets.in_context_learning_evaluation import get_lm_task_dataloader, get_mc_task_dataloader

from composer.loggers import InMemoryLogger
from composer.models.gpt2 import create_gpt2
from composer.trainer import Trainer
from omegaconf import OmegaConf as om
import sys

DATALOADER = {
    "language_modeling": get_lm_task_dataloader,
    "multiple_choice": get_mc_task_dataloader
}

def build_evaluators(cfg):
    
    tokenizer = TOKENIZER_REGISTRY[cfg.tokenizer.type](**cfg.tokenizer.args)
    evaluators = []
    for eval_cfg in cfg.icl_tasks:
        label = eval_cfg.get("label")
        dataset_uri = eval_cfg.get("dataset_uri")
        type = eval_cfg.get("type")
        num_fewshots = eval_cfg.get("num_fewshot")
        batch_size = eval_cfg.get("batch_size")
        metrics = list(eval_cfg.get("metrics"))
        preamble_string = list(eval_cfg.get("formatting_options").get("preamble_string"))
        example_delimiter = list(eval_cfg.get("formatting_options").get("example_delimiter"))
        continuation_delimiter = list(eval_cfg.get("formatting_options").get("continuation_delimiter"))

        for num_fewshot in num_fewshots:
            dl = DATALOADER[type](
                dataset_uri,
                tokenizer,
                batch_size=batch_size,
                max_seq_len=cfg.tokenizer.args.max_seq_len,
                eos_tok_id=tokenizer.pad_token_id,
                num_fewshot=num_fewshot,
                preamble_string=preamble_string,
                example_delimiter=example_delimiter,
                continuation_delimiter=continuation_delimiter
            )
            evaluators.append(Evaluator(label=label, dataloader=dl, metric_names=metrics))

    return evaluators


if __name__ == '__main__':
    yaml_path, args_list = sys.argv[1], sys.argv[2:]
    with open(yaml_path) as f:
        yaml_cfg = om.load(f)
    cli_cfg = om.from_cli(args_list)
    cfg = om.merge(yaml_cfg, cli_cfg)


    evaluators = build_evaluators(cfg)
    breakpoint()
    in_memory_logger = InMemoryLogger()  # track the logged metrics in the in_memory_logger
    # dataset_uri = 's3://mosaicml-internal-dataset-lambda/lambada/lambada_test.json'
    dataset_uri = 's3://mosaicml-internal-dataset-hellaswag/hellaswag.jsonz'

    tokenizer = transformers.AutoTokenizer.from_pretrained('gpt2')

    dl = get_mc_task_dataloader(dataset_uri,
                                tokenizer,
                                batch_size=16,
                                max_seq_len=2048,
                                eos_tok_id=tokenizer.eos_token_id,
                                num_fewshot=5,
                                preamble_string='',
                                example_delimiter='\n',
                                continuation_delimiter=': ')

    evaluator = Evaluator(label='hellaswag', dataloader=dl, metric_names=['InContextLearningMultipleChoiceAccuracy'])
    model = create_gpt2(use_pretrained=True, pretrained_model_name='EleutherAI/gpt-neo-1.3B')
    trainer = Trainer(model=model, max_duration='1ba', loggers=in_memory_logger)
    trainer.eval(eval_dataloader=evaluator)
    assert 'metrics/hellaswag/InContextLearningMultipleChoiceAccuracy' in in_memory_logger.data.keys()
    print(in_memory_logger.data['metrics/hellaswag/InContextLearningMultipleChoiceAccuracy'][0][1].item())