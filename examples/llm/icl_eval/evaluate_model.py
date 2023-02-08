# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

import sys
import time

import torch
from composer.loggers import InMemoryLogger
from composer.trainer import Trainer
from model_loading import load_composer_model
from omegaconf import DictConfig
from omegaconf import OmegaConf as om

from examples.common.builders import build_evaluators
from examples.llm.src.tokenizer import TOKENIZER_REGISTRY


def validate_cfg(eval_cfg):
    assert eval_cfg.get('dataset_uri',
                        None) is not None, 'dataset_uri is required!'
    assert 'type' in eval_cfg
    assert 'num_fewshot' in eval_cfg
    assert 'batch_size' in eval_cfg
    assert 'metrics' in eval_cfg
    assert 'formatting_options' in eval_cfg
    assert 'prompt_string' in eval_cfg.get('formatting_options')
    assert 'example_delimiter' in eval_cfg.get('formatting_options')
    assert 'continuation_delimiter' in eval_cfg.get('formatting_options')
    assert 'label' in eval_cfg


def get_evaluators_from_config(cfg):
    tokenizer = TOKENIZER_REGISTRY[cfg.tokenizer.type](**cfg.tokenizer.args)
    evaluators = []
    logger_keys = []
    for eval_cfg in cfg.icl_tasks:
        validate_cfg(eval_cfg)
        dataset_uri = eval_cfg.get('dataset_uri')
        icl_task_type = eval_cfg.get('type')
        num_fewshots = eval_cfg.get('num_fewshot')
        batch_size = eval_cfg.get('batch_size')
        metrics = list(eval_cfg.get('metrics'))
        prompt_string = eval_cfg.get('formatting_options').get('prompt_string')
        example_delimiter = eval_cfg.get('formatting_options').get(
            'example_delimiter')
        continuation_delimiter = eval_cfg.get('formatting_options').get(
            'continuation_delimiter')
        label = eval_cfg.get('label')
        max_seq_len = cfg.tokenizer.args.max_seq_len

        res = build_evaluators(label, icl_task_type, dataset_uri, tokenizer,
                               batch_size, max_seq_len, prompt_string,
                               example_delimiter, continuation_delimiter,
                               metrics, num_fewshots)
        evaluators.extend(res[0])
        logger_keys.extend(res[1])

    return evaluators, logger_keys


if __name__ == '__main__':
    yaml_path, args_list = sys.argv[1], sys.argv[2:]
    with open(yaml_path) as f:
        yaml_cfg = om.load(f)
    cli_cfg = om.from_cli(args_list)
    cfg = DictConfig(om.merge(yaml_cfg, cli_cfg))

    composer_model = load_composer_model(cfg)

    evaluators, logger_keys = get_evaluators_from_config(cfg)
    for evaluator in evaluators:
        composer_model.add_eval_metrics(evaluator)

    in_memory_logger = InMemoryLogger()  # track metrics in the in_memory_logger

    fsdp_config = cfg.get('fsdp_config', None)
    fsdp_config = om.to_container(fsdp_config,
                                  resolve=True) if fsdp_config is not None else None

    load_path = cfg.get('load_path', None)

    trainer = Trainer(
        model=composer_model,
        loggers=in_memory_logger,
        fsdp_config=fsdp_config,  # type: ignore
        load_path=load_path,
        load_weights_only=True,
        progress_bar=False,
        log_to_console=True)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    a = time.time()
    trainer.eval(eval_dataloader=evaluators)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    b = time.time()

    print(f'Ran eval in: {b-a} seconds')

    for key in logger_keys:
        if key in in_memory_logger.data:
            result = in_memory_logger.data[key][0][1].item()
            print(f'{key}: {result}')
