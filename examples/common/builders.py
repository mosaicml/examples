# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

from composer import algorithms
from composer.callbacks import LRMonitor, MemoryMonitor, OptimizerMonitor
from composer.core import Evaluator
from composer.datasets.in_context_learning_evaluation import \
    get_icl_task_dataloader
from composer.loggers import WandBLogger
from composer.optim import DecoupledAdamW
from composer.optim.scheduler import (ConstantWithWarmupScheduler,
                                      CosineAnnealingWithWarmupScheduler,
                                      LinearWithWarmupScheduler)

from examples.common.speed_monitor_w_mfu import SpeedMonitorMFU
from examples.common.text_data import build_text_dataloader


def build_callback(name, kwargs):
    if name == 'lr_monitor':
        return LRMonitor()
    elif name == 'memory_monitor':
        return MemoryMonitor()
    elif name == 'speed_monitor':
        return SpeedMonitorMFU(window_size=kwargs.get('window_size', 1),
                               gpu_flops_available=kwargs.get(
                                   'gpu_flops_available', None))
    elif name == 'optimizer_monitor':
        return OptimizerMonitor(log_optimizer_metrics=kwargs.get(
            'log_optimizer_metrics', True),)
    else:
        raise ValueError(f'Not sure how to build callback: {name}')


def build_logger(name, kwargs):
    if name == 'wandb':
        return WandBLogger(**kwargs)
    else:
        raise ValueError(f'Not sure how to build logger: {name}')


def build_algorithm(name, kwargs):
    if name == 'gradient_clipping':
        return algorithms.GradientClipping(**kwargs)
    elif name == 'alibi':
        return algorithms.Alibi(**kwargs)
    elif name == 'fused_layernorm':
        return algorithms.FusedLayerNorm(**kwargs)
    elif name == 'gated_linear_units':
        return algorithms.GatedLinearUnits(**kwargs)
    elif name == 'low_precision_layernorm':
        return algorithms.LowPrecisionLayerNorm(**kwargs)
    else:
        raise ValueError(f'Not sure how to build algorithm: {name}')


def build_optimizer(cfg, model):
    if cfg.name == 'decoupled_adamw':
        return DecoupledAdamW(model.parameters(),
                              lr=cfg.lr,
                              betas=cfg.betas,
                              eps=cfg.eps,
                              weight_decay=cfg.weight_decay)
    else:
        raise ValueError(f'Not sure how to build optimizer: {cfg.name}')


def build_scheduler(cfg):
    if cfg.name == 'constant_with_warmup':
        return ConstantWithWarmupScheduler(t_warmup=cfg.t_warmup)
    elif cfg.name == 'cosine_with_warmup':
        return CosineAnnealingWithWarmupScheduler(t_warmup=cfg.t_warmup,
                                                  alpha_f=cfg.alpha_f)
    elif cfg.name == 'linear_decay_with_warmup':
        return LinearWithWarmupScheduler(t_warmup=cfg.t_warmup,
                                         alpha_f=cfg.alpha_f)
    else:
        raise ValueError(f'Not sure how to build scheduler: {cfg.name}')


def build_dataloader(cfg, device_batch_size):
    if cfg.name == 'text':
        return build_text_dataloader(cfg, device_batch_size)
    else:
        raise ValueError(f'Not sure how to build dataloader with config: {cfg}')


def build_evaluators(label, icl_task_type, dataset_uri, tokenizer, batch_size,
                     max_seq_len, prompt_string, example_delimiter,
                     continuation_delimiter, metrics, num_fewshots):

    evaluators = []
    logger_keys = []
    for num_fewshot in num_fewshots:
        label = f'{label}_{num_fewshot}-shot'
        dl = get_icl_task_dataloader(
            icl_task_type,
            dataset_uri,
            tokenizer,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            pad_tok_id=tokenizer.pad_token_id,
            num_fewshot=num_fewshot,
            prompt_string=prompt_string,
            example_delimiter=example_delimiter,
            continuation_delimiter=continuation_delimiter)
        logger_keys.extend([f'metrics/{label}/{metric}' for metric in metrics])
        evaluators.append(
            Evaluator(label=label, dataloader=dl, metric_names=metrics))

    return evaluators, logger_keys
