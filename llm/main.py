# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import warnings

from composer import Trainer
from composer.callbacks import LRMonitor, MemoryMonitor, SpeedMonitor
from composer.loggers import WandBLogger
from composer.optim import DecoupledAdamW
from composer.optim.scheduler import (ConstantWithWarmupScheduler,
                                      CosineAnnealingWithWarmupScheduler)
from composer.utils import dist, reproducibility
from omegaconf import OmegaConf as om

from src.data import build_c4_dataloader
from src.mosaic_gpt import ComposerMosaicGPT


def build_logger(name, kwargs):
    if name == 'wandb':
        return WandBLogger(**kwargs)
    else:
        raise ValueError(f'Not sure how to build logger: {name}')

def build_callback(name, kwargs):
    if name == 'lr_monitor':
        return LRMonitor()
    elif name == 'memory_monitor':
        return MemoryMonitor()
    elif name == 'speed_monitor':
        return SpeedMonitor(window_size=kwargs.get('window_size', 1))
    else:
        raise ValueError(f'Not sure how to build callback: {name}')

def build_optimizer(cfg, model):
    if cfg.name == 'decoupled_adamw':
        return DecoupledAdamW(
            model.parameters(),
            lr=cfg.lr,
            betas=cfg.betas,
            eps=cfg.eps,
            weight_decay=cfg.weight_decay
        )
    else:
        raise ValueError(f'Not sure how to build optimizer: {cfg.name}')


def build_scheduler(cfg):
    if cfg.name == 'constant_with_warmup':
        return ConstantWithWarmupScheduler(
            t_warmup=cfg.t_warmup)
    elif cfg.name == 'cosine_with_warmup':
        return CosineAnnealingWithWarmupScheduler(
            t_warmup=cfg.t_warmup,
            alpha_f=cfg.alpha_f)
    else:
        raise ValueError(f'Not sure how to build scheduler: {cfg.name}')

# Coming soon: this conversion math will be done inside Composer Trainer rather than entrypoint
def get_batch_size_info(cfg):
    global_train_batch_size = cfg.global_train_batch_size
    device_train_batch_size = global_train_batch_size // dist.get_world_size()
    device_train_microbatch_size = cfg.device_train_microbatch_size
    if device_train_microbatch_size == 'auto':
        device_train_grad_accum = 'auto'
        device_eval_microbatch_size = 'auto'
        device_eval_batch_size = device_train_batch_size
    elif isinstance(device_train_microbatch_size, int):
        if device_train_microbatch_size > device_train_batch_size:
            print (f"WARNING: device_train_microbatch_size > device_train_batch_size, will be reduced from {device_train_microbatch_size} -> {device_train_batch_size}.")
            cfg.device_train_microbatch_size = device_train_batch_size
            device_train_microbatch_size = device_train_batch_size
        device_train_grad_accum = device_train_batch_size // device_train_microbatch_size
        device_eval_microbatch_size = device_train_microbatch_size
        device_eval_batch_size = device_eval_microbatch_size
    else:
        raise ValueError(
            f'Not sure how to parse {device_train_microbatch_size=}')

    return device_train_batch_size, device_train_grad_accum, device_eval_batch_size, device_eval_microbatch_size


def build_composer_model(cfg):
    warnings.filterwarnings(action='ignore', message='Torchmetrics v0.9 introduced a new argument class property')

    if cfg.name == 'mosaic_gpt':
        return ComposerMosaicGPT(cfg)
    else:
        raise ValueError(f'Not sure how to build model with name={cfg.name}')

def build_dataloader(cfg, device_batch_size):
    if cfg.name == 'c4':
        return build_c4_dataloader(cfg, device_batch_size)
    else:
        raise ValueError(f'Not sure how to build model with name={cfg.name}')

def main(cfg):
    reproducibility.seed_all(cfg.seed)

    # Run Name
    cfg.run_name = cfg.get('run_name', os.environ.get('COMPOSER_RUN_NAME', 'llm'))

    # Read FSDP Config as a dict
    fsdp_config = cfg.get('fsdp_config', None)
    fsdp_config = om.to_container(fsdp_config, resolve=True) if fsdp_config else None

    # Build Model
    # For fast initialization of MosaicGPT, use cfg.model.device='meta'
    print('Initializing model...')
    model = build_composer_model(cfg.model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'{n_params=:.2e}')

    # Get batch size info
    device_train_batch_size, device_train_grad_accum, device_eval_batch_size, device_eval_microbatch_size = get_batch_size_info(cfg)

    # Dataloaders
    print("Building train loader...")
    train_loader = build_dataloader(cfg.train_loader, device_train_batch_size)
    print("Building eval loader...")
    eval_loader = build_dataloader(cfg.eval_loader, device_eval_batch_size)

    # Optimizer
    optimizer = build_optimizer(cfg.optimizer, model)

    # Scheduler
    scheduler = build_scheduler(cfg.scheduler)

    # Loggers
    loggers = [build_logger(name, logger_cfg) for name, logger_cfg in cfg.get('loggers', {}).items()]

    # Callbacks
    callbacks = [build_callback(name, callback_cfg) for name, callback_cfg in cfg.get('callbacks', {}).items()]

    # Build the Trainer
    trainer = Trainer(
        run_name=cfg.run_name,
        seed=cfg.seed,
        model=model,
        train_dataloader=train_loader,
        eval_dataloader=eval_loader,
        optimizers=optimizer,
        schedulers=scheduler,
        max_duration=cfg.max_duration,
        eval_interval=cfg.eval_interval,
        progress_bar=cfg.progress_bar,
        log_to_console=cfg.log_to_console,
        loggers=loggers,
        callbacks=callbacks,
        precision=cfg.precision,
        grad_clip_norm=cfg.grad_clip_norm,
        grad_accum=device_train_grad_accum,
        fsdp_config=fsdp_config,
        save_folder=cfg.get('save_folder', None),
        save_interval=cfg.get('save_interval', '1000ba'),
        save_num_checkpoints_to_keep=cfg.get('save_num_checkpoints_to_keep', -1),
        load_path=cfg.get('load_path', None),
        load_weights_only=cfg.get('load_weights_only', False),
    )

    print("Logging config...")
    cfg.n_gpus = dist.get_world_size()
    cfg.n_params = n_params
    cfg.device_train_batch_size = device_train_batch_size
    cfg.device_eval_batch_size = device_eval_batch_size
    cfg.device_eval_microbatch_size = device_eval_microbatch_size
    print(om.to_yaml(cfg))
    if 'wandb' in cfg.callbacks:
        try:
            import wandb
        except ImportError as e:
            raise e
        wandb.config.update(om.to_container(cfg, resolve=True))

    print("Starting training...")
    trainer.fit()


if __name__ == '__main__':
    yaml_path, args_list = sys.argv[1], sys.argv[2:]
    with open(yaml_path) as f:
        yaml_cfg = om.load(f)
    cli_cfg = om.from_cli(args_list)
    cfg = om.merge(yaml_cfg, cli_cfg)
    main(cfg)
