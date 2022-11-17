# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import warnings

from composer import Trainer
from composer.callbacks import LRMonitor, MemoryMonitor, SpeedMonitor
from composer.loggers import WandBLogger
from composer.optim import DecoupledAdamW
from composer.algorithms import SelectiveBackprop
from torch_optimizer import Adafactor
from composer.optim.scheduler import (ConstantWithWarmupScheduler,
                                      CosineAnnealingWithWarmupScheduler)
from composer.utils import dist, reproducibility
from omegaconf import OmegaConf as om

from src.data_c4 import build_c4_dataloader
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
    elif cfg.name == "Adafactor":
        return Adafactor(model.parameters(), lr=cfg.lr)
    else:
        raise ValueError(f'Not sure how to build optimizer: {cfg.name}')

def build_algorithm(cfg):
    if cfg.name == "selective_backprop":
        return SelectiveBackprop(start=cfg.start, end=cfg.end, interrupt=cfg.interrupt, keep=cfg.keep)
    else:
        raise ValueError(f"Not sure how to build algorithm: {cfg.name}")

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
def update_batch_size_info(cfg):
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

    cfg.n_gpus = dist.get_world_size()
    cfg.device_train_batch_size = device_train_batch_size
    cfg.device_train_grad_accum = device_train_grad_accum
    cfg.device_eval_batch_size = device_eval_batch_size
    cfg.device_eval_microbatch_size = device_eval_microbatch_size
    return cfg

def log_config(cfg):
    print(om.to_yaml(cfg))
    if 'wandb' in cfg.get('loggers', {}):
        try:
            import wandb
        except ImportError as e:
            raise e
        if wandb.run:
            wandb.config.update(om.to_container(cfg, resolve=True))

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

    # Get batch size info
    cfg = update_batch_size_info(cfg)

    # Read FSDP Config as a dict
    fsdp_config = cfg.get('fsdp_config', None)
    fsdp_config = om.to_container(fsdp_config, resolve=True) if fsdp_config else None

    # Build Model
    # For fast initialization of MosaicGPT, use cfg.model.device='meta'
    print('Initializing model...')
    model = build_composer_model(cfg.model)
    cfg.n_params = sum(p.numel() for p in model.parameters())
    print(f'{cfg.n_params=:.2e}')

    # Dataloaders
    print("Building train loader...")
    train_loader = build_dataloader(cfg.train_loader, cfg.device_train_batch_size)
    print("Building eval loader...")
    eval_loader = build_dataloader(cfg.eval_loader, cfg.device_eval_batch_size)

    # Optimizer
    optimizer = build_optimizer(cfg.optimizer, model)

    # Scheduler
    scheduler = build_scheduler(cfg.scheduler)

    # Loggers
    loggers = [build_logger(name, logger_cfg) for name, logger_cfg in cfg.get('loggers', {}).items()]

    # Callbacks
    callbacks = [build_callback(name, callback_cfg) for name, callback_cfg in cfg.get('callbacks', {}).items()]

    algorithms = [build_algorithm(name, algorithm_cfg) for name, algorithm_cfg in cfg.get("algorithms", {}).items()]

    # Build the Trainer
    trainer = Trainer(
        run_name=cfg.run_name,
        seed=cfg.seed,
        model=model,
        algorithms=algorithms,
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
        grad_accum=cfg.device_train_grad_accum,
        fsdp_config=fsdp_config,
        save_folder=cfg.get('save_folder', None),
        save_interval=cfg.get('save_interval', '1000ba'),
        save_num_checkpoints_to_keep=cfg.get('save_num_checkpoints_to_keep', -1),
        load_path=cfg.get('load_path', None),
        load_weights_only=cfg.get('load_weights_only', False),
    )

    print("Logging config...")
    log_config(cfg)

    print("Starting training...")
    trainer.fit()


if __name__ == '__main__':
    yaml_path, args_list = sys.argv[1], sys.argv[2:]
    with open(yaml_path) as f:
        yaml_cfg = om.load(f)
    cli_cfg = om.from_cli(args_list)
    cfg = om.merge(yaml_cfg, cli_cfg)
    main(cfg)
