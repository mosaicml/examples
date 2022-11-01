# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import warnings

import wandb
from composer import Trainer
from composer.callbacks import LRMonitor, MemoryMonitor, SpeedMonitor
from composer.loggers import ProgressBarLogger, WandBLogger
from composer.optim import DecoupledAdamW
from composer.optim.scheduler import (ConstantWithWarmupScheduler,
                                      CosineAnnealingWithWarmupScheduler,
                                      LinearWithWarmupScheduler)
from composer.utils import S3ObjectStore, dist, reproducibility
from omegaconf import OmegaConf as om

from src.data_c4 import build_c4_dataloader
from src.hf_bert import create_bert_mlm


def build_logger(name, kwargs):
    if name == 'progress_bar':
        return ProgressBarLogger(
            progress_bar=kwargs.get('progress_bar', True),
            log_to_console=kwargs.get('log_to_console', True),
        )
    elif name == 'wandb':
        return WandBLogger(**kwargs)
    elif name == 's3':
        object_store_logger = ObjectStoreLogger(
            object_store_cls=S3ObjectStore,
            object_store_kwargs=kwargs,
        )
        return object_store_logger
    else:
        raise ValueError(f'Not sure how to build logger: {name}')

def build_object_store(name, kwargs):
    if name == 's3':
        return S3ObjectStore(**kwargs)
    else:
        raise ValueError(f'Not sure how to build object store: {name}')

def build_callback(name, kwargs):
    if name == 'lr_monitor':
        return LRMonitor()
    elif name == 'memory_monitor':
        return MemoryMonitor()
    elif name == 'speed_monitor':
        return SpeedMonitor(window_size=kwargs.get('window_size', 1))
    else:
        raise ValueError(f'Not sure how to build callback: {name}')

def build_scheduler(cfg):
    if cfg.name == 'constant_with_warmup':
        return ConstantWithWarmupScheduler(
            t_warmup=cfg.t_warmup)
    elif cfg.name == 'linear_decay_with_warmup':
        return LinearWithWarmupScheduler(
            t_warmup=cfg.t_warmup,
            alpha_f=cfg.alpha_f
        )
    elif cfg.name == 'cosine_with_warmup':
        return CosineAnnealingWithWarmupScheduler(
            t_warmup=cfg.t_warmup,
            alpha_f=cfg.alpha_f)
    else:
        raise ValueError(f'Not sure how to build scheduler: {cfg.name}')

def build_model(cfg):
    warnings.filterwarnings(action='ignore', message='Torchmetrics v0.9 introduced a new argument class property')

    if cfg.name == 'hf_bert':
        return create_bert_mlm(
            pretrained_model_name=cfg.pretrained_model_name,
            use_pretrained=cfg.get('use_pretrained', None),
            model_config=cfg.get('model_config', None),
            tokenizer_name=cfg.get('tokenizer_name', None),
            gradient_checkpointing=cfg.get('gradient_checkpointing', None)
        )
    else:
        raise ValueError(f'Not sure how to build model with name={cfg.name}')

def build_dataloader(cfg, device_batch_size):
    if cfg.name == 'c4':
        return build_c4_dataloader(cfg, device_batch_size)
    else:
        raise ValueError(f'Not sure how to build model with name={cfg.name}')

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


def main(cfg):
    print("Training using config: ")
    print(om.to_yaml(cfg))
    reproducibility.seed_all(cfg.seed)

    # # Read FSDP Config as a dict
    # fsdp_config = cfg.get('fsdp_config', None)
    # fsdp_config = om.to_container(fsdp_config, resolve=True) if fsdp_config else None

    # Build Model
    # For fast initialization, use `meta` device
    print('Initializing model...')
    model = build_model(cfg.model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'{n_params=:.2e}')

    # Get batch size info
    global_train_batch_size = cfg.global_train_batch_size
    device_train_batch_size = global_train_batch_size // dist.get_world_size()
    device_eval_batch_size = device_train_batch_size
    # device_train_batch_size, device_train_grad_accum, device_eval_batch_size, device_eval_microbatch_size = get_batch_size_info(cfg)

    # Dataloaders
    print("Building train loader...")
    train_loader = build_dataloader(cfg.train_loader, device_train_batch_size)
    print("Building eval loader...")
    eval_loader = build_dataloader(cfg.eval_loader, device_eval_batch_size)

    # Optimizer
    assert cfg.optimizer.name == 'decoupled_adamw'
    optimizer = DecoupledAdamW(
        model.parameters(),
        lr=cfg.optimizer.lr,
        betas=cfg.optimizer.betas,
        eps=cfg.optimizer.eps,
        weight_decay=cfg.optimizer.weight_decay)

    # Scheduler
    scheduler = build_scheduler(cfg.scheduler)

    # Loggers
    loggers = [build_logger(name, logger_cfg) for name, logger_cfg in cfg.loggers.items()]

    # Callbacks
    callbacks = [build_callback(name, callback_cfg) for name, callback_cfg in cfg.callbacks.items()]

    # (Optional) Load object store
    load_object_store = cfg.get('load_object_store', None)
    if load_object_store is not None:
        name = list(load_object_store.keys())[0]
        kwargs = load_object_store[name]
        load_object_store = build_object_store(name, kwargs)

    if 'run_name' in cfg:
        run_name = cfg['run_name']
    else:
        run_name = os.environ['COMPOSER_RUN_NAME']

    # Build the Trainer
    trainer = Trainer(
        run_name=run_name,
        seed=cfg.seed,
        model=model,
        train_dataloader=train_loader,
        eval_dataloader=eval_loader,
        train_subset_num_batches=cfg.get('train_subset_num_batches', -1),
        eval_subset_num_batches=cfg.get('eval_subset_num_batches', -1),
        optimizers=optimizer,
        schedulers=scheduler,
        max_duration=cfg.max_duration,
        eval_interval=cfg.eval_interval,
        loggers=loggers,
        callbacks=callbacks,
        precision=cfg.precision,
        device=cfg.get('device', None),
        grad_clip_norm=cfg.grad_clip_norm,
        grad_accum=cfg.get('grad_accum', 'auto'),
        # fsdp_config=fsdp_config,
        save_folder=cfg.get('save_folder', None),
        save_filename=cfg.get('save_filename', None),
        save_interval=cfg.get('save_interval', '1000ba'),
        save_num_checkpoints_to_keep=cfg.get('save_num_checkpoints_to_keep', -1),
        save_overwrite=cfg.get('save_overwrite', None),
        load_path=cfg.get('load_path', None),
        load_object_store=load_object_store,
        load_weights_only=cfg.get('load_weights_only', False),
    )

    print("Logging config...")
    config_dict = om.to_container(cfg, resolve=True)
    config_dict.update({
        'n_gpus': dist.get_world_size(),
        'n_params': n_params,
        'device_train_batch_size': device_train_batch_size,
        'device_eval_batch_size': device_eval_batch_size,
        # 'device_eval_microbatch_size': device_eval_microbatch_size,
    })
    if wandb.run is not None:
        wandb.config.update(config_dict)

    print("Starting training...")
    trainer.fit()


if __name__ == '__main__':
    yaml_path, args_list = sys.argv[1], sys.argv[2:]
    with open(yaml_path) as f:
        yaml_cfg = om.load(f)
    cli_cfg = om.from_cli(args_list)
    cfg = om.merge(yaml_cfg, cli_cfg)
    main(cfg)
