# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0
"""Example script to finetune a Stable Diffusion Model."""

import os
import sys
from typing import Dict

import torch
from composer import Trainer
from composer.algorithms import EMA
from composer.callbacks import LRMonitor, MemoryMonitor, SpeedMonitor
from composer.loggers import ProgressBarLogger, WandBLogger
from composer.optim import ConstantScheduler
from composer.utils import dist
from data import build_image_caption_datapsec, build_prompt_dataspec
from model import build_stable_diffusion_model
from callbacks import LogDiffusionImages
from omegaconf import DictConfig, OmegaConf


def build_logger(name: str, kwargs: Dict):
    if name == 'progress_bar':
        return ProgressBarLogger()
    elif name == 'wandb':
        return WandBLogger(**kwargs)
    else:
        raise ValueError(f'Not sure how to build logger: {name}')


def log_config(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    if 'wandb' in cfg.loggers:
        try:
            import wandb
        except ImportError as e:
            raise e
        if wandb.run:
            wandb.config.update(OmegaConf.to_container(cfg, resolve=True))


def main(config):
    if config.grad_accum == 'auto' and not torch.cuda.is_available():
        raise ValueError(
            'grad_accum="auto" requires training with a GPU; please specify grad_accum as an integer'
        )
    # Divide batch sizes by number of devices if running multi-gpu training
    train_batch_size = config.dataset.train_batch_size
    eval_batch_size = config.dataset.eval_batch_size

    # Initialize dist to ensure dataset is only downloaded by rank 0
    device = 'gpu' if torch.cuda.is_available() else 'cpu'
    if dist.get_world_size() > 1:
        dist.initialize_dist(device, 180)
        train_batch_size //= dist.get_world_size()
        train_batch_size = train_batch_size or 1

        eval_batch_size //= dist.get_world_size()
        eval_batch_size = eval_batch_size or 1

    print('Building Composer model')
    model = build_stable_diffusion_model(
        model_name_or_path=config.model.name,
        train_text_encoder=config.model.train_text_encoder,
        num_images_per_prompt=config.model.num_images_per_prompt,
        image_key=config.model.image_key,
        caption_key=config.model.caption_key)

    # Train dataset
    print('Building dataloader')
    train_dataspec = build_image_caption_datapsec(
        name=config.dataset.name,
        resolution=config.dataset.resolution,
        tokenizer=model.tokenizer,
        mean=config.dataset.mean,
        std=config.dataset.std,
        image_column=config.dataset.image_column,
        caption_column=config.dataset.caption_column,
        batch_size=train_batch_size)

    # Eval dataset
    eval_dataspec = build_prompt_dataspec(config.dataset.prompts,
                                          batch_size=eval_batch_size)

    # Optimizer
    print('Building optimizer and learning rate scheduler')

    optimizer = torch.optim.AdamW(params=model.parameters(),
                                  lr=config.optimizer.lr,
                                  weight_decay=config.optimizer.weight_decay)

    # Learning rate scheduler: LR warmup for 8 epochs, then cosine decay for the rest of training

    lr_scheduler = ConstantScheduler()
    print('Built optimizer and learning rate scheduler\n')

    # Callbacks for logging
    print('Building Speed, LR, and Memory monitoring callbacks')
    speed_monitor = SpeedMonitor(
        window_size=50
    )  # Measures throughput as samples/sec and tracks total training time
    lr_monitor = LRMonitor()  # Logs the learning rate
    memory_monitor = MemoryMonitor()  # Logs memory utilization

    image_logger = LogDiffusionImages()  # Logs images generated from prompts in the eval set

    # Callback for checkpointing
    print('Built Speed, LR, and Memory monitoring callbacks\n')

    print('Building algorithms')
    if config.use_ema:
        algorithms = [EMA(half_life='100ba', update_interval='20ba')]
    else:
        algorithms = None
    print('Built algorithms')

    loggers = [
        build_logger(name, logger_config)
        for name, logger_config in config.loggers.items()
    ]

    # Create the Trainer!
    print('Building Trainer')
    device = 'gpu' if torch.cuda.is_available() else 'cpu'
    precision = 'amp' if device == 'gpu' else 'fp32'  # Mixed precision for fast training when using a GPU
    trainer = Trainer(
        run_name=config.run_name,
        model=model,
        train_dataloader=train_dataspec,
        eval_dataloader=eval_dataspec,
        optimizers=optimizer,
        schedulers=lr_scheduler,
        algorithms=algorithms,
        loggers=loggers,
        max_duration=config.max_duration,
        eval_interval=config.eval_interval,
        callbacks=[speed_monitor, lr_monitor, memory_monitor, image_logger],
        save_folder=config.save_folder,
        save_interval=config.save_interval,
        save_num_checkpoints_to_keep=config.save_num_checkpoints_to_keep,
        load_path=config.load_path,
        device=device,
        precision=precision,
        grad_accum=config.grad_accum,
        seed=config.seed)
    print('Built Trainer\n')

    print('Logging config')
    log_config(config)

    print('Train!')
    trainer.fit()

    # Return trainer for testing purposes
    return trainer


if __name__ == '__main__':
    if len(sys.argv) < 2 or not os.path.exists(sys.argv[1]):
        raise ValueError('The first argument must be a path to a yaml config.')

    yaml_path, args_list = sys.argv[1], sys.argv[2:]
    with open(yaml_path) as f:
        yaml_config = OmegaConf.load(f)
    cli_config = OmegaConf.from_cli(args_list)
    config = OmegaConf.merge(yaml_config, cli_config)
    main(config)
