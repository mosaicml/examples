# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

"""Example script to finetune a Stable Diffusion Model."""

import os
import sys

import torch
from callbacks import LogDiffusionImages
from composer import Trainer
from composer.algorithms import EMA
from composer.callbacks import LRMonitor, MemoryMonitor, SpeedMonitor
from composer.loggers import WandBLogger
from composer.optim import ConstantScheduler
from composer.utils import dist, reproducibility
from data import build_hf_image_caption_datapsec, build_prompt_dataspec
from model import build_stable_diffusion_model
from omegaconf import DictConfig, OmegaConf


def log_config(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    if 'wandb' in cfg.get('loggers', {}):
        try:
            import wandb
        except ImportError as e:
            raise e
        if wandb.run:
            wandb.config.update(OmegaConf.to_container(cfg, resolve=True))


def build_logger(name, kwargs):
    if name == 'wandb':
        return WandBLogger(**kwargs)
    else:
        raise ValueError(f'Not sure how to build logger: {name}')


def main(config: DictConfig):  # type: ignore
    reproducibility.seed_all(config.seed)
    device = 'gpu' if torch.cuda.is_available() else 'cpu'
    if dist.get_world_size(
    ) > 1:  # initialize the pytorch distributed process group if training on multiple gpus.
        dist.initialize_dist(device)

    if config.device_train_microbatch_size == 'auto' and device == 'cpu':
        raise ValueError(
            'device_train_microbatch_size="auto" requires training with a GPU; please specify device_train_microbatch_size as an integer'
        )
    # calculate batch size per device and add it to config (These calculations will be done inside the composer trainer in the future)

    print('Building Composer model')
    model = build_stable_diffusion_model(
        model_name_or_path=config.model.name,
        train_text_encoder=config.model.train_text_encoder,
        train_unet=config.model.train_unet,
        num_images_per_prompt=config.model.num_images_per_prompt,
        image_key=config.model.image_key,
        caption_key=config.model.caption_key,
    )

    # Train dataset
    print('Building dataloaders')
    train_dataspec = build_hf_image_caption_datapsec(
        name=config.dataset.name,
        resolution=config.dataset.resolution,
        tokenizer=model.tokenizer,
        mean=config.dataset.mean,
        std=config.dataset.std,
        image_column=config.dataset.image_column,
        caption_column=config.dataset.caption_column,
        batch_size=config.global_train_batch_size // dist.get_world_size(),
    )

    # Eval dataset
    eval_dataspec = build_prompt_dataspec(
        config.dataset.prompts,
        batch_size=config.get('eval_device_batch_size',
                              config.global_train_batch_size) //
        dist.get_world_size(),
    )

    # Optimizer
    print('Building optimizer and learning rate scheduler')
    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=config.optimizer.lr,
        weight_decay=config.optimizer.weight_decay,
    )

    # Constant LR for fine-tuning
    lr_scheduler = ConstantScheduler()

    print('Building loggers')
    loggers = [
        build_logger(name, logger_config)
        for name, logger_config in config.loggers.items()
    ]

    # Callbacks for logging
    print('Building Speed, LR, and Memory monitoring callbacks')
    speed_monitor = SpeedMonitor(
        window_size=50
    )  # Measures throughput as samples/sec and tracks total training time
    lr_monitor = LRMonitor()  # Logs the learning rate
    memory_monitor = MemoryMonitor()  # Logs memory utilization
    image_logger = LogDiffusionImages(
    )  # Logs images generated from prompts in the eval set

    print('Building algorithms')
    if config.use_ema:
        algorithms = [EMA(half_life='100ba', update_interval='20ba')]
    else:
        algorithms = None

    # Create the Trainer!
    print('Building Trainer')
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
        precision=config.precision,
        device_train_microbatch_size=config.device_train_microbatch_size,
        seed=config.seed)

    print('Logging config')
    log_config(config)

    print('Eval!')
    trainer.eval()  # show outputs from model without fine-tuning.

    print('Train!')
    trainer.fit()
    return trainer  # Return trainer for testing purposes.


if __name__ == '__main__':
    if len(sys.argv) < 2 or not os.path.exists(sys.argv[1]):
        raise ValueError('The first argument must be a path to a yaml config.')

    yaml_path, args_list = sys.argv[1], sys.argv[2:]
    with open(yaml_path) as f:
        yaml_config = OmegaConf.load(f)
    cli_config = OmegaConf.from_cli(args_list)
    config = OmegaConf.merge(yaml_config, cli_config)
    main(config)  # type: ignore
