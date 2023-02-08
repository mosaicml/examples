# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0
"""Example script to finetune a Stable Diffusion Model."""

import os
import sys
from omegaconf import OmegaConf, DictConfig

import torch
from composer import Trainer
from composer.utils import reproducibility
from composer.algorithms import EMA
from composer.callbacks import LRMonitor, MemoryMonitor, SpeedMonitor
from composer.optim import ConstantScheduler
from composer.utils import dist

from data import build_hf_image_caption_datapsec, build_prompt_dataspec
from model import build_stable_diffusion_model
from callbacks import LogDiffusionImages
from common.builders import build_logger
from common.logging_utils import log_config



def main(config: DictConfig):
    reproducibility.seed_all(config.seed)
    device = 'gpu' if torch.cuda.is_available() else 'cpu'
    if config.grad_accum == 'auto' and not torch.cuda.is_available():
        raise ValueError('grad_accum="auto" requires training with a GPU; please specify grad_accum as an integer')
    # Divide batch sizes by number of devices if running multi-gpu training
    train_batch_size = config.dataset.global_train_batch_size
    eval_batch_size = config.dataset.global_eval_batch_size

    # If there is more than one device, initialize dist to ensure dataset 
    # is only downloaded by rank 0 and divide global batch size by num devices
    if dist.get_world_size() > 1:
        dist.initialize_dist(device, timeout=180)
        train_batch_size //= dist.get_world_size()
        eval_batch_size //= dist.get_world_size()

    print('Building Composer model')
    model = build_stable_diffusion_model(
        model_name_or_path=config.model.name,
        train_text_encoder=config.model.train_text_encoder,
        train_unet=config.model.train_unet,
        num_images_per_prompt=config.model.num_images_per_prompt,
        image_key=config.model.image_key,
        caption_key=config.model.caption_key)

    # Train dataset
    print('Building dataloader')
    train_dataspec = build_hf_image_caption_datapsec(
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

    # Constant LR for fine-tuning
    lr_scheduler = ConstantScheduler()
    print('Built optimizer and learning rate scheduler\n')
    
    loggers = [build_logger(name, logger_config) for name, logger_config in config.loggers.items()]
    
    # Callbacks for logging
    print('Building Speed, LR, and Memory monitoring callbacks')
    speed_monitor = SpeedMonitor(window_size=50)  # Measures throughput as samples/sec and tracks total training time
    lr_monitor = LRMonitor()  # Logs the learning rate
    memory_monitor = MemoryMonitor()  # Logs memory utilization
    image_logger = LogDiffusionImages()  # Logs images generated from prompts in the eval set
    print('Built Speed, LR, and Memory monitoring callbacks\n')


    print('Building algorithms')
    if config.use_ema:
        algorithms = [EMA(half_life='100ba', update_interval='20ba')]
    else:
        algorithms = None
    print('Built algorithms')

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
    return trainer  # Return trainer for testing purposes


if __name__ == '__main__':
    if len(sys.argv) < 2 or not os.path.exists(sys.argv[1]):
        raise ValueError('The first argument must be a path to a yaml config.')

    yaml_path, args_list = sys.argv[1], sys.argv[2:]
    with open(yaml_path) as f:
        yaml_config = OmegaConf.load(f)
    cli_config = OmegaConf.from_cli(args_list)
    config = OmegaConf.merge(yaml_config, cli_config)
    main(config)
