# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

"""Example script to finetune a Stable Diffusion Model."""

from pathlib import Path
import os
import sys

import torch
from callbacks import LogDiffusionImages, SaveClassImages
from composer import Trainer
from composer.algorithms import EMA
from composer.callbacks import LRMonitor, MemoryMonitor, SpeedMonitor
from composer.optim import ConstantScheduler
from composer.utils import dist, reproducibility
from data import build_prompt_dataloader, build_dreambooth_dataloader
from model import build_stable_diffusion_model
from omegaconf import DictConfig, OmegaConf

from examples.common import build_logger, calculate_batch_size_info, log_config


def main(config: DictConfig):  # type: ignore
    reproducibility.seed_all(config.seed)
    device = 'gpu' if torch.cuda.is_available() else 'cpu'
    if dist.get_world_size(
    ) > 1:  # initialize the pytorch distributed process group if training on multiple gpus.
        dist.initialize_dist(device)

    if config.grad_accum == 'auto' and device == 'cpu':
        raise ValueError(
            'grad_accum="auto" requires training with a GPU; please specify grad_accum as an integer'
        )
    # calculate batch size per device and add it to config (These calculations will be done inside the composer trainer in the future)
    config.train_device_batch_size, _, _ = calculate_batch_size_info(
        config.global_train_batch_size, 'auto')
    config.eval_device_batch_size, _, _ = calculate_batch_size_info(
        config.global_eval_batch_size, 'auto')

    print('Building Composer model')
    model = build_stable_diffusion_model(
        model_name_or_path=config.model.name,
        train_text_encoder=config.model.train_text_encoder,
        train_unet=config.model.train_unet,
        num_images_per_prompt=config.model.num_images_per_prompt,
        image_key=config.model.image_key,
        caption_key=config.model.caption_key)
    
    if config.use_prior_preservation:
        class_images_dir = Path(config.dataset.class_data_root)
        if not class_images_dir.exists():
            class_images_dir.mkdir(parents=True, exist_ok=True)
        cur_class_images = len(list(class_images_dir.iterdir()))

        if cur_class_images < config.num_class_images:
            # duplicate the class prompt * num class samples to generate class images
            images_to_generate = config.num_class_images - cur_class_images
            class_prompts = [config.dataset.class_prompt] * images_to_generate
            prompt_dataloader = build_prompt_dataloader(class_prompts,
                                                        batch_size=config.eval_device_batch_size)
            save_class_images = SaveClassImages(class_data_root=config.dataset.class_data_root)
            
            model.num_images_per_prompt = 1 # set for prior preservation image generation
            trainer = Trainer(model=model, eval_dataloader=prompt_dataloader, callbacks=save_class_images)
            trainer.eval() # eval run will save images via the SaveClassImages callback
            model.num_images_per_prompt = config.model.num_images_per_prompt

    # Train dataset
    print('Building dataloaders')
    train_dataloader = build_dreambooth_dataloader(
        instance_data_root=config.dataset.instance_data_root,
        instance_prompt=config.dataset.instance_prompt,
        use_prior_preservation=config.use_prior_preservation, 
        class_data_root=config.dataset.get("class_data_root"),
        class_prompt=config.dataset.get("class_prompt"),
        resolution=config.dataset.resolution,
        center_crop=config.dataset.center_crop,
        tokenizer=model.tokenizer,
        batch_size=config.train_device_batch_size)

    # Eval dataset
    eval_dataloader = build_prompt_dataloader(
        config.dataset.eval_prompts, batch_size=config.eval_device_batch_size)

    # Optimizer
    print('Building optimizer and learning rate scheduler')
    optimizer = torch.optim.AdamW(params=model.parameters(),
                                  lr=config.optimizer.lr,
                                  weight_decay=config.optimizer.weight_decay)

    # Constant LR for fine-tuning
    lr_scheduler = ConstantScheduler()

    print('Building loggers')
    loggers = [
        build_logger(name, logger_config)
        for name, logger_config in config.loggers.items()
    ]

    # Callbacks for logging
    print('Building Speed, LR, and Memory monitoring callbacks')
    # Measures throughput as samples/sec and tracks total training time
    speed_monitor = SpeedMonitor(window_size=50)
    lr_monitor = LRMonitor()  # Logs the learning rate
    memory_monitor = MemoryMonitor()  # Logs memory utilization
    # Logs images generated from prompts in the eval set
    image_logger = LogDiffusionImages()

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
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
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
        grad_accum=config.grad_accum,
        seed=config.seed)

    print('Logging config')
    log_config(config)

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
