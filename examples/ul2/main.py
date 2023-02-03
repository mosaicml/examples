# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

import os
import sys

from composer import Trainer
from composer.utils import reproducibility
from omegaconf import OmegaConf as om
from transformers import Adafactor

from examples.common import builders
from examples.common.config_utils import log_config, update_batch_size_info
from examples.ul2.src import (InverseSquareRootScheduler,
                              MixtureOfDenoisersPrinterCallback,
                              build_super_glue_task_dataloader,
                              build_text_denoising_dataloader,
                              create_hf_prefix_lm, create_hf_t5)


def build_callback(name, cfg):
    """Adds mixture-of-denoiser printer callback to common callbacks."""
    if name == 'mod_printer':
        return MixtureOfDenoisersPrinterCallback(**cfg)
    return builders.build_callback(name, cfg)


def build_optimizer(cfg, model):
    """Adds Adafactor to common optimizers."""
    if cfg.name == 'adafactor':
        return Adafactor(
            params=model.parameters(),
            lr=cfg.get(
                'lr', 1.0
            ),  # Recommend using InverseSquareRootScheduler with default settings when using these defaults
            weight_decay=cfg.get('weight_decay', 0.0),
            beta1=cfg.get('beta1'),
            scale_parameter=cfg.get('scale_parameter', True),
            relative_step=cfg.get('relative_step', False),
            warmup_init=cfg.get('warmup_init', False))
    return builders.build_optimizer(cfg, model)


def build_scheduler(cfg):
    """Adds inverse-square-root to common schedulers."""
    if cfg.name == 'inverse_square_root':
        return InverseSquareRootScheduler(
            alpha_max=cfg.alpha_max,
            scale=cfg.get('scale', 1.0),
        )
    return builders.build_scheduler(cfg)


def build_model(cfg):
    """Constructs a HuggingFace T5 or PrefixLM."""
    if cfg.name == 'hf_t5':
        return create_hf_t5(
            pretrained_model_name=cfg.pretrained_model_name,
            use_pretrained=cfg.get('use_pretrained'),
            model_config=cfg.get('model_config'),
            tokenizer_name=cfg.get('tokenizer_name'),
            z_loss=cfg.get('z_loss', 0.0),
            task_finetuning=cfg.get('task_finetuning', False),
        )
    elif cfg.name == 'hf_prefix_lm':
        return create_hf_prefix_lm(
            pretrained_model_name=cfg.pretrained_model_name,
            tokenizer_name=cfg.tokenizer_name,
            use_pretrained=cfg.get('use_pretrained'),
            model_config=cfg.get('model_config'),
            z_loss=cfg.get('z_loss', 0.0),
            task_finetuning=cfg.get('task_finetuning', False),
        )
    else:
        raise ValueError(f'Not sure how to build model with name={cfg.name}')


def build_dataloader(cfg, device_batch_size, mode):
    """Constructs a text_denoising or super_glue dataloader."""
    if cfg.name == 'text_denoising':
        return build_text_denoising_dataloader(cfg, device_batch_size)
    elif cfg.name == 'super_glue':
        return build_super_glue_task_dataloader(cfg, device_batch_size, mode)
    else:
        raise ValueError(
            f'Not sure how to build dataloader with name={cfg.name}')


def main(cfg):
    print('Training using config: ')
    print(om.to_yaml(cfg))
    reproducibility.seed_all(cfg.seed)

    cfg = update_batch_size_info(cfg)

    # Read FSDP Config as a dict
    fsdp_config = cfg.get('fsdp_config')
    fsdp_config = om.to_container(fsdp_config,
                                  resolve=True) if fsdp_config else None

    # Build Model
    print('Initializing model...')
    model = build_model(cfg.model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'{n_params=:.4e}')

    # Dataloaders
    print('Building train loader...')
    train_loader = build_dataloader(cfg.train_loader,
                                    cfg.device_train_batch_size,
                                    mode='train')
    print('Building eval loader...')
    eval_loader = build_dataloader(cfg.eval_loader,
                                   cfg.device_eval_batch_size,
                                   mode='eval')

    # Optimizer and scheduler
    optimizer = build_optimizer(cfg.optimizer, model)
    scheduler = build_scheduler(cfg.scheduler)

    # Loggers
    loggers = [
        builders.build_logger(name, logger_cfg)
        for name, logger_cfg in cfg.get('loggers', {}).items()
    ]

    # Callbacks
    callbacks = [
        build_callback(name, callback_cfg)
        for name, callback_cfg in cfg.get('callbacks', {}).items()
    ]

    # Algorithms
    algorithms = [
        builders.build_algorithm(name, algorithm_cfg)
        for name, algorithm_cfg in cfg.get('algorithms', {}).items()
    ]

    run_name = cfg.get('run_name')
    if run_name is None:
        run_name = os.environ['COMPOSER_RUN_NAME']

    # Build the Trainer
    trainer = Trainer(
        run_name=run_name,
        seed=cfg.seed,
        model=model,
        algorithms=algorithms,
        train_dataloader=train_loader,
        eval_dataloader=eval_loader,
        train_subset_num_batches=cfg.get('train_subset_num_batches', -1),
        eval_subset_num_batches=cfg.get('eval_subset_num_batches', 1000),
        fsdp_config=fsdp_config,  # type: ignore
        optimizers=optimizer,
        schedulers=scheduler,
        max_duration=cfg.max_duration,
        eval_interval=cfg.eval_interval,
        progress_bar=cfg.progress_bar,
        log_to_console=cfg.log_to_console,
        console_log_interval=cfg.get('console_log_interval', '50ba'),
        loggers=loggers,
        callbacks=callbacks,
        precision=cfg.precision,
        device=cfg.get('device'),
        device_train_microbatch_size=cfg.device_train_microbatch_size,
        save_folder=cfg.get('save_folder'),
        save_interval=cfg.get('save_interval', '1000ba'),
        save_num_checkpoints_to_keep=cfg.get('save_num_checkpoints_to_keep',
                                             -1),
        load_path=cfg.get('load_path'),
        load_weights_only=cfg.get('load_weights_only', False),
    )

    print('Logging config...')
    log_config(cfg)

    print('Starting training...')
    trainer.fit()

    print('Done.')


if __name__ == '__main__':
    yaml_path, args_list = sys.argv[1], sys.argv[2:]
    with open(yaml_path) as f:
        yaml_cfg = om.load(f)
    cli_cfg = om.from_cli(args_list)
    cfg = om.merge(yaml_cfg, cli_cfg)
    main(cfg)
