# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

import os
import sys

from composer import Trainer
from composer.utils import dist, reproducibility
from omegaconf import OmegaConf as om
from transformers import Adafactor

from examples.common.builders import build_algorithm
from examples.common.builders import build_callback as _build_callback
from examples.common.builders import build_logger
from examples.common.builders import build_optimizer as _build_optimizer
from examples.common.builders import build_scheduler as _build_scheduler
from examples.common.logging_utils import log_config
from examples.ul2.src.data_denoising import build_text_denoising_dataloader
from examples.ul2.src.hf_prefix_lm import create_hf_prefix_lm
from examples.ul2.src.hf_t5 import create_hf_t5
from examples.ul2.src.inverse_sqrt_scheduler import InverseSquareRootScheduler
from examples.ul2.src.mod_print_callback import \
    MixtureOfDenoisersPrinterCallback
from examples.ul2.src.super_glue.data import build_super_glue_task_dataloader


def build_callback(name, kwargs):
    """Adds mixture-of-denoiser printer callback to common callbacks."""
    if name == 'mod_printer':
        return MixtureOfDenoisersPrinterCallback(**kwargs)
    else:
        return _build_callback(name, kwargs)


def build_optimizer(cfg, model):
    """Adds Adafactor to common optimizers."""
    if cfg.name == 'adafactor':
        return Adafactor(
            params=model.parameters(),
            lr=cfg.get(
                'lr', 1.0
            ),  # Recommend using InverseSquareRootScheduler with default settings when using these defaults
            weight_decay=cfg.get('weight_decay', 0.0),
            beta1=cfg.get('beta1', None),
            scale_parameter=cfg.get('scale_parameter', True),
            relative_step=cfg.get('relative_step', False),
            warmup_init=cfg.get('warmup_init', False))
    else:
        return _build_optimizer(cfg, model)


def build_scheduler(cfg):
    """Adds inverse-square-root to common schedulers."""
    if cfg.name == 'inverse_square_root':
        return InverseSquareRootScheduler(
            alpha_max=cfg.alpha_max,
            scale=cfg.get('scale', 1.0),
        )
    else:
        return _build_scheduler(cfg)


def build_model(cfg):
    """Constructs a HuggingFace T5 or PrefixLM."""
    if cfg.name == 'hf_t5':
        return create_hf_t5(
            pretrained_model_name=cfg.pretrained_model_name,
            use_pretrained=cfg.get('use_pretrained', None),
            model_config=cfg.get('model_config', None),
            tokenizer_name=cfg.get('tokenizer_name', None),
            z_loss=cfg.get('z_loss', 0.0),
            task_finetuning=cfg.get('task_finetuning', False),
        )
    elif cfg.name == 'hf_prefix_lm':
        return create_hf_prefix_lm(
            pretrained_model_name=cfg.pretrained_model_name,
            tokenizer_name=cfg.tokenizer_name,
            use_pretrained=cfg.get('use_pretrained', None),
            model_config=cfg.get('model_config', None),
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

    # Read FSDP Config as a dict
    fsdp_config = cfg.get('fsdp_config', None)
    fsdp_config = om.to_container(fsdp_config,
                                  resolve=True) if fsdp_config else None

    # Build Model
    print('Initializing model...')
    model = build_model(cfg.model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'{n_params=:.4e}')

    # Get batch size info
    if cfg.global_train_batch_size % dist.get_world_size() != 0:
        raise ValueError(
            f'Global batch size {cfg.global_train_batch_size} is not divisible by {dist.get_world_size()} '
            'as a result, the batch size would be truncated, please adjust `global_train_batch_size` '
            f'to be divisible by world size, {dist.get_world_size()}.')
    device_train_batch_size = cfg.global_train_batch_size // dist.get_world_size(
    )
    device_eval_batch_size = cfg.get(
        'global_eval_batch_size',
        cfg.global_train_batch_size) // dist.get_world_size()

    # Dataloaders
    print('Building train loader...')
    train_loader = build_dataloader(cfg.train_loader,
                                    device_train_batch_size,
                                    mode='train')
    print('Building eval loader...')
    eval_loader = build_dataloader(cfg.eval_loader,
                                   device_eval_batch_size,
                                   mode='eval')

    # Optimizer
    optimizer = build_optimizer(cfg.optimizer, model)

    # Scheduler
    scheduler = build_scheduler(cfg.scheduler)

    # Loggers
    loggers = [
        build_logger(name, logger_cfg)
        for name, logger_cfg in cfg.get('loggers', {}).items()
    ]

    # Callbacks
    callbacks = [
        build_callback(name, callback_cfg)
        for name, callback_cfg in cfg.get('callbacks', {}).items()
    ]

    # Algorithms
    algorithms = [
        build_algorithm(name, algorithm_cfg)
        for name, algorithm_cfg in cfg.get('algorithms', {}).items()
    ]

    if 'run_name' in cfg:
        run_name = cfg['run_name']
    else:
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
        device=cfg.get('device', None),
        grad_accum=cfg.get('grad_accum', 'auto'),
        save_folder=cfg.get('save_folder', None),
        save_interval=cfg.get('save_interval', '1000ba'),
        save_num_checkpoints_to_keep=cfg.get('save_num_checkpoints_to_keep',
                                             -1),
        load_path=cfg.get('load_path', None),
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
