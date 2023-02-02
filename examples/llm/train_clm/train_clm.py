# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import warnings

from composer import Trainer
from composer.utils import reproducibility
from omegaconf import OmegaConf as om

from examples.common.builders import (build_algorithm, build_callback,
                                      build_dataloader, build_logger,
                                      build_optimizer, build_scheduler)
from examples.common.config_utils import log_config, update_batch_size_info
from examples.llm.src.huggingface.hf_causal_lm import ComposerHFCausalLM
from examples.llm.src.mosaic_gpt.model import ComposerMosaicGPT


def build_composer_model(cfg):
    if cfg.name == 'mosaic_gpt':
        return ComposerMosaicGPT(cfg)
    elif cf.gname == 'hf_causal_lm':
        return ComposerHFCausalLM(cfg)
    else:
        raise ValueError(f'Not sure how to build model with name={cfg.name}')


def main(cfg):
    # Filter deprecation warning from torch internal
    warnings.filterwarnings(
        action='ignore',
        category=UserWarning,
        message=
        f'torch.distributed.*_base is a private function and will be deprecated.*'
    )

    # Seed the environment
    reproducibility.seed_all(cfg.seed)

    # Run Name
    cfg.run_name = cfg.get('run_name', os.environ.get('COMPOSER_RUN_NAME',
                                                      'llm'))

    # Get batch size info
    cfg = update_batch_size_info(cfg)

    # Read FSDP Config as a dict
    fsdp_config = cfg.get('fsdp_config', None)
    fsdp_config = om.to_container(fsdp_config,
                                  resolve=True) if fsdp_config else None

    # Restrict model init device to 'meta' and 'cpu',
    # using 'cuda' vs. 'cuda:id' is tricky and can lead to common user errors
    # when multiple GPUs are available.
    # Also 'meta' is only valid when using FSDP
    assert cfg.model.init_device in ['meta', 'cpu']
    if fsdp_config is None and cfg.model.init_device == 'meta':
        print(
            "\nUsing init device `cfg.model.init_device='meta'` is only valid when using FSDP! "
            "Reverting to `cfg.model.init_device='cpu'`.")
        cfg.model.init_device = 'cpu'

    # Build Model
    print('Initializing model...')
    model = build_composer_model(cfg.model)
    cfg.n_params = sum(p.numel() for p in model.parameters())
    print(f'n_params={cfg.n_params:.2e}')
    if hasattr(model, 'num_fwd_flops'):
        print(f'nun_fwd_flops={model.num_fwd_flops:.2e}')

    # Dataloaders
    print('\nBuilding train loader...')
    train_loader = build_dataloader(cfg.train_loader,
                                    cfg.device_train_batch_size)
    print('Building eval loader...')
    eval_loader = build_dataloader(cfg.eval_loader, cfg.device_eval_batch_size)

    # Optimizer
    optimizer = build_optimizer(cfg.optimizer, model)

    # Scheduler
    scheduler = build_scheduler(cfg.scheduler)

    # Loggers
    loggers = [
        build_logger(name, logger_cfg)
        for name, logger_cfg in (cfg.get('loggers') or {}).items()
    ]

    # Callbacks
    callbacks = [
        build_callback(name, callback_cfg)
        for name, callback_cfg in (cfg.get('callbacks') or {}).items()
    ]

    # Algorithms
    algorithms = [
        build_algorithm(name, algorithm_cfg)
        for name, algorithm_cfg in (cfg.get('algorithms') or {}).items()
    ]

    # Build the Trainer
    print('\nBuilding trainer...')
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
        eval_subset_num_batches=cfg.get('eval_subset_num_batches', -1),
        progress_bar=cfg.get('progress_bar', False),
        log_to_console=cfg.get('log_to_console', True),
        console_log_interval=cfg.get('console_log_interval', '1ba'),
        loggers=loggers,
        callbacks=callbacks,
        precision=cfg.precision,
        algorithms=algorithms,
        device_train_microbatch_size=cfg.get('device_train_microbatch_size',
                                             'auto'),
        fsdp_config=fsdp_config,  # type: ignore
        save_folder=cfg.get('save_folder', None),
        save_interval=cfg.get('save_interval', '1000ba'),
        save_num_checkpoints_to_keep=cfg.get('save_num_checkpoints_to_keep',
                                             -1),
        save_overwrite=cfg.get('save_overwrite', False),
        load_path=cfg.get('load_path', None),
        load_weights_only=cfg.get('load_weights_only', False),
    )

    print('\nLogging config...')
    log_config(cfg)

    print('\nStarting training...')
    trainer.fit()

    print('\nDone.')


if __name__ == '__main__':
    yaml_path, args_list = sys.argv[1], sys.argv[2:]
    with open(yaml_path) as f:
        yaml_cfg = om.load(f)
    cli_cfg = om.from_cli(args_list)
    cfg = om.merge(yaml_cfg, cli_cfg)
    main(cfg)
