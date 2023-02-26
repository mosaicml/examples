# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

import os
import warnings

import pytest
import torch
from composer import Trainer
from composer.core import Evaluator
from composer.utils import dist, get_device, reproducibility
from omegaconf import OmegaConf as om

from examples.common.builders import (build_algorithm, build_callback,
                                      build_dataloader, build_logger,
                                      build_optimizer, build_scheduler)
from examples.common.config_utils import log_config
from examples.llm.src.model_registry import COMPOSER_MODEL_REGISTRY


def gpt_tiny_moe_cfg(conf_path='yamls/mosaic_gpt/125m_moe.yaml',
                     pyramid=False,
                     pg=None):
    """Create gpt tiny moe cfg."""
    with open(conf_path) as f:
        test_cfg = om.load(f)
    test_cfg.train_loader.dataset.split = 'train_small'

    test_cfg.model.init_device = 'cpu'

    bsize = 2
    test_cfg.device_eval_batch_size = bsize
    test_cfg.device_train_microbatch_size = bsize
    test_cfg.global_train_batch_size = 2 * bsize * dist.get_world_size()

    test_cfg.max_duration = '4ba'
    test_cfg.eval_interval = '4ba'
    test_cfg.eval_subset_num_batches = 2
    test_cfg.save_interval = '4ba'
    test_cfg.run_name = 'gpt-moe-test'
    test_cfg.max_seq_len = 256

    test_cfg.tokenizer.args.max_seq_len = test_cfg.max_seq_len
    test_cfg.train_loader.dataset.max_seq_len = test_cfg.max_seq_len
    test_cfg.eval_loader.dataset.max_seq_len = test_cfg.max_seq_len

    test_cfg.model.max_seq_len = test_cfg.max_seq_len
    test_cfg.model.d_model = 32
    test_cfg.model.n_heads = 2
    test_cfg.model.n_layers = 5
    if pyramid:
        test_cfg.model.moe.num_experts = [2]
        for _ in range(test_cfg.model.n_layers - 1):
            experts = 2 * test_cfg.model.moe.num_experts[-1]
            test_cfg.model.moe.num_experts += [experts]
    else:
        test_cfg.model.moe.num_experts = 4

    if pg is not None:
        test_cfg.model.moe.pg = pg
    else:
        test_cfg.model.moe.pg = 'self'

    return test_cfg


def check_tensors_different_across_ranks(tensor):
    """Verify tenor is different across ranks.

    Parallelization methods such as FSDP or DDP might sync parameters or
    gradients. This method allows us to verify that tensors are not sync'd
    across ranks
    """
    world_size = dist.get_world_size()
    tensors = dist.all_gather(tensor)
    for i in range(world_size):
        for j in range(i + 1, world_size):
            if tensors[i].equal(tensors[j]):
                raise RuntimeError(f'tensors are equal on ranks {i=} and {j=}')


def test_tutel_moe_expert_notsync(pyramid=False,
                                  pg=None,
                                  check_tensor_diff=True):
    if not os.path.isdir('./my-copy-c4/val') or not os.path.isdir(
            './my-copy-c4/train_small'):
        pytest.xfail('c4 dataset not set up as expected')
    if not torch.cuda.is_available() and not dist.get_world_size() > 1:
        pytest.xfail('test requires multiple GPUs')

    cfg = gpt_tiny_moe_cfg(conf_path='yamls/mosaic_gpt/125m_moe.yaml',
                           pyramid=pyramid,
                           pg=pg)
    print(f'{pyramid=}, {pg=}')

    reproducibility.seed_all(cfg.seed)
    dist.initialize_dist(get_device(None))

    # Read FSDP Config as a dict
    fsdp_config = cfg.get('fsdp_config', None)
    fsdp_config = om.to_container(fsdp_config,
                                  resolve=True) if fsdp_config else None

    # Restrict model init_device to 'meta' and 'cpu',
    # using 'cuda' vs. 'cuda:id' is tricky and can lead to common user errors
    # when multiple GPUs are available.
    # Also 'meta' is only valid when using FSDP
    init_device = cfg.model.get('init_device', 'cpu')
    assert init_device in ['meta', 'cpu']
    if fsdp_config is None and init_device == 'meta':
        warnings.warn(
            "Using `cfg.model.init_device='meta'` is only valid when using FSDP! "
            "Reverting to `cfg.model.init_device='cpu'`.")
        cfg.model.init_device = 'cpu'

    # Build Model
    print('Initializing model...')
    model = COMPOSER_MODEL_REGISTRY[cfg.model.name](cfg.model)
    if hasattr(model, 'param_count'):
        cfg.n_params = model.param_count
    else:
        cfg.n_params = sum(p.numel() for p in model.parameters())
    print(f'{cfg.n_params=:.2e}')
    if hasattr(model, 'num_fwd_flops'):
        print(f'{model.num_fwd_flops=:.2e}')

    model = model.cuda()

    if check_tensor_diff:
        for n, m in model.model.named_modules():
            if n.endswith('.mlp.moe.experts'):
                for _n, p in m.named_parameters():
                    # verify expert parameters are initialized independently
                    check_tensors_different_across_ranks(p)
                    assert p.grad is None

    # Build the Trainer
    print('Building trainer...')
    trainer = Trainer(
        run_name=cfg.run_name,
        seed=cfg.seed,
        model=model,
        train_dataloader=build_dataloader(cfg.train_loader,
                                          cfg.device_train_microbatch_size),
        eval_dataloader=[
            Evaluator(label='eval',
                      dataloader=build_dataloader(cfg.eval_loader,
                                                  cfg.device_eval_batch_size),
                      metric_names=list(model.train_metrics.keys()))
        ],
        optimizers=build_optimizer(cfg.optimizer, model),
        schedulers=build_scheduler(cfg.scheduler),
        max_duration=cfg.max_duration,
        eval_interval=cfg.eval_interval,
        eval_subset_num_batches=cfg.get('eval_subset_num_batches', -1),
        progress_bar=cfg.get('progress_bar', False),
        log_to_console=cfg.get('log_to_console', True),
        console_log_interval=cfg.get('console_log_interval', '1ba'),
        loggers=[
            build_logger(name, logger_cfg)
            for name, logger_cfg in (cfg.get('loggers') or {}).items()
        ],
        callbacks=[
            build_callback(name, callback_cfg)
            for name, callback_cfg in (cfg.get('callbacks') or {}).items()
        ],
        precision=cfg.precision,
        algorithms=[
            build_algorithm(name, algorithm_cfg)
            for name, algorithm_cfg in (cfg.get('algorithms') or {}).items()
        ],
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

    print('Logging config...')
    log_config(cfg)

    if check_tensor_diff:
        for n, m in trainer.state.model.model.named_modules():
            if n.endswith('.mlp.moe.experts'):
                for _n, p in m.named_parameters():
                    # verify expert parameters are initialized independently
                    check_tensors_different_across_ranks(p)
                    assert p.grad is None

    print('Starting training...')
    trainer.fit()

    if check_tensor_diff:
        for n, m in trainer.state.model.model.named_modules():
            if n.endswith('.mlp.moe.experts'):
                for _n, p in m.named_parameters():
                    # verify expert parameters not expert parameter gradients are sync'd
                    check_tensors_different_across_ranks(p)
                    check_tensors_different_across_ranks(p.grad)

    trainer.close()

    print('Done.')


if __name__ == '__main__':
    for pyramid, pg, check_tensor_diff in (
        (False, None, True),
        (True, None, True),
            # # test fsdp custom process groups
        (False, 'self', True),
        (True, 'self', True),
        (False, 'node', False),
        (True, 'node', False),
        (False, 'local_rank_across_nodes', False),
        (True, 'local_rank_across_nodes', False),
        (False, 'set1', True),
        (False, 'set2', False),
        (True, 'set2', False),
        (False, 'set4', False),
        (True, 'set4', False),
        (False, 'mod2', False),
        (True, 'mod2', False),
        (False, 'mod4', False),
        (True, 'mod4', False),
    ):
        test_tutel_moe_expert_notsync(pyramid=pyramid,
                                      pg=pg,
                                      check_tensor_diff=check_tensor_diff)
