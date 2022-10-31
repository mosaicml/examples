import copy
import multiprocessing as mp
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor as Pool
from multiprocessing.managers import SyncManager
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
from urllib.parse import urlparse

import numpy as np
import omegaconf as om
import torch
from finetuning_jobs import COLAJob, MNLIJob, MRPCJob, QNLIJob, QQPJob, RTEJob, SST2Job, STSBJob

from composer.callbacks import LRMonitor, SpeedMonitor
from composer.loggers import WandBLogger
from composer.optim import LinearWithWarmupScheduler
from composer.utils import reproducibility
from composer.utils.file_helpers import get_file
from composer.utils.object_store import S3ObjectStore

TASK_NAME_TO_CLASS = {
    'mnli': MNLIJob,
    'rte': RTEJob,
    'mrpc': MRPCJob,
    'qnli': QNLIJob,
    'qqp': QQPJob,
    'sst2': SST2Job,
    'stsb': STSBJob,
    'cola': COLAJob
}


def build_logger(name, kwargs):
    if name == 'wandb':
        return WandBLogger(**kwargs)
    else:
        raise ValueError(f'Not sure how to build logger: {name}')


def build_callback(name, kwargs):
    if name == 'lr_monitor':
        return LRMonitor()
    elif name == 'speed_monitor':
        return SpeedMonitor()
    else:
        raise ValueError(f'Not sure how to build callback: {name}')


def build_scheduler(cfg):
    if cfg.name == 'linear_with_warmup':
        return LinearWithWarmupScheduler(t_warmup=cfg.t_warmup)
    else:
        raise ValueError(f'Not sure how to build scheduler: {cfg.name}')


def get_values_from_path(path: str, separator: str = '/') -> Dict[str, str]:
    """Parses out information from a path/string that looks like ...<separator>key=value<separator..."""
    dict_output = {}
    underscore_split = path.split(separator)
    for item in underscore_split:
        if '=' not in item:
            continue

        key, value = item.split('=')
        dict_output[key] = value
    return dict_output


def get_checkpoint_name_from_path(path: str) -> str:
    """To go from checkpoint name to path, replace | with /"""
    return path.lstrip('/').replace('/', '|')


def download_starting_checkpoint(starting_checkpoint_load_path: List[str],
                                 local_pretrain_checkpoints_folder: str) -> Tuple[List[str], List[str]]:
    """Downloads the pretrained checkpoints to start from. Currently only supports S3 and URLs"""
    load_object_store = None
    parsed_path = urlparse(starting_checkpoint_load_path)
    if parsed_path.scheme == "s3":
        load_object_store = S3ObjectStore(bucket=parsed_path.netloc)
    local_checkpoint_paths = []

    download_path = parsed_path.path if parsed_path.scheme == "s3" else starting_checkpoint_load_path
    os.makedirs(local_pretrain_checkpoints_folder, exist_ok=True)
    local_path = os.path.join(local_pretrain_checkpoints_folder, get_checkpoint_name_from_path(parsed_path.path))
    if not os.path.exists(local_path):
        get_file(destination=local_path, path=download_path, object_store=load_object_store, progress_bar=True)
    local_checkpoint_paths.append(local_path)

    return local_checkpoint_paths


def _setup_gpu_queue(num_gpus: int, manager: SyncManager):
    """Returns a queue with [0, 1, .. num_gpus]."""
    gpu_queue = manager.Queue(num_gpus)
    for gpu_id in range(num_gpus):
        gpu_queue.put(gpu_id)
    return gpu_queue


def create_job_configs(main_config: om.DictConfig, tasks_to_run: Set[str], pretrained_checkpoint_path: str):
    configs = []
    for task_name, task_config in main_config.tasks:
        if task_name not in tasks_to_run:
            continue
        for task_seed in task_config.get('seeds', [main_config.default_seed]):
            run_name = f'{main_config.base_run_name}_task={task_name}_seed={str(task_seed)}'
            logger_configs = copy.deepcopy(main_config.get('loggers', {}))
            for logger_name, logger_config in logger_configs.items():
                if logger_name == 'wandb':
                    logger_config['group'] = main_config.base_run_name
                    logger_config['name'] = run_name
            task_seed_config = om.OmegaConf.create({
                'job_name':
                    run_name,
                'seed':
                    task_seed,
                'pretrained_model_name':
                    main_config.pretrained_model_name,
                'tokenizer_name':
                    main_config.pretrained_model_name,
                'scheduler':
                    main_config.scheduler,
                'load_path':
                    pretrained_checkpoint_path,
                'save_folder':
                    os.path.join(main_config.save_finetune_checkpoint_folder, f'task={task_name}', f'seed={task_seed}'),
                'loggers':
                    logger_configs,
                'callbacks':
                    main_config.get('callbacks', {}),
                'precision':
                    main_config.get('precision', None),
            })
            configs.append(task_seed_config)

    return configs


def run_job_worker(config: om.DictConfig, gpu_queue: Optional[mp.Queue] = None) -> Any:
    """Instantiates the job object and runs it"""
    instantiated_job = TASK_NAME_TO_CLASS[config.task](
        job_name=config.job_name,
        seed=config.seed,
        pretrained_model_name=config.pretrained_model_name,
        tokenizer_name=config.tokenizer_name,
        scheduler=build_scheduler(config.scheduler),
        load_path=config.load_path,
        save_folder=config.save_folder,
        loggers=[build_logger(name, logger_config) for name, logger_config in config.loggers],
        callbacks=[build_callback(name, callback_config) for name, callback_config in config.callbacks],
        precision=config.precision,
    )
    return instantiated_job.run(gpu_queue)


def run_jobs_parallel(configs: Sequence[om.DictConfig]):
    """Runs a list of jobs (passed in as Hydra configs) across GPUs.

    Returns a dictionary mapping job name to the result and original config
    Each job's results is a dict of:

    * 'checkpoints': list of saved_checkpoints, if any,
    * 'metrics': nested dict of results, accessed by
                 dataset and metric name, e.g.
                 ``metrics['glue_mnli']['Accuracy']``.
    * 'job_name': The job name, helpful for keeping track of results during multiprocessing
    """
    num_gpus = torch.cuda.device_count()
    results = []

    with mp.Manager() as manager:
        # workers get gpu ids from this queue
        # to set the GPU to run on
        gpu_queue = _setup_gpu_queue(num_gpus, manager)

        ctx = mp.get_context('spawn')
        with Pool(max_workers=min(num_gpus, len(configs)), mp_context=ctx) as pool:
            results = pool.map(run_job_worker, [config for config in configs], [gpu_queue for _ in configs])

    job_name_to_config = {config.job_name: config for config in configs}
    finished_results = {}
    for result in results:
        job_name = result['job_name']
        finished_results[job_name] = {'result': result, 'config': job_name_to_config[job_name]}

    return finished_results


def run_jobs_serial(configs):
    """Runs the jobs serially, rather than in parallel. Useful for debugging"""
    results = {}
    for config in configs:
        result = run_job_worker(config)
        results[config.job_name] = {'result': result, 'config': config}
    return results


def format_job_name(job_name: str) -> str:
    """Formats the job name for pretty printing"""
    dict_output = get_values_from_path(job_name, separator='_')
    return f'{dict_output["task"].upper()}(seed={dict_output["seed"]})'


def _print_table(results: Dict[str, Dict[str, Any]]):
    """Pretty prints a table given a results dictionary."""
    header = '{job_name:50}| {eval_task:25}| {name:20}|'
    hyphen_count = 50 + 25 + 20 + 11
    row_format = header + ' {value:.2f}'
    print('\nCollected Job Results: \n')
    print('-' * hyphen_count)
    print(header.format(job_name='Job', eval_task='Dataset', name='Metric'))
    print('-' * hyphen_count)
    for job_name, result in results.items():
        for eval_task, eval_results in result['result']['metrics'].items():
            for name, metric in eval_results.items():
                print(
                    row_format.format(
                        job_name=format_job_name(job_name),
                        eval_task=eval_task,
                        name=name,
                        value=metric * 100,
                    ))
    print('-' * hyphen_count)
    print('\n')


def _print_averaged_glue_results(glue_results: List[Tuple[Tuple[str, str], float]]):
    """Pretty prints a table of glue results averaged across seeds"""
    header = '{job_name:50}|'
    hyphen_count = 50 + 11
    row_format = header + ' {value:.2f}'
    print('\nCollected Job Results: \n')
    print('-' * hyphen_count)
    print(header.format(job_name='Task'))
    print('-' * hyphen_count)
    for (task_name, ckpt_idx), result in glue_results:
        print(row_format.format(
            job_name=f'{task_name.upper()}(ckpt-idx={ckpt_idx})',
            value=result,
        ))
    print('-' * hyphen_count)
    print('\n')


def train(config: om.DictConfig) -> None:
    """
    Args:
        config (DictConfig): Configuration composed by Hydra
    """
    start_time = time.time()

    # Initial default seed
    reproducibility.seed_all(config.default_seed)

    # Quiet down WandB
    os.environ["WANDB_SILENT"] = "true"

    # Set tokenizer parallelism
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Downloads the starting checkpoint ahead of time so that
    # the different tasks don't all try to download it at the same time
    local_pretrain_checkpoint_path = download_starting_checkpoint(config.starting_checkpoint_load_path,
                                                                  config.local_pretrain_checkpoint_folder)

    # Gets the dictionary mapping task name to random seeds to run it over
    task_seeds = config.get('task_seeds', {})

    # Builds round 1 configs and runs them
    round_1_task_names = {'cola', 'sst2', 'qqp', 'qnli', 'mnli'}
    round_1_job_configs = create_job_configs(config, round_1_task_names, local_pretrain_checkpoint_path)

    round_1_results = {}
    if len(round_1_job_configs) > 0:
        if config.parallel:
            round_1_results = run_jobs_parallel(round_1_job_configs)
        else:
            round_1_results = run_jobs_serial(round_1_job_configs)

    # Builds up the information needed to run the second round, starting from the MNLI checkpoints
    mnli_checkpoint_path = None
    for job_name, output_dict in round_1_results.items():
        job_results = output_dict['result']
        job_values = get_values_from_path(job_name, separator='_')
        task_name = job_values['task']

        if task_name != 'mnli':
            continue

        mnli_checkpoint_path = job_results['checkpoints'][-1]
        break

    # Builds round 2 configs and runs them
    round_2_task_names = {'rte', 'mrpc', 'stsb'}
    round_2_job_configs = create_job_configs(config, round_2_task_names, mnli_checkpoint_path)

    round_2_results = {}
    if len(round_2_job_configs) > 0:
        if config.parallel:
            round_2_results = run_jobs_parallel(round_2_job_configs)
        else:
            round_2_results = run_jobs_serial(round_2_job_configs)

    end_time = time.time()

    print('-' * 30)
    print(f'Training completed in {(end_time-start_time):.2f} seconds')
    print('-' * 30)

    # Join the results and pretty print them
    all_results = round_1_results | round_2_results
    _print_table(all_results)

    # Average the GLUE results across seeds and pretty print them
    glue_results = defaultdict(list)
    for job_name, result in all_results.items():
        job_values = get_values_from_path(job_name, separator='_')
        for _, eval_results in result['result']['metrics'].items():
            for _, metric in eval_results.items():
                glue_results[(job_values['task'], job_values['ckpt-idx'])].append(metric * 100)
    glue_results = {key: np.mean(values) for key, values in glue_results.items()}

    overall_glue = defaultdict(list)
    for (_, ckpt_idx), average_metric in glue_results.items():
        overall_glue[('glue', ckpt_idx)].append(average_metric)
    overall_glue = {key: np.mean(values) for key, values in overall_glue.items()}

    _print_averaged_glue_results([(key, value) for key, value in glue_results.items()] +
                                 [(key, value) for key, value in overall_glue.items()])


if __name__ == '__main__':
    yaml_path, args_list = sys.argv[1], sys.argv[2:]
    with open(yaml_path) as f:
        yaml_cfg = om.load(f)
    cli_cfg = om.from_cli(args_list)
    cfg = om.merge(yaml_cfg, cli_cfg)
    train(cfg)
