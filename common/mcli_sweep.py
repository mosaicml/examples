

from omegaconf import OmegaConf as om
import omegaconf
from typing import Dict, List, Tuple
import itertools
import sys
from mcli.sdk import RunConfig, create_run
import time
import pandas as pd
import os

def insert_subconfig(base_config, path, subconfig):
    recursive_parent_config = base_config
    for node in path[:-1]:
        recursive_parent_config = recursive_parent_config[node]
    recursive_parent_config[path[-1]] = subconfig
    return base_config


def compose_param_choices(
    base_config: omegaconf.dictconfig.DictConfig,
    sweep_params: omegaconf.dictconfig.DictConfig
):
    paths = []
    param_configs = []
    for key, val in sweep_params.items():
        path = key.split('.')
        paths.append(path)
        param_configs.append(val)
    
    result_configs = []
    for param_combo in itertools.product(*param_configs):
        new_config = base_config.copy()
        info = {}
        for path, param_sub_cfg in zip(paths, param_combo):
            info['-'.join(path)] = param_sub_cfg.key
            new_config = insert_subconfig(new_config, path, param_sub_cfg.value)
        
        result_configs.append(
            {
                "info": info,
                "config": new_config
            }
        )

    return result_configs         

def make_name(base_name, info):
    name = base_name
    for k,v in info.items():
        name += f"-{k.split('-')[-1]}-{v}"
    
    name += f"-{str(abs(hash(time.time())))[:4]}"
    return name.replace('.', '-').replace('_', '-')


def launch_jobs(base_name, configs):
    jobs = []
    run_configs = []
    for config_info in configs:
        info = config_info['info']
        config = config_info['config']
        config = om.to_container(config, resolve=True)
        config = RunConfig.from_dict(config)
        name = make_name(base_name, info)
        config.name = name
        config.run_name = name
        info['run_name'] = config.run_name
        jobs.append(info)
        run_configs.append(config)
    
    input1 = input(f"Are you sure you want to submit {len(jobs)} jobs to cluster {config.cluster}? (y/n) ")
    if input1 == 'y':
        for config in run_configs:
            create_run(config)
        

    return jobs

def build_sweep_configs(yaml):
    base_config = yaml.base_config
    sweep_config = yaml.sweep_config

    return compose_param_choices(base_config, sweep_config)


if __name__ == "__main__":
    yaml_path, out_path = sys.argv[1], sys.argv[2]
    with open(yaml_path) as f:
        yaml_cfg = om.load(f)
    sweep_configs = build_sweep_configs(yaml_cfg)
    jobs = launch_jobs("gpt_eval", sweep_configs)
    df = pd.DataFrame(jobs)

    if os.path.exists(out_path):
        with open(out_path, 'a') as f:
            df.to_csv(f, sep='\t', index=None, header=None)
    else:
        with open(out_path, 'w') as f:
            df.to_csv(f, sep='\t', index=None)