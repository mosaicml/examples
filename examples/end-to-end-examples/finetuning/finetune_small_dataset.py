from enum import Enum
from mcli.sdk import create_run, RunConfig
from utils import get_mpt_parameters_dict

# from config_registry import CONFIG_REGISTRY

class Models(Enum):
  MPT_7B = 'mpt-7b'
  MPT_30B = 'mpt-30b'

# User fills these in
MODEL_NAME: Models = Models.MPT_7B # mpt-30b or mpt-7b
TRAIN_DATA_PATH: str = 'mosaicml/dolly_hhrlhf' # blobstore path or hugging face path to training data
EVAL_DATA_PATH: str = 'mosaicml/dolly_hhrlhf' # hugging face path to eval data
SAVE_FOLDER: str = '' # where to upload checkpoints
CLUSTER_NAME: str = ''

# For testing
run_name = f'finetune-{MODEL_NAME}'
SAVE_FOLDER = f'oci://nancy-test/checkpoints/{run_name}/'
CLUSTER_NAME = 'r1z1'

def main(model_name: Models, train_data_path: str, eval_data_path: str, save_folder: str, cluster_name: str):
    # Set up the finetuning run config
    parameters = get_mpt_parameters_dict(
        model_name=model_name.value,
        train_data_path=train_data_path,
        eval_data_path=eval_data_path,
        save_folder=save_folder)
    command: str = 'cd llm-foundry/scripts && composer train/train.py /mnt/config/parameters.yaml'
    integrations: list = [{
        'integration_type': 'git_repo',
        'git_repo': 'mosaicml/llm-foundry',
        'pip_install': '-e .[gpu]',
        'ssh_clone': False, # Should be true if using a private repo
    }]

    run_config: RunConfig = None
    gpu_num: int = None
    gpu_type: str = None
    if model_name == Models.MPT_7B:
        gpu_num = 8
        gpu_type = 'a100_80gb'
    elif model_name == Models.MPT_30B:
        gpu_num = 16
        gpu_type = 'a100_80gb'
    else:
        raise ValueError(f'Invalid model name: {model_name}')
    
    run_config = RunConfig(
        run_name=run_name,
        gpu_num=gpu_num,
        gpu_type=gpu_type,
        cluster=cluster_name,
        image='mosaicml/pytorch:1.13.1_cu117-python3.10-ubuntu20.04',
        integrations=integrations,
        command=command,
        parameters=parameters)

    # Run the finetuning job
    run = create_run(run_config)
    print(f'Created finetuning run with name: {run.name}. Tail the logs with `mcli logs --follow {run.name}`')


if __name__ == '__main__':
    main(
        model_name=MODEL_NAME,
        train_data_path=TRAIN_DATA_PATH,
        cluster_name=CLUSTER_NAME,
        save_folder=SAVE_FOLDER)
