
import sys
from enum import Enum
from mcli.sdk import create_run, RunConfig

# from config_registry import CONFIG_REGISTRY

# User fills these in
# TODO: make this work with argparse so the user doesn't have to modify the python file
MODEL_NAME = '' # mpt-30b or mpt-7b
TRAIN_DATA_PATH = '' # blobstore path to training data
SAVE_FOLDER = '' # where to upload checkpoints

class Models(Enum):
  MPT_7B = 'mpt-7b'
  MPT_30B = 'mpt-30b'


def main(model_name: str, train_data_path: str, save_folder: str):
    # Load the finetuning config
    config: RunConfig = None
    if model_name == Models.MPT_7B.value:
        # TODO: Copy / move this YAML to the same folder as this python script
        # Note that we use this one because it loads a model with pretrained: true
        config = RunConfig.from_file('../sec_10k_qa/mcli-yamls/03_finetune_on_10ks.yaml')
    elif model_name == Models.MPT_30B.value:
        # TODO: load 30B config
        config = RunConfig.from_file('../sec_10k_qa/')
        # config = CONFIG_REGISTRY[Models.MPT_30B.value]
    else:
        print(f'Error unsupported model {model_name} please choose one of {Models.MPT_7B.value} or {Models.MPT_30B.value}')
        sys.exit(1)

    # Fill in the config with user-supplied values
    config.parameters['data_remote'] = train_data_path
    config.parameters['save_folder'] = save_folder

    # Run the finetuning job
    run = create_run(config)

    # TODO: Monitor the run until it is complete

    # TODO: When complete, deploy the model






if __name__ == '__main__':

    main(MODEL_NAME, TRAIN_DATA_PATH, SAVE_FOLDER)
