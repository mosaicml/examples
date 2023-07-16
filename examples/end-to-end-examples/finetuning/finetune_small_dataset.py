from enum import Enum
from mcli.sdk import create_run, create_inference_deployment, get_run, \
                    InferenceDeployment, InferenceDeploymentConfig, RunConfig, Run
from utils import get_mpt_parameters_dict, check_run_status_and_raise_if_error, validate_oci_bucket
from utils_convert_to_mds import convert_data
from datetime import datetime

class Models(Enum):
  MPT_7B = 'mpt-7b'
  MPT_30B = 'mpt-30b'

# User fills these in
# For data conversion
INPUT_DATA_PATH: str = ''
# For finetuning
MODEL_NAME: Models = Models.MPT_7B # mpt-30b or mpt-7b
TRAIN_DATA_PATH: str = f'oci://nancy-test' # Directory where your jsonl, pq, or csv file is stored
# TRAIN_DATA_PATH: str = 'mosaicml/dolly_hhrlhf' # blobstore path or hugging face path to training data
EVAL_DATA_PATH: str = 'mosaicml/dolly_hhrlhf' # hugging face path to eval data
SAVE_FOLDER: str = 'oci://nancy-test/checkpoints/finetune-mpt-7b' # where to upload checkpoints
CLUSTER_NAME: str = 'r1z1'
# For inference
# HUGGING_FACE_PATH: str = f'oci://nancy-test/checkpoints/hf-path-{datetime.utcnow().strftime("%d-%m-%y::%H:%M:%S")}'
HUGGING_FACE_PATH: str = 'oci://nancy-test/checkpoints/hf-path-08-07-23::04:06:08'
INFERENCE_CLUSTER_NAME = 'r7z14'
NUM_REPLICAS: int = 1

# TODO: remove, this is for testing
run_name = f'finetune-{MODEL_NAME.value}'
CLUSTER_TO_GPU_MAP = {
    'r1z1': 'a100_80gb',
    'r7z2': 'a100_40gb',
}


###### Finetuning ######
def finetune(
        model_name: Models,
        train_data_path: str,
        eval_data_path: str,
        save_folder: str,
        cluster_name: str
) -> Run:
    """Finetune a model on a small dataset
    
    Args:
        model_name (Models): Model to finetune
        train_data_path (str): Path to training data
        eval_data_path (str): Path to eval data
        save_folder (str): Where to save checkpoints
        cluster_name (str): Name of cluster to run finetuning job on
    """
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
        gpu_num = 24
        gpu_type = 'a100_40gb'
        parameters['optimizer']['name'] = 'decoupled_lionw'
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
    print(f'Created finetuning run with name: {run.name}.\n Tail the logs with `mcli logs --follow {run.name}`')
    return run

###### Inference ######
def convert_composer_to_hf(
    run_name: str,
    cluster_name: str,
    checkpoint_path: str,
    hugging_face_path: str,
    gpu_num: int,
    gpu_type: str,
) -> Run:
    """Convert a composer checkpoint to a hugging face checkpoint

    Args:
        run_name (str): Name of the run
        cluster_name (str): Name of the cluster to run the job on
        checkpoint_path (str): Path to the finetuned model checkpoint
        hugging_face_path (str): Path to save the hugging face checkpoint
    """
    integrations : list = [{
        'integration_type': 'git_repo',
        'git_repo': 'mosaicml/llm-foundry',
        'git_commit': 'v0.2.0',
        'pip_install': '-e .',
        'ssh_clone': False, # Should be true if using a private repo
    }]
    command = f"""
        cd llm-foundry/scripts/inference && \
        python convert_composer_to_hf.py \
            --composer_path {checkpoint_path} \
            --hf_output_path {hugging_face_path} \
            --output_precision bf16
            """
    run_config = RunConfig(
        run_name=f'{run_name.lower()}-convert-to-hf',
        gpu_num=gpu_num,
        gpu_type=gpu_type,
        cluster=cluster_name,
        image='mosaicml/pytorch:1.13.1_cu117-python3.10-ubuntu20.04', # Use the Docker image provided by MosaicML,
        integrations=integrations,
        command=command,
    )
    run = create_run(run_config)
    print(f'Created convert to hugging face run with name: {run.name}.\n Tail the logs with `mcli logs --follow {run.name}`')
    return run


def create_deployment(
    cluster_name: str,
    num_replicas: int,
    hugging_face_path: str,
    model_name: str,
) -> InferenceDeployment:
    compute_config = {
        'cluster': cluster_name,
        'instance': 'oci.vm.gpu.a10.1',
        'gpu_type': 'a10',
        'gpus': 1}
    
    integrations = [{
        'integration_type': 'git_repo',
        'git_repo': 'mosaicml/examples',
        'git_branch': 'main',
        'ssh_clone': False,
    },
    {
        'integration_type': 'git_repo',
        'git_repo': 'mosaicml/llm-foundry',
        'git_commit': '2c92faa5ce31888214bdb582ac7f5756d0d3dacd',
        'ssh_clone': False,
    }]

    command = """
        export PYTHONPATH=$PYTHONPATH:/code/llm-foundry:/code/examples:/code
        pip uninstall packaging -y
        rm /usr/lib/python3/dist-packages/packaging-23.1.dist-info/REQUESTED
        pip install composer[streaming,libcloud,oci]==0.14.1
        pip install packaging==23.1
    """

    model = {
        'downloader': 'examples.end-to-end-examples.sec_10k_qa.deployment_download_helper.download_and_convert',
        'download_parameters': {
            'remote_uri': hugging_face_path
        },
        'model_handler': 'examples.inference-deployments.mpt.mpt_ft_handler.MPTFTModelHandler', # Use the provided MPT handler
        'model_parameters': {
            'ft_lib_path': '/code/FasterTransformer/build/lib/libth_transformer.so',
            'model_name_or_path': f'mosaicml/{model_name}',
            'gpus': 1,
        },
        'backend': 'faster_transformers',
    }

    deployment_config = InferenceDeploymentConfig(
        name=f'finetune-deployment-{model_name}',
        compute=compute_config,
        integrations=integrations,
        replicas=num_replicas,
        command=command,
        model=model,
    )
    deployment = create_inference_deployment(deployment_config)
    print(f'Created inference deployment with name: {deployment.name}, DNS: {deployment.public_dns}')
    print(f'Check the status with: `mcli get deployments` and logs with `mcli get deployment logs --follow {deployment.name}`')
    return deployment


if __name__ == '__main__':
    # Data conversion
    convert_data(
        tokenizer_name=f'mosaicml/{MODEL_NAME.value}',
        output_folder='oci://nancy-test',
        train_index_path='oci://nancy-test/10k_train.jsonl',
        eval_index_path='oci://nancy-test/10k_validate.jsonl',
        concat_tokens=0,
        eos_text='</s>',
        bos_text='<s>',
        no_wrap=True,
        max_workers=1,
        compression='gzip'
    )

    # Validation
    # if using oci
    # validate_oci_bucket(bucket_name='nancy-test', object_name=TRAIN_DATA_PATH)

    # Finetune
    # run = finetune(
    #     model_name=MODEL_NAME,
    #     train_data_path=TRAIN_DATA_PATH,
    #     eval_data_path=EVAL_DATA_PATH,
    #     cluster_name=CLUSTER_NAME,
    #     save_folder=SAVE_FOLDER)

    # check_run_status_and_raise_if_error(run)
    
    # # For testing
    # run = get_run('finetune-mpt-7b-bsrsNL')
    # checkpoint_path = f'{SAVE_FOLDER}/latest-rank0.pt.symlink'
    # validate_oci_bucket(bucket_name='nancy-test', filepath=checkpoint_path)
    
    # # Prep the model for inference
    # conversion_run = convert_composer_to_hf(
    #     run_name=run.name,
    #     cluster_name=CLUSTER_NAME,
    #     checkpoint_path=f'{SAVE_FOLDER}/latest-rank0.pt.symlink',
    #     hugging_face_path=HUGGING_FACE_PATH,
    #     gpu_num=8,
    #     gpu_type=CLUSTER_TO_GPU_MAP[CLUSTER_NAME],
    # )

    # check_run_status_and_raise_if_error(conversion_run)

    # print(f'Checkpoint converted to Hugging Face format and saved to {HUGGING_FACE_PATH}')

    run = get_run('finetune-mpt-7b-bsrsnl-convert-to-hf-dDdgPM')

    # [Optional]: Create an inference deployment with your model
    print(f'Starting inference deployment...')
    deployment: InferenceDeployment = create_deployment(
        cluster_name=INFERENCE_CLUSTER_NAME,
        num_replicas=NUM_REPLICAS,
        hugging_face_path=HUGGING_FACE_PATH,
        model_name=MODEL_NAME.value,
    )
    # TODO: wait until deployment is ready and return the endpoint
    # TODO: deploy instruct model

    # TODO: Evaluate the model
