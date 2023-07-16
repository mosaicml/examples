from mcli.sdk import Run, RunStatus, get_run
import oci
from time import sleep
from datetime import datetime
import sys

def check_run_status_and_raise_if_error(run: Run) -> None:
  terminated_run_states = [RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.STOPPED, RunStatus.UNKNOWN, RunStatus.FAILED_PULL]
    
  start_time = datetime.now()
  while run.status not in terminated_run_states:
      sleep(10)
      run = get_run(run.name)
      print(f'Run status: {run.status}, elapsed time: {datetime.now() - start_time}')
  
  if run.status != 'COMPLETED':
      print(f'Run failed with status: {run.status}')
      sys.exit(1)
  print(f'Run completed successfully with status: {run.status}, elapsed time: {datetime.now() - start_time}')

def get_mpt_parameters_dict(model_name: str, save_folder: str, train_data_path: str, eval_data_path: str) -> dict:
  return {
    # Checkpoint to local filesystem or remote object store
    'save_interval': '500ba', # How frequently to save checkpoints
    # save_num_checkpoints_to_keep: 1  # Important, this cleans up checkpoints saved to DISK
    # 'save_folder': f'./{run_name}/checkpoints',
    'save_folder': save_folder,
    'dist_timeout': 60000, # Set large dist_timeout to allow for checkpoint uploading on a slow connection

    # Maximum sequence length of the model
    # For MPT, you can change this to a different number if you would like to train on longer sequences
    # Note that you would also need to reprocess your data to contain longer sequences
    'max_seq_len': 2048,

    # Random seed to ensure reproducibility
    'global_seed': 17,

    # Model
    # This section is used by LLM-foundry to construct the model
    'model':
      {'name': 'hf_causal_lm',
      'init_device': 'mixed', # Initially only create the model on CPU once per node to reduce system memory requirements
      'pretrained_model_name_or_path': f'mosaicml/{model_name}',
      'pretrained': True, # We will load the Composer checkpoint from the previous step, so don't need the original weights
      'config_overrides': # Override the default model config (comment this out if you change the model from MPT)
        {'attn_config':
          {'attn_impl': 'triton', # Use the triton implementation of attention
          # We are not concatenating samples together, so this is False
          # If you turn on packing_ratio below, you may want to set this to True
          # to restrict attention to within each concatenated sequence
          # Setting this to True does use some additional GPU memory, so for a large model
          # and/or large max sequence length, you may want to leave it False.
          # In our runs, we have successfully trained models both with and without this set to True.
          'attn_uses_sequence_id': False}}},

    # Tokenizer
    # This section is used by LLM-foundry to construct the tokenizer
    'tokenizer':
      {'name': f'mosaicml/{model_name}',
      'kwargs':
        {'model_max_length': 2048}},

    # Dataloaders
    # Here we are using the finetuning dataloader
    # see LLM-foundry scripts/train/finetune_example for more details on finetuning data
    'train_loader':
      {'name': 'finetuning',
      'dataset': {
        'hf_name': train_data_path,
        'split': 'train',
        'max_seq_len': 2048,
        'allow_pad_trimming': False,
        'decoder_only_format': True,
        # Use `python llmfoundry/data/packing.py --yaml-path /path/to/this/yaml/ ...`
        # to profile this run's optimal packing_ratio as it depends on GPU count,
        # batch size, sequence length. Turning on packing by setting packing_ratio
        # here will pack multiple examples into one sequence for increased efficiency.
        # For the mosaicml/dolly_hhrlhf and max sequence length 2048,
        # 3 is a good packing_ratio that will not cause any examples to be trimmed.
        # As an approximate rule of thumb, if you, for example, double max_seq_len you can double packing_ratio.
        # packing_ratio:
        'shuffle': True,
        'shuffle_seed': 17,
      },
        
      'drop_last': True,
      'num_workers': 8,
      'pin_memory': False,
      'prefetch_factor': 2,
      'persistent_workers': True,
      'timeout': 0},

    'eval_loader':
      {'name': 'finetuning',
      'dataset':
        {'hf_name': eval_data_path,
        'split': 'test',
        'max_seq_len': 2048,
        'allow_pad_trimming': False,
        'decoder_only_format': True,
        # packing_ratio:
        'shuffle': True},
      'drop_last': True,
      'num_workers': 8,
      'pin_memory': False,
      'prefetch_factor': 2,
      'persistent_workers': True,
      'timeout': 0},

    # Learning rate scheduler
    # see LLM-foundry llmfoundry/utils/builders.py::build_scheduler for other built-in options
    'scheduler':
      {'name': 'linear_decay_with_warmup',
      't_warmup': '50ba',
      'alpha_f': 0},

    # Optimizer
    # see LLM-foundry llmfoundry/utils/builders.py::build_optimizer for other built-in options
    'optimizer':
      {# Based on Dolly
      'name': 'decoupled_adamw',
      'lr': 5.0e-6,
      'betas': [0.9, 0.999],
      'eps': 1.0e-8,
      'weight_decay': 0},

    # Algorithms to apply
    # see LLM-foundry llmfoundry/utils/builders.py::build_algorithm for other built-in options
    'algorithms':
      {'gradient_clipping':
        {'clipping_type': 'norm',
        'clipping_threshold': 1.0}},

    # Run configuration
    # TODO: change to 2ep for final run
    'max_duration': '10ba', # Maximum duration of the run. Change to something shorter (e.g. 10ba) for a quick test run
    'eval_interval': '500ba', # How frequently to evaluate the model
    'eval_first': True, # Whether to evaluate the model before training
    'eval_subset_num_batches': -1, # How many batches to evaluate on. -1 means evaluate on the entire dataset
    'global_train_batch_size': 128,  # Global batch size. This is the batch size across all GPUs
    'seed': 17,
    'device_eval_batch_size': 8, # Evaluation batch size per GPU
    # TODO: change to 'auto' for final run
    'device_train_microbatch_size': 1,
    # 'device_train_microbatch_size': 'auto', # Automatically determine the microbatch size per GPU
    'precision': 'amp_bf16',

    # Configuration settings for FSDP
    # https://docs.mosaicml.com/projects/composer/en/latest/notes/distributed_training.html#fullyshardeddataparallel-fsdp
    # for more information about FSDP in Composer
    'fsdp_config':
      {'sharding_strategy': 'FULL_SHARD',
      'mixed_precision': 'PURE',
      'activation_checkpointing': True,
      'activation_checkpointing_reentrant': False,
      'activation_cpu_offload': False,
      'limit_all_gathers': True,
      'verbose': False},

    # Logging configuration
    'progress_bar': False,
    'log_to_console': True,
    'console_log_interval': '1ba',
    'python_log_level': 'debug',

    # Uncomment to log to WandB
    # see LLM-foundry llmfoundry/utils/builders.py::build_logger for other built-in options
    # loggers:
    #   wandb: {}

    # Callbacks
    # see LLM-foundry llmfoundry/utils/builders.py::build_callback for other built-in options
    'callbacks':
      # Periodically generate text from the model, uncomment only if you are logging to WandB
      # generate_callback:
      #   batch_log_interval: 500
      #   do_sample: True
      #   max_new_tokens: 100
      #   prompts:
      #   - The quick brown fox jumps over
      #   - |-
      #     Vegan Banana Bread
      #     Instructions:
      #     1.
      #   - The other day I was explaining what generative AI is to my five year old.
      #   - We are a global semiconductor company primarily offering
      #   - Our company had revenue of
      #   - Our business operations are subject to numerous risks, including
      #   - What was AMD's revenue in 2019?
      #   - What is net operating income?
      #   temperature: 1
      #   top_k: 50
      #   top_p: 0.95
      #   use_cache: True
      # Log information about the processing speed of the model
      {'speed_monitor':
        {'window_size': 10},
      # Log the learning rate over the course of training
      'lr_monitor': {},
      # Log information about the memory usage of the model
      'memory_monitor': {},
      # Log an estimate of how long the training run has left to complete
      'runtime_estimator': {}},
  }

def validate_oci_bucket(bucket_name: str, filepath: str):
   object_name = filepath.split(f'{bucket_name}/')[-1]
   config = oci.config.from_file()
   object_storage = oci.object_storage.ObjectStorageClient(config)
   namespace = object_storage.get_namespace().data
   try:
      response = object_storage.get_object(namespace, bucket_name, object_name)
      if response.status != 200:
        print(f'Error reading OCI bucket {bucket_name} with object_name: {object_name}. Status was: {response.status}')
        sys.exit(1)
      return
   except Exception as e:
      print(f'Error reading OCI bucket {bucket_name} with object_name: {object_name}. Error was: {e}')
      sys.exit(1)
