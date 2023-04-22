# LLM Finetuning

This directory contains utilities for Seq2Seq finetuning LLMs, for example, Supervised Finetuning (SFT) (aka Instruction(Fine)Tuning (IFT)), or finetuning a base LLM to focus on a specific task like summarization.

## Usage

### Using an MDS-formatted dataset (locally or in an object store)

1. Set the `data_remote` or `data_local` value in your YAML

### Using a dataset on the HuggingFace Hub

1. In `_task.py`, write a function for processing the dataset, to split it into prompt and response
1. Register this function using `@dataset_constructor.register('hf-hub/identifier')`
1. Reference this in a training yaml, such as the one in `yamls/mosaic_gpt/finetune/7b_dolly_sft.yaml`

### Using a local dataset

1. In `_task.py`, write a function for processing the dataset, to split it into prompt and response
1. Register this function using `@dataset_constructor.register('some_name')`
1. Reference this in a training yaml, such as the one in `yamls/mosaic_gpt/finetune/1b_local_data_sft.yaml`
