## Overview

In this tutorial, we will be creating an application that answers questions about the MosaicML composer codebase. The basic structure of this application will be a retrieval question answering system where the user will provide the chatbot with a question, and then a language model will answer the question based on the retrieved text. See some [great](https://python.langchain.com/en/latest/modules/chains/index_examples/vector_db_qa.html#retrieval-question-answering) [materials](https://blog.langchain.dev/langchain-chat/) from [LangChain](https://python.langchain.com/en/latest/index.html) for more exploration on this type of application.

By default the model that is used throughout is [MPT-7b](https://huggingface.co/mosaicml/mpt-7b), a 7-billion parameter large language model trained by MosaicML. See [our blog](https://www.mosaicml.com/blog/mpt-7b) for more details. We have also released a more powerful, 30-billion parameter model, which you can swap in to this example very easily. See [MPT-30b](https://huggingface.co/mosaicml/mpt-30b) for more details. To swap in the 30b model, all you need to do is change all occurrences of `mpt-7b` to `mpt-30b`. The important ones to change are the `model.pretrained_model_name_or_path` and the `tokenizer.name`, in the various [finetune](./mcli-yamls/finetune/) yamls. The other occurrences are just in names of runs and save folders. Depending on your hardware, and particularly if you get a CUDA `c10` error, you may also need to change `device_train_microbatch_size` from `auto` to `1` in the [finetune](./mcli-yamls/finetune/) yamls.

![demo](web_app_screenshot.png)


## Which MosaicML tools will we use?
- [LLM-foundry](https://github.com/mosaicml/llm-foundry): An open-source PyTorch library of tools for training, finetuning, evaluating, and deploying LLMs for inference.
- [Composer](https://github.com/mosaicml/composer): An open-source PyTorch library for easy large scale deep learning.
- [Streaming](https://github.com/mosaicml/streaming): An open-source PyTorch library for efficiently and accurately streaming data from the cloud to your training job, whereever that job is running.
- [MCLI](https://docs.mosaicml.com/projects/mcli/en/latest/): The command line interface for running training and inference jobs on the MosaicML platform.


## Setup

Jobs can be submitted to the MosaicML platform either using the [SDK](https://docs.mosaicml.com/projects/mcli/en/latest/training/working_with_runs.html#manage-a-run-with-the-sdk), or MCLI yamls. All commands in this tutorial are going to be run using MCLI yamls. For more MosaicML platform documentation, see the [MosaicML documentation](https://docs.mosaicml.com/projects/mcli/en/latest/), and for a detailed explanation of our yamls, see [training yaml documentation](https://docs.mosaicml.com/projects/mcli/en/latest/training/yaml_schema.html) and [inference yaml documentation](https://docs.mosaicml.com/projects/mcli/en/latest/inference/inference_schema.html). For understanding how this works, and what is actually running, there are a few important details to understand.

1) The MosaicML platform does not have permanent storage on the compute nodes. This means that all data will be streamed in/out to/from a cloud object store. In this tutorial we will use [MosaicML Streaming](https://github.com/mosaicml/streaming) to accomplish this. See [MosaicML setup](#mosaicml-platform-setup) for more details on setting up your cloud provider of choice.
1) The `command` section of the yaml is what will actually get run on the compute node. If you are trying to debug/run something locally, you should run what appears in the `command` section (after setting up your local environment).
1) The `parameters` section of the yaml is mounted as a single `.yaml` file at `/mnt/config/parameters.yaml`, which your `command` can then read from. This `parameters` section is how we will pass the training configuration parameters to the training script.


### MosaicML platform setup

Before starting this tutorial, you should make sure that you have access to the MosaicML platform. You'll need access to both training and inference services to complete this tutorial, although you can follow this tutorial up to the [deployment](#6-deploy-your-model-and-an-embedding-model) section if you just have access to training. Please [reach out](https://forms.mosaicml.com/demo?utm_source=inference&utm_medium=mosaicml.com&utm_campaign=always-on) if you would like to sign up, and reach out if you are already a customer and need to gain access to either service.

1. Go through the [getting started guide](https://docs.mosaicml.com/projects/mcli/en/latest/quick_start/getting_started.html).
1. Set up the object store of your choice, by following the [secrets guide](https://docs.mosaicml.com/projects/mcli/en/latest/resources/secrets/index.html) for your cloud provider of choice.
1. [Optional] Set up [Weights & Biases](https://docs.mosaicml.com/projects/mcli/en/latest/resources/integrations/wandb.html) or [CometML](https://docs.mosaicml.com/projects/mcli/en/latest/resources/integrations/comet.html) for tracking your experiments.

Once you have done all of this, you should be ready to get started with this tutorial!


### Local setup

All that is needed for local setup is to clone this repository and install a few dependencies, as the only thing you will be running locally is the final application. Everything else will be run through the MosaicML platform.
```bash
git clone https://github.com/mosaicml/examples
cd cd examples/examples/end-to-end-examples/sec_10k_qa
python -m venv examples-10k-venv
source examples-10k-venv/bin/activate
pip install -r requirements-cpu.txt
# Your api token can be found by running `mcli config`. This token is set an environment variable for the langchain integration
export MOSAICML_API_TOKEN=<your api token>
```


## How to follow this tutorial

Each section of this tutorial will have a command to run, which fields you need to edit before running the command, the expected input and output of that section, and a description of what is happening. The steps should be run sequentially, and you must wait for each prior step to complete before running the next one. The commands will be run using MCLI yamls (except the final front end which will run locally). All intermediate output will be written to cloud object store. Everywhere that a path is used, it will be of the form `CLOUD://BUCKET_NAME/path/to/my/folder/`. You will need to fill in the `CLOUD` (e.g. `s3`, `oci`, `gs`) and the `BUCKET_NAME` (e.g. `my-bucket`). The description of what is happening will be a high level overview of the steps that are being taken, and will not go into detail about the code, but the code and yamls will have detailed comments, and the description will contain pointers to where to learn more. We encourage you to read the yamls in detail to gain a better understanding of the various options that are available.
