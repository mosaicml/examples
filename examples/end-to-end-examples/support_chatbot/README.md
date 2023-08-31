## Overview

In this tutorial, we will be creating an application that answers questions about the MosaicML composer codebase. The basic structure of this application will be a retrieval question answering system where the user will provide the chatbot with a question, and then a language model will answer the question based on the retrieved text. See some [great](https://python.langchain.com/en/latest/modules/chains/index_examples/vector_db_qa.html#retrieval-question-answering) [materials](https://blog.langchain.dev/langchain-chat/) from [LangChain](https://python.langchain.com/en/latest/index.html) for more exploration on this type of application.

By default the model that is used throughout is [MPT-30b](https://huggingface.co/mosaicml/mpt-30b), a 30-billion parameter large language model trained by MosaicML. See [our blog](https://www.mosaicml.com/blog/mpt-30b) for more details. Depending on your hardware, and particularly if you get a CUDA `c10` error, you may need to change `device_train_microbatch_size` from `auto` to `1` in the [finetune](./mcli-yamls/finetune/) yamls.

![demo](web_app_screenshot.png)


## Which MosaicML tools will we use?
- [LLM-foundry](https://github.com/mosaicml/llm-foundry): An open-source PyTorch library of tools for training, finetuning, evaluating, and deploying LLMs for inference.
- [Composer](https://github.com/mosaicml/composer): An open-source PyTorch library for easy large scale deep learning.
- [Streaming](https://github.com/mosaicml/streaming): An open-source PyTorch library for efficiently and accurately streaming data from the cloud to your training job, whereever that job is running.
- [MCLI](https://docs.mosaicml.com/projects/mcli/en/latest/): The command line interface for running training and inference jobs on the MosaicML platform.


## Setup

Jobs can be submitted to the MosaicML platform either using the [SDK](https://docs.mosaicml.com/projects/mcli/en/latest/training/working_with_runs.html#manage-a-run-with-the-sdk), or MCLI yamls. All commands in this tutorial are going to be run using MCLI yamls. For more MosaicML platform documentation, see the [MosaicML documentation](https://docs.mosaicml.com/projects/mcli/en/latest/), and for a detailed explanation of our yamls, see [training yaml documentation](https://docs.mosaicml.com/projects/mcli/en/latest/training/yaml_schema.html) and [inference yaml documentation](https://docs.mosaicml.com/projects/mcli/en/latest/inference/inference_schema.html). For understanding how this works, and what is actually running, there are a few important details to understand.

1) The MosaicML platform does not have permanent storage on the compute nodes. This means that all data will be streamed in/out to/from a cloud object store (please note that our deployment only works for s3 ,gs, and HuggingFace). In this tutorial we will use [MosaicML Streaming](https://github.com/mosaicml/streaming) to accomplish this. See [MosaicML setup](#mosaicml-platform-setup) for more details on setting up your cloud provider of choice.
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
cd examples/examples/end-to-end-examples/support_chatbot
python -m venv examples-chatbot-venv
source examples-chatbot-venv/bin/activate
pip install -r requirements-cpu.txt
# Your api token can be found by running `mcli config`. This token is set as an environment variable for the langchain integration
export MOSAICML_API_TOKEN=<your api token>
```


## How to follow this tutorial

Each section of this tutorial will have a command to run, which fields you need to edit before running the command, the expected input and output of that section, and a description of what is happening. The steps should be run sequentially, and you must wait for each prior step to complete before running the next one. The commands will be run using MCLI yamls (except the final front end which will run locally). All intermediate output will be written to cloud object store. Everywhere that a path is used, it will be of the form `CLOUD://BUCKET_NAME/path/to/my/folder/`. You will need to fill in the `CLOUD` (e.g. `s3`, `oci`, `gs`) and the `BUCKET_NAME` (e.g. `my-bucket`). The description of what is happening will be a high level overview of the steps that are being taken, and will not go into detail about the code, but the code and yamls will have detailed comments, and the description will contain pointers to where to learn more. We encourage you to read the yamls in detail to gain a better understanding of the various options that are available.

## Step 1: Getting Our Data

Let's first start with downloading the github repository that we want to finetune our model on so that it can get a basic understanding of the codebase. The [repo_downloader](./scripts/repo_downloader.py) will clone the git repository into a designated cloning directory where it will flatten the repository to be an assortment of text files in a local directory: `retrieval_data/{REPOSITORY_NAME}`. It will then erase the cloning directory.

```bash
python scripts/repo_downloader.py REPO_LINKS
```

**Fields to replace with your values:** `REPO_LINKS` (in the command line). For instance, to download the MosaicML Composer repository, we run:
```bash
python scripts/repo_downloader.py https://github.com/mosaicml/composer
```

## Step 2: Converting to MDS

As mentioned in the [MosaicML platform setup](#mosaicml-platform-setup) section, the MosaicML platform does not have permanent storage on the compute nodes. This means that all data will be streamed in and out from a cloud provider. In order to make this process as efficient as possible during a training run, we will convert the data into a format that is optimized for streaming, using our [Streaming](https://github.com/mosaicml/streaming) library. This format is called [MDS](https://docs.mosaicml.com/projects/streaming/en/stable/fundamentals/dataset_format.html#mds), and is a simple format that is optimized for streaming.

This example will contain a script for a [text to MDS](./scripts/conversion/convert_txt_to_stream.py) conversion. For text to MDS conversion, we will run:

```bash
mcli run -f mcli_yamls/conversion/convert_txt_to_stream.yaml --cluster CLUSTER
```

**The conversion YAMLs will not work on local directories if the data is not git pushed onto the repository linked by the YAML.** This means that it will not be able to recognize and oci path to data, but only a local path in the github repository (you can feed it the relative path from *step 1*). Thus, if you choose to use the YAML method, make sure that you push the data downloaded from *step* one to your repository. That being said, github repositories are typically very small and you can probably just run this locally:

```bash
python scripts/conversion/convert_txt_to_stream.py \
    --out_root CLOUD://BUCKET/support-bot-demo/data/DATA_NAME_MDS/ \
    --in_root PATH_TO_TXT_FOLDER
```

**Fields to replace with your values:** `CLUSTER` (in the command line), `CLOUD` (in the yaml), `BUCKET` (in the yaml), 'DATA_NAME_MDS' (in the yaml), `PATH_TO_TXT_FOLDER` (in the yaml). Please note that PATH_TO_TXT_FOLDER **MUST** be a local directory (not an OCI link) due to some limitations from OCI. To follow with our previous example in Step 1, if we want to convert the folder containing all of composer as text, we will run with `DATA_NAME_MDS = composer_codebase_mds` and `PATH_TO_TXT_FOLDER = retrieval_data/composer`

## Step 3: Finetuning on our Repository

Next, we will finetune our pretrained model on the train split of our data for the end task of answering questions about the codebase.

Please check out the [training directory](./mcli-yamls/finetune) for all of the details. This yaml will load the pretrained weights for `mpt-30b` available on the [HuggingFace Hub](https://huggingface.co/mosaicml/mpt-30b), and then train using the normal causal language modeling objective on our datasets that we processed in the previous step. The [training script](https://github.com/mosaicml/llm-foundry/blob/main/scripts/train/train.py) itself, is from LLM-foundry.

To run finetuning on the composer codebase, run the following:

```bash
mcli run -f mcli_yamls/finetune/finetune_composer_codebase.yaml --cluster CLUSTER
```
**Fields to replace with your values:** `CLUSTER` (in the command line), `CLOUD` (in the yaml), `BUCKET_NAME` (in the yaml).

**Inputs:** `composer_codebase_mds` cloud folder from step 2

**Outputs:** the checkpoints from your training, saved to the `save_folder` specified in the yaml

## Step 4: Finetuning on Chatv2

Next, we will finetune our model on the train split of the chatv2 dataset to ensure that the bot still has its QA abilities. Chatv2 is the dataset that was used to finetune the MPT-30B model to be MPT-30B-Chat. To run finetuning on Chatv2, run the following:

```bash
mcli run -f mcli_yamls/finetune/finetune_30b_chat.yaml --cluster CLUSTER
```
**Fields to replace with your values:** `CLUSTER` (in the command line), `CLOUD` (in the yaml), `BUCKET_NAME` (in the yaml).

**Outputs:** the checkpoints from your training, saved to the `save_folder` specified in the yaml

## Step 5: Converting Composer Checkpoint to HuggingFace

Before we can deploy our model, we need to convert it into the standard HuggingFace checkpoint folder. We will use the [conversion script](https://github.com/mosaicml/llm-foundry/blob/main/scripts/inference/convert_composer_to_hf.py) from LLM-foundry to do this. This script will take the Composer checkpoint, and write out all the files that HuggingFace expects in a checkpoint folder. You can additionally add the `--hf_repo_for_upload` argument if you would like to upload directly to a private repo on the HuggingFace Hub (you will also need to [set the `HUGGING_FACE_HUB_TOKEN` environment variable](https://docs.mosaicml.com/projects/mcli/en/latest/resources/secrets/env.html) to do this).

**Fields to replace with your values:** `REPLACE_WITH_YOUR_CLUSTER` (in the command), `CLOUD` (in the yaml), `BUCKET_NAME` (in the yaml), `CHECKPOINT_FOLDER_NAME` (in the yaml), `HF_FOLDER_NAME` (in the yaml)

**Inputs:** the final checkpoint from step 4 saved to the `save_folder` specified in the previous yaml and where you want the converted checkpoint to go in `hf_output_path`

**Command:**
```bash
mcli run -f mcli-yamls/convert_checkpoint_to_huggingface.yaml --cluster REPLACE_WITH_YOUR_CLUSTER
```

**Outputs:** the ``CLOUD://BUCKET_NAME/support-bot-demo/converted_checkpoints/HF_FOLDER_NAME`` folder, containing the HuggingFace checkpoint files

## Step 6: Deploy your model

Now that we have our trained model, we will deploy it using MosaicML inference. This will allow us to use the model as an API. For more examples of inference deployments, see [inference-deployments](../../inference-deployments/)


**Fields to replace with your values:** `REPLACE_WITH_YOUR_CLUSTER` (in the command), `BUCKET_NAME` (in the yaml), `HF_FOLDER_NAME` (in the yaml)`

**Inputs:** the HuggingFace format checkpoint from step 5

**Command**:
```bash
mcli deploy -f mcli-yamls/deploy_llm.yaml --cluster REPLACE_WITH_YOUR_CLUSTER
```

**Outputs:** A deployment for the language model


## Step 7: Application with gradio

Now that we've processed our data, trained our models, and deployed our models, we can run the application! We will use Gradio and LangChain to make a simple question answering interface.

We will make use of the MosaicML integration in LangChain for [LLMs](https://github.com/hwchase17/langchain/blob/master/langchain/llms/mosaicml.py) and [embeddings](https://github.com/hwchase17/langchain/blob/master/langchain/embeddings/mosaicml.py), and use the [`RetrievalQA`](https://python.langchain.com/en/latest/modules/chains/index_examples/vector_db_qa.html?highlight=retrievalqa) abstraction with the [`FAISS`](https://python.langchain.com/en/latest/modules/indexes/vectorstores/examples/faiss.html?highlight=faiss) to run the application locally.

Upon the first run of this app, expect that it might take ~20 minutes to embed all of the data and store the vector store in `retrieval_data/vectors.pickle`. After the first run, unless you delete the pickle file, it will just reference the the pickle file and not re-embed.

Play around with the application and imagine ways you could improve it or apply a similar approach to your data!

You can find the names of your deployments by running `mcli get deployments`.

After running the `python` command, you should see link to your application. It is the link after `Running on local URL:`, _not_ the url after `Launching in *reload mode* on:`.

**Command**:
```bash
python app.py
```

What this will do is combine everything together to create your chatbot! With default args, it downloads all of the public MosaicML repositories as well as uses a [web_downloader.py](./web_downloader.py) to download all of the MosaicML Docs. After, it will embed the data to create a vector store that LangChain can run similarity searches on to provide the model with context when answering questions.

Please note that you can customize this app by updating the parser args. Please see what you can customize in [`app_demo.py`](./app_demo.py)


## What next?

Now that you've seen how to use MosaicML to train and deploy a language model, you can try it out on your own data! Here are some ideas for where to go from here:
- Play around with the hyperparameters and prompts for all of the components in [`app.py`](./app.py) and see how they change the output
- Try out different models from the HuggingFace Hub, both for the text embedding and for the LLM
- Try changing optimization parameters in the training yamls to see how they affect the training
- Try swapping in a new dataset, or applying the models to a new task
- Read more about the MosaicML components in this tutorial ([LLM-foundry](https://github.com/mosaicml/llm-foundry), [Composer](https://docs.mosaicml.com/projects/composer/en/latest/), [Streaming](https://docs.mosaicml.com/projects/streaming/en/latest/), [MCLI](https://docs.mosaicml.com/projects/mcli/en/latest/))
