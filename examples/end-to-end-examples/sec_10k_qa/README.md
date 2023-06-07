### Overview

In this tutorial, we will be creating an application that answers questions based on [SEC Form 10-K](https://en.wikipedia.org/wiki/Form_10-K) documents. These are annual financial performance summaries that all companies are required to file with the SEC annually. Analyzing these filings and extracting information from them is an important part staying on top of financial data about public companies. The basic structure of the application will be a retrieval question answering system. The user will provide what company and year they want to ask questions about, and the question they would like to ask. Based on the question, a text embedding model will retrieve relevant sections of the 10-K document, and then a language model will answer the question based on the retrieved text. See some [great](https://python.langchain.com/en/latest/modules/chains/index_examples/vector_db_qa.html#retrieval-question-answering) [materials](https://blog.langchain.dev/langchain-chat/) from [LangChain](https://python.langchain.com/en/latest/index.html) for more exploration on this type of application.

The goal of this tutorial is to show how we can build a prototype of this application using the MosaicML platform. We will cover:
- Setting up your development environment and the MosaicML platform
- Downloading and processing the data for domain specific pretraining of [MPT-7b](https://huggingface.co/mosaicml/mpt-7b)
- Using the MosaicML inference integration in LangChain to build the text retrieval component
- Deploying your finetuned MPT-7b for inference using the MosaicML inference service
- Building a simple frontend to tie everything together

At the end of this tutorial, you will have a simple web application that answers questions about SEC Form 10-K documents using a model that you finetuned and deployed using MosaicML! All the normal caveats about language models and making things up apply, and you there are a variety of avenues you can pursue after going through this tutorial to improve the quality :)

### Setup

All commands in this tutorial are going to be run using MCLI yamls. For understanding how this works, and what is actually running, there are a few important details to understand. 1) The MosaicML platform does not have permanent storage on the compute nodes. This means that all data will be streamed in and out from a cloud provider. See [MosaicML setup](#mosaicml-platform-setup) for more details on setting up your cloud provider. 2) The `command` section of the yaml is what will actually get run on the compute node. If you are trying to debug/run something locally, you should run what appears in the `command` section. 3) The `parameters` section of the yaml is mounted as a single `.yaml` file at `/mnt/config/parameters.yaml`, which your `command` section can then read from. For more MosaicML platform documentation, see the [MosaicML documentation](https://docs.mosaicml.com/projects/mcli/en/latest/), and for a detailed explanation of our training yaml, see [https://docs.mosaicml.com/projects/mcli/en/latest/training/yaml_schema.html](https://docs.mosaicml.com/projects/mcli/en/latest/training/yaml_schema.html).

## MosaicML platform setup

Before starting this tutorial, you should make sure that you have access to the MosaicML platform. You'll need access to both training and inference services. Please reach out if you need to gain access. First, you should go through the [getting started guide](https://docs.mosaicml.com/projects/mcli/en/latest/quick_start/getting_started.html). Second, you should set up the object store of your choice, by following the [secrets guide](https://docs.mosaicml.com/projects/mcli/en/latest/resources/secrets/index.html) for your cloud provider of choice. Lastly, you may want to set up [Weights & Biases](https://docs.mosaicml.com/projects/mcli/en/latest/resources/integrations/wandb.html) or [CometML](https://docs.mosaicml.com/projects/mcli/en/latest/resources/integrations/comet.html) for tracking your experiments. Once you have done all of this, you should be ready to get started with this tutorial!

### Acquiring the SEC 10-K data

**Local command:** `python 01_process_and_upload_10ks.py --folder_for_upload oci://mosaicml-internal-checkpoints/daniel/sec-filings-large/ --dataset_subset large_full`

We will use the version of the 10-K data kindly uploaded to HuggingFace (https://huggingface.co/datasets/JanosAudran/financial-reports-sec) by `JanosAudran`. Note that reprocessing the data may improve the quality, as this version of the data appears to largely be missing tables, which are an important part of financial statements. Each row in this dataset corresponds to a sentence, so we will need to reprocess the data into full text documents before we can use it. Throughout this tutorial we will use the `large_full` subset of the data. If you would like to simply go through all of the steps quickly and make sure that they run, you can instead use the `small_full` subset (throughout the tutorial), which contains a small subset of the full data.

### MDS Conversion

**Local command:** `python 02_convert_10ks_to_mds.py --tokenizer EleutherAI/gpt-neox-20b --eos_text '<|endoftext|>' --concat_tokens 2048 --out_root oci://mosaicml-internal-checkpoints/daniel/data/sec-large-mds/ --in_root oci://mosaicml-internal-checkpoints/daniel/data/sec-filings-large/ --dataset_subset large_full`

As mentioned in the [MosaicML platform setup](#mosaicml-platform-setup) section, the MosaicML platform does not have permanent storage on the compute nodes. This means that all data will be streamed in and out from a cloud provider. In order to make this process as efficient as possible, we will convert the data into a format that is optimized for streaming, using our [Streaming](https://github.com/mosaicml/streaming) library. This format is called MDS, and is a simple format that is optimized for streaming. The `02_convert_10ks_to_mds.py` script will convert the data into this format. The `--tokenizer` argument specifies the tokenizer to use, and the `--eos_text` argument specifies the text to use as the end of sequence token. The `--concat_tokens` argument specifies the maximum number of tokens to concatenate into a single text example. The `--out_root` argument specifies the output root for the MDS files, and the `--in_root` argument specifies the input root for the 10-K data. The `--dataset_subset` argument specifies which subset of the data to use. The `--dataset_subset` argument is optional, and defaults to `small_full`.

### Finance Finetune MPT-7b

**Input**: `oci://mosaicml-internal-checkpoints/daniel/data/sec-large-mds/`

**MCLI command**: `mcli run -f 03_finetune_on_10ks.yaml`

Next, we will finetune our base model on the train split of the 10-K data in order to tune it on data that is in-domain for the end task of answering questions about 10-K forms. This process is called "domain tuning," and can be useful for adapting a model that has already been trained on a huge amount of data (e.g. MPT-7b) to a new domain. For the purposes of this example, we will use the train/validation/test split provided with the dataset, which splits by company. So each company (which has multiple years of 10-K forms) will only appear in one of the splits. We will use the validation split as validation data, and reserve the test split for our final testing of our application.

Please check out the [training yaml](./yamls/mcli/03_finetune_on_10ks.yaml) for all of the details. This yaml will load the pretrained weights for `mpt-7b` available on the [HuggingFace Hub](https://huggingface.co/mosaicml/mpt-7b), and then train using the normal causal language modeling objective on the 10-K form dataset that we processed in the previous step. TODO: abstract

TODO: explain HF checkpoint versus composer checkpoint

**Output:**: `daniel/checkpoints/sec-finetune-neo-125-2-CwFq6r/ep1-ba367-rank0.pt`

### Instruct Finetune MPT-7b

**Input:**: `daniel/checkpoints/sec-finetune-neo-125-2-CwFq6r/ep1-ba367-rank0.pt`

**MCLI command:** `mcli run -f yamls/mcli/04_instruction_finetune_on_dolly_hh.yaml`

Now that we have trained our model on in-domain financial text, we will train it to better be able to follow instructions. For this step, we will follow the exact same process that we used to create [`mpt-7b-instruct`](https://huggingface.co/mosaicml/mpt-7b-instruct). Namely, we will finetune the model on instruction formatted data following the form of the [`databricks-dolly-15k` dataset](https://huggingface.co/datasets/databricks/databricks-dolly-15k), shown below. See [`mosaicml/dolly_hhrlhf`](https://huggingface.co/datasets/mosaicml/dolly_hhrlhf) for more details on the specific dataset we are using.

```python
INSTRUCTION_KEY = "### Instruction:"
RESPONSE_KEY = "### Response:"
INTRO_BLURB = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
PROMPT_FOR_GENERATION_FORMAT = """{intro}
{instruction_key}
{instruction}
{response_key}
""".format(
    intro=INTRO_BLURB,
    instruction_key=INSTRUCTION_KEY,
    instruction="{instruction}",
    response_key=RESPONSE_KEY,
)

example = "James decides to run 3 sprints 3 times a week. He runs 60 meters each sprint. How many total meters does he run a week? Explain before answering."
fmt_ex = PROMPT_FOR_GENERATION_FORMAT.format(instruction=example)
```

TODO: reexplain HF checkpoint versus composer checkpoint

### Convert the Composer checkpoint to a HuggingFace checkpoint

**Input:** `oci://mosaicml-internal-checkpoints/daniel/checkpoints/sec-finetune-dolly-mpt-1-WhKWb3/ep1-ba367-rank0.pt`

**MCLI command:** `python 05_convert_composer_to_hf.py --composer_path oci://mosaicml-internal-checkpoints/daniel/checkpoints/sec-finetune-neo-125-3-8UEv7I/ep1-ba367-rank0.pt --hf_output_path oci://mosaicml-internal-checkpoints/daniel/checkpoints/sec-finetune-neo-125-3-8UEv7I/ep1-ba367-rank0/`
The last step before we can deploy our newly-trained large language model is to convert it to a format that can be used by the HuggingFace library. This is a simple process that can be done using the `05_convert_composer_to_hf.py` script. The `--composer_path` argument specifies the path to the Composer checkpoint, and the `--hf_output_path` argument specifies the path to the output HuggingFace checkpoint.

**Output:** `oci://mosaicml-internal-checkpoints/daniel/checkpoints/sec-finetune-neo-125-3-8UEv7I/ep1-ba367-rank0/`

### Deploy your model and an embedding model

**MCLI command**: `mcli deploy -f 06_deploy_llm.yaml --cluster r7z13`

**MCLI command**: `mcli deploy -f 07_deploy_embedding_model.yaml --cluster r7z13`

Now that we have our trained model, we will deploy it using MosaicML inference. This will allow use to use the model as an API. We will additionally deploy a text embedding model to perform retrieval of relevant text sections from the 10-K form as context for the language model to answer questions.

Make sure to edit the `06_deploy_llm.yaml` file to point to _your_ checkpoint path. `07_deploy_embedding_model.yaml` does not need to be edited because it is downloading a model from the HuggingFace Hub.

### Application with gradio
play around
prompt dataset to eval

### What next?
your data
more data
better instruct data
finetuning hyperparams
more prompt engineering
domain specific retrieval
read more foundry
read more composer
read more streaming
adapt this piece by piece to your task
