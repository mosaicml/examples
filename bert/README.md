# Welcome!

This benchmark covers pre-training and fine-tuning a BERT model from scratch. With this starter code, you'll be able to do Masked Language Modeling (MLM) pre-training on the C4 dataset and classification fine-tuning on GLUE benchmark tasks. We also provide the source code and recipe behind our Mosaic BERT model, which you can train yourself using this repo.

## Contents

You'll find in this folder:

* This `README.md`
* `main.py` — A straightforward script for parsing YAMLs, building a [Composer](https://github.com/mosaicml/composer) Trainer, and kicking off an MLM pre-training training job, locally or on Mosaic's cloud
* `glue.py` - A more complex script for parsing YAMLs and orchestrating the numerous fine-tuning training jobs across 8 GLUE tasks (we exclude the WNLI task here), locally or on Mosaic's cloud
* `convert_c4.py` — Code for creating a streaming C4 dataset, which can be used for pre-training. See [Dataset preparation](#Dataset-preparation)
* `yamls/` - Pre-baked configs for training both our sped-up `Mosaic BERT` as well as the reference `HuggingFace BERT`
* `requirements.txt` — All needed Python dependencies
* `src/data_c4.py` — A [MosaicML streaming dataset](https://docs.mosaicml.com/projects/streaming/en/latest/) that can be used with a vanilla PyTorch dataloader or [Composer](https://github.com/mosaicml/composer)
* `src/hf_bert.py` — HuggingFace BERT models for MLM (pre-training) or classification (GLUE fine-tuning), wrapped in [`ComposerModel`](https://docs.mosaicml.com/en/v0.11.1/api_reference/generated/composer.models.HuggingFaceModel.html) for compatibility with the [Composer Trainer](https://docs.mosaicml.com/en/v0.11.1/api_reference/generated/composer.Trainer.html#composer.Trainer)
* `src/mosaic_bert.py` — Mosaic BERTs model for MLM (pre-training) or classification (GLUE fine-tuning), wrapped in [`ComposerModel`](https://docs.mosaicml.com/en/v0.11.1/api_reference/generated/composer.models.HuggingFaceModel.html) for compatibility with the [Composer Trainer](https://docs.mosaicml.com/en/v0.11.1/api_reference/generated/composer.Trainer.html#composer.Trainer)
* `src/bert_layers.py` — The Mosaic BERT layers/modules with our custom speed up methods built in, with an eye towards HuggingFace API compatibility
* `src/bert_padding.py` — Source code for Mosaic BERT that supports reshaping tensors in order avoid inefficiencies due to padding
* `src/flash_attn_triton.py` - Source code for the [FlashAttention](https://arxiv.org/abs/2205.14135) implementation used in Mosaic BERT
* `src/glue/data.py` - Datasets used by `glue.py` in GLUE fine-tuning
* `src/glue/finetuning_jobs.py` - Custom classes, one for each GLUE task, instantiated by `glue.py`, which handle individual fine-tuning jobs and encode task-specific hyperparameters and datasets

# Mosaic BERT

Our starter code provides support for standard HuggingFace BERT models, as well as our own **Mosaic BERT**, which incorporates numerous methods to improve throughput and training.
Our goal in developing Mosaic BERT was to apply a combination of methods from the literature to seriously speed up training time, and to package it in a way that's easy for you to use on your own problems!

We apply:
* [ALiBi (Press et al, 2021)](https://arxiv.org/abs/2108.12409v1)
* [Gated Linear Units (Shazeer, 2020)](https://arxiv.org/abs/2002.05202)
* ["The Unpadding Trick"](https://github.com/mlcommons/training_results_v1.1/blob/main/NVIDIA/benchmarks/bert/implementations/pytorch/fmha.py)
* [FusedLayerNorm (NVIDIA)](https://nvidia.github.io/apex/layernorm.html)
* [FlashAttention (Tri Dao, 2022)](https://arxiv.org/abs/2205.14135)

... and get them to work together! To our knowledge, many of these methods have never been combined before.

If you're reading this, we're still profiling the exact speedup and performance gains offered by Mosaic BERT compared to comparable HuggingFace BERT models. Stay tuned for incoming results!

## Motivation

The average reader of this README likely does not train BERT-style models from scratch, but instead starts with pre-trained weights, likely using HuggingFace code like `model = AutoModel.from_pretrained('bert-base-uncased')`. Such a reader may wonder: why would you train from scratch? We provide a detailed rationale below, but the strongest case for training your own Mosaic BERT is simple: **better accuracy, with a faster model, at a low cost.**

There is mounting evidence that pre-training on domain specific data improves downstream evaluation accuracy:

* [Downstream Datasets Make Surprisingly Good Pretraining Corpora](https://arxiv.org/abs/2209.14389)
* [Quality Not Quantity: On the Interaction between Dataset Design and Robustness of CLIP](https://arxiv.org/abs/2208.05516)
* [Insights into Pre-training via Simpler Synthetic Tasks](https://arxiv.org/abs/2206.10139)
* [Pre-train or Annotate? Domain Adaptation with a Constrained Budget](https://arxiv.org/abs/2109.04711)
* [Foundation Models of Scientific Knowledge for Chemistry: Opportunities, Challenges and Lessons Learned](https://openreview.net/forum?id=SLX-I2MHUZ9)
* [Domain-Specific Language Model Pretraining for Biomedical Natural Language Processing](https://arxiv.org/abs/2007.15779)
* [LinkBERT: Pretraining Language Models with Document Links](https://arxiv.org/abs/2203.15827)

In addition to being able to take advantage of pre-training on in-domain data, training from scratch means that you control the data start to finish. Publicly available pre-training corpuses cannot be used in many commercial cases due to legal considerations. Our starter code can easily be modified to handle custom datasets beyond the C4 example we provide.

One may wonder, why start from scratch when public data isn't a concern? Granted that it is better to train on domain-specific data, can't that happen as "domain adaptation" from a pre-trained checkpoint? There are two reasons not to do this, one theoretical and one practical. The theory says that, because we are doing non-convex optimization, domain adaptation "may not be able to completely undo suboptimal initialization from the general-domain language model" [Gu et al., 2020](https://arxiv.org/abs/2007.15779).

The practical reason is that certain outcomes are only available if the model and tokenizer are pre-trained from scratch.

So, for example, if you want to use ALiBi positional embeddings (and [you probably should](https://ofir.io/The-Use-Case-for-Relative-Position-Embeddings/), they seem to improve LM perplexity, downstream accuracy, and allow the model to generalize to longer sequences than seen at train time), you need to train from scratch (or fine-tune from a checkpoint which was pre-trained with ALiBi positional embeddings, which we will be releasing!). Or if you want to use Gated Linear Units in your feedforward layers ([as recommended by Noam Shazeer](https://arxiv.org/abs/2002.05202), one of the authors of the original Transformers paper), again, you have to train with them from scratch.

Another good example is domain-specific tokenization. In the biomedical domain, words may be split by the pre-trained BERT tokenizer in ways that make downstream tasks more difficult and computationally expensive. For example, the common drug "naloxone" in tokenized by `bert-base-uncased` tokenizer into the 4 tokens `([na, ##lo, ##xon, ##e]` [Gu et al., 2020](https://arxiv.org/abs/2007.15779), making tasks like NER more difficult and using more of the limited sequence length available.

Now that we've convinced you that you should train a Mosaic BERT from scratch, let's get into the how :) 

# Setup

Here's what you need to get started with our BERT implementation:

* Use a Docker image with PyTorch 1.12+, e.g. [MosaicML's PyTorch base image](https://hub.docker.com/r/mosaicml/pytorch/tags)
  * Recommended tag: `mosaicml/pytorch:1.12.1_cu116-python3.9-ubuntu20.04`
  * This image comes pre-configured with the following dependencies:
    * PyTorch Version: 1.12.1
    * CUDA Version: 11.6
    * Python Version: 3.9
    * Ubuntu Version: 20.04

* Use a system with NVIDIA GPU(s)

* Install requirements via: `pip install -r requirements.txt`

* Prepare a local copy of the C4 dataset via instructions below.

## Dataset preparation

To run pre-training, you'll need to make yourself a local copy of the C4 pre-training dataset.
If you only want to profile these BERTs, we recommend that you **only download and prepare the `val` split**,
and use it for both train and eval in your script. Just change `split: train` to `split: val` in your run YAML.
Alternatively, feel free to substitute our dataloader with one of your own in the script [main.py](./main.py#L101)!

In this benchmark, we train BERTs on the [C4: Colossal, Cleaned, Common Crawl dataset](https://huggingface.co/datasets/c4).
We first convert the dataset from its native format (a collection of zipped JSONs)
to MosaicML's streaming dataset format (a collection of binary `.mds` files).
Once in `.mds` format, we can store the dataset in a central location (filesystem, S3, GCS, etc.)
and stream the data to any compute cluster, with any number of devices, and any number of CPU workers, and it all ~ just works ~ .
You can read more about the benefits of using mosaicml-streaming [here](https://docs.mosaicml.com/projects/streaming/en/latest/).

### Converting C4 to streaming dataset `.mds` format

To make yourself a copy of C4, use `convert_c4.py` like so:

```bash
# Download the 'val' split and convert to StreamingDataset format
# This will take 10 sec to 1 min depending on your Internet bandwidth
# You should see a dataset folder `./my-copy-c4/val` that is ~0.5GB
python convert_c4.py --out_root ./my-copy-c4 --splits val

# Download the 'train' split if you really want to train the model (not just profile)
# This will take 1-to-many hours depending on bandwidth, # CPUs, etc.
# The final folder `./my-copy-c4/train` will be ~800GB so make sure you have space!
python convert_c4.py --out_root ./my-copy-c4 --splits train
```

### Test the Dataloader

To verify that the dataloader works, run a quick test on your `val` split like so:

```bash
# This will construct a `StreamingC4` dataset from your `val` split,
# pass it into a PyTorch Dataloader, and iterate over it and print samples.
# Since remote and local are set to the same path, no streaming/copying takes place.
python src/data_c4.py ./my-copy-c4 ./my-copy-c4

# This will do the same thing, but stream data from {remote} -> {local}.
# The remote path can be a filesystem or object store URI.
python src/data_c4.py ./my-copy-c4 /tmp/cache-c4
python src/data_c4.py s3://my-bucket/my-copy-c4 /tmp/cache-c4
```

# Training

Now that you've installed dependencies and built a local copy of the C4 dataset, let's start training! We'll start with MLM pre-training on C4.

**Please remember** to edit the `data_remote` and `data_local` paths in your YAML to point to your local C4 copy.
Our streaming dataloader always streams from `data_remote` -> `data_local`, and if both paths are the same,
then no extra copying is done.

**Also remember** that if you only downloaded the `val` split, you need to make sure your train_dataloader is pointed at that split.
Just change `split: train` to `split: val` in your YAML.
This is already done in the testing YAML `yamls/test-cpu/main.py`, which you can also use to test your configuration.

## MLM pre-training

We run the `main.py` script using our `composer` launcher, which generates N processes (1 process per GPU device).

If training on a single node, the `composer` launcher will autodetect the number of devices, so all you need to do to pre-train a Mosaic BERT is run:

```bash
composer main.py yamls/mosaic-bert-base-uncased.yaml
```

**Note:** The `yamls/*.yaml` files are intended to be used with `main.py`.

**Please remember** to modify the reference YAMLs (e.g., `yamls/mosiac-bert-base-uncased.yaml`) to customize saving and loading locations -- only the YAMLs in `yamls/test-cpu/` are ready to use out-of-the-box.

## GLUE fine-tuning

The GLUE benchmark measures the average performance across 8 NLP classification tasks (again, here we exclude the WNLI task). This performance is typically used to evaluate the quality of the pre-training: once you have a set of weights from your MLM task, you fine-tune those weights separately for each task and then compute the average performance across the tasks, with higher averages indicating higher pre-training quality.

To handle this complicated fine-tuning pipeline, we provide the `glue.py` script. which handles parallelizing each of these fine-tuning jobs across all the GPUs on your node.
To clarify, the `glue.py` script takes advantage of the small dataset/batch size of the GLUE tasks through *task* parallelism (where different tasks are trained in parallel, each using its own GPU), rather than the typical data parallelism (where one task is trained at a time and training batches are parallelized across GPUs).

Once you have modified the YAMLs in `yamls/glue/` to reference your pre-trained checkpoint as the GLUE starting point, use non-default hyperparameters, etc., run the `glue.py` script using the standard `python` launcher (we don't use the `composer` launcher here because `glue.py` does its own multi-process orchestration):

```bash
python glue.py yamls/glue/mosaic-bert-base-uncased.yaml
```

Aggregate GLUE scores will be printed out at the end of the script and can also be tracked using Weights and Biases, if enabled via the YAML.
Any of the other (composer supported loggers)[https://docs.mosaicml.com/en/latest/trainer/logging.html#available-loggers] can be added easily as well!

**Note:** The `yamls/glue/*.yaml` files are intended to be used with `glue.py`.

**Please remember** to modify the reference YAMLs (e.g., `yamls/glue/mosiac-bert-base-uncased.yaml`) to customize saving and loading locations -- only the YAMLs in `yamls/test-cpu/` are ready to use out-of-the-box.

### Running on the MosaicML Cloud

If you have configured a compute cluster to work with the MosaicML Cloud, you can use the `mcloud_run.yaml` reference YAMLs for examples of how to run pre-training and fine-tuning remotely!

Once you have filled in the missing YAML fields (and made any other modifications you want), you can launch pre-training by simply running:

```bash
mcli run -f yamls/mcoud_run.yaml
```

Similarly, for GLUE fine-tuning just fill in the missing YAML fields (e.g., to use the pre-training checkpoint as the starting point) and run:

```bash
mcli run -f yamls/glue/mcoud_run.yaml
```

## Multi-node training

To train with high performance on *multi-node* clusters, the easiest way is with MosaicML Cloud ;)

But if you want to try this manually on your own cluster, then just provide a few variables to `composer`, either directly via CLI or via environment variables. Then launch the appropriate command on each node.

**Note:** multi-node training will only work `main.py` (the `glue.py` script handles its own orchestration across devices and is not built to be used with the `composer` launcher).

### Multi-Node via CLI args

```bash
# Using 2 nodes with 8 devices each
# Total world size is 16
# IP Address for Node 0 = [0.0.0.0]

# Node 0
composer --world_size 16 --node_rank 0 --master_addr 0.0.0.0 --master_port 7501 main.py yamls/mosaic-bert-base-uncased.yaml

# Node 1
composer --world_size 16 --node_rank 1 --master_addr 0.0.0.0 --master_port 7501 main.py yamls/mosaic-bert-base-uncased.yaml

```

### Multi-Node via environment variables

```bash
# Using 2 nodes with 8 devices each
# Total world size is 16
# IP Address for Node 0 = [0.0.0.0]

# Node 0
# export WORLD_SIZE=16
# export NODE_RANK=0
# export MASTER_ADDR=0.0.0.0
# export MASTER_PORT=7501
composer main.py yamls/mosaic-bert-base-uncased.yaml

# Node 1
# export WORLD_SIZE=16
# export NODE_RANK=1
# export MASTER_ADDR=0.0.0.0
# export MASTER_PORT=7501
composer main.py yamls/mosaic-bert-base-uncased.yaml
```

You should see logs being printed to your terminal.
You can also easily enable other experiment trackers like Weights and Biases or CometML by using [Composer's logging integrations](https://docs.mosaicml.com/en/v0.11.1/trainer/logging.html).

# Contact Us

If you run into any problems with the code, please file Github issues directly to this repo.

If you want train BERT-style models on MosaicML Cloud, reach out to us at [demo@mosaicml.com](mailto:demo@mosaicml.com)!
