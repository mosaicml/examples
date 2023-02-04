# MosaicML Examples

This repo contains reference examples for training ML models quickly and to high accuracy. It's designed to be easily forked and modified.

It currently features the following examples:

* [ResNet-50 + ImageNet](#resnet-50--imagenet)
* [DeeplabV3 + ADE20k](#deeplabv3--ade20k)
* [GPT / Large Language Models](#large-language-models-llms)
* [BERT](#bert)

# Installation

To get started, either clone or fork this repo and install whichever example\[s\] you're interested in. E.g., to get started training GPT-style language models, just:

```bash
git clone https://github.com/mosaicml/examples.git
cd examples
pip install -e ".[llm]"  # or pip install -e ".[llm-cpu]" if no NVIDIA GPU
cd examples/llm
```

# Extending an example

The examples in this repository are intended to be easily extensible. If you would like to extend an example, make sure you install the package in editable mode (i.e. use `pip install -e .[<example-name>]`).

Each example provides a short `main.py` which constructs the `Trainer` object and all of the arguments to it. To extend an example, all you need to do is modify the code that creates the `Trainer` to construct the arguments to `Trainer` with your new code. For example, if you wanted to make use of an optimizer that we haven't included in this repository yet (either one from an existing library or your own implementation), the simplest way is to modify the function `build_optimizer` in [common](./examples/common/builders.py) to import your optimizer and construct it. For more complicated extensions (e.g. creating a new model class or modifying the data loading), the process is pretty much the same. All you need to do is modify `main.py` to construct the relevant objects the way you want before passing them into the `Trainer`. If you run into any issues extending the code, or just want to discuss an idea you have, please open an [issue](https://github.com/mosaicml/examples/issues/new) or join our [community Slack](https://join.slack.com/t/mosaicml-community/shared_invite/zt-1btms90mc-GipE2ufuPkKY0QBrmF3LSA)!

# Tests and Linting

If you already have the dependencies for a given example installed, you can just run:
```bash
pre-commit run --all-files  # autoformatting
pyright .  # type checking
pytest tests/  # run tests
```
from the example's directory.

To run the full suite of tests for all examples, invoke `make test` in the project's root directory. Similarly, invoke `make lint` to autoformat your code and detect type issues throughout the whole codebase. This is much slower than linting or testing just one example because it installs all the dependencies for each example from scratch in a fresh virtual environment.

# Examples

This repo features the following examples, each as their own subdirectory:

## ResNet-50 + ImageNet
<img src="https://assets-global.website-files.com/61fd4eb76a8d78bc0676b47d/62a12d1e4eb9b83915be37a6_r50_overall_pareto.png" alt="drawing" width="500"/>

*Figure 1: Comparison of MosaicML recipes against other results, all measured on 8x A100s on MosaicML Cloud.*

Train the MosaicML ResNet, the fastest ResNet50 implementation that yields a :sparkles: 7x :sparkles: faster time-to-train compared to a strong baseline. See our [blog](https://www.mosaicml.com/blog/mosaic-resnet) for more details and recipes. Our recipes were also demonstrated at [MLPerf](https://www.mosaicml.com/blog/mlperf-2022), a cross industry ML benchmark.

:rocket: Get started with the code [here](./examples/resnet/).


## DeepLabV3 + ADE20k
<img src="https://assets-global.website-files.com/61fd4eb76a8d78bc0676b47d/637512d993030157b04ad4f8_Frame%2010%20(1).png" alt="drawing" width="500"/>

Train the MosaicML DeepLabV3 that yields a :sparkles:5x:sparkles: faster time-to-train compared to a strong baseline. See our [blog](https://www.mosaicml.com/blog/mosaic-image-segmentation) for more details and recipes.

:rocket: Get started with the code [here](./examples/deeplab/).


## Large Language Models (LLMs)

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./examples/llm/assets/loss-curve-dark.png">
  <img alt="Training curves for various LLM sizes." src="./examples/llm/assets/loss-curve-light.png" width=500px>
</picture>

A simple yet feature complete implementation of GPT, that scales to 70B parameters while maintaining high performance on GPU clusters. Flexible code, written with vanilla PyTorch, that uses [PyTorch FSDP](https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/) and some recent efficiency improvements.

:rocket: Get started with the code [here](./examples/llm/).


## BERT

This benchmark covers both pre-training and fine-tuning a BERT model. With this starter code, you'll be able to do Masked Language Modeling (MLM) pre-training on the C4 dataset and classification fine-tuning on GLUE benchmark tasks.

We also provide the source code and recipe behind our Mosaic BERT model, which you can train yourself using this repo.

:rocket: Get started with the code [here](./examples/bert/).
