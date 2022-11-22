<br />
<h2><p align="center">CIFAR10 Benchmark</p></h2>

<h3><p align='center'>
<a href="https://www.mosaicml.com">[Website]</a>
- <a href="https://docs.mosaicml.com/">[Composer Docs]</a>
- <a href="https://docs.mosaicml.com/en/stable/method_cards/methods_overview.html">[Methods]</a>
- <a href="https://www.mosaicml.com/team">[We're Hiring!]</a>
</p></h3>

<p align="center">
    <a href="https://join.slack.com/t/mosaicml-community/shared_invite/zt-w0tiddn9-WGTlRpfjcO9J5jyrMub1dg">
        <img alt="Chat @ Slack" src="https://img.shields.io/badge/slack-chat-2eb67d.svg?logo=slack">
    </a>
    <a href="https://github.com/mosaicml/benchmarks/blob/main/LICENSE">
        <img alt="License" src="https://img.shields.io/badge/License-Apache%202.0-green.svg?logo=slack">
    </a>
</p>
<br />

This folder contains starter code for training a CIFAR ResNet architecture. You can swap out the model and dataset if desired, but we recommend using the [ResNet + ImageNet benchmark](../resnet/) for new models and datasets.

The specific files in this folder are:
* `model.py` - Creates a [ComposerModel](https://docs.mosaicml.com/en/v0.11.0/composer_model.html) from a CIFAR ResNet model defined in the script
* `data.py` - Creates a PyTorch dataset for a local copy of CIFAR10 and a [MosaicML streaming dataset](https://docs.mosaicml.com/projects/streaming/en/latest/) for CIFAR10
* `main.py` - Trains a CIFAR ResNet on CIFAR10 using the [Composer](https://github.com/mosaicml/composer) [Trainer](https://docs.mosaicml.com/en/stable/api_reference/generated/composer.Trainer.html#trainer)
* `tests/` - A suite of tests to check each training component
* `yamls/`
  * `resnet56.yaml` - Configuration for a CIFAR ResNet56 training run, to be used as the first argument to `main.py`
  * `mcloud_run.yaml` - yaml to use if running on the [MosaicML Cloud](https://www.mosaicml.com/blog/introducing-mosaicml-cloud)