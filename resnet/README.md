<br />
<p align="center">
   <img src="https://assets.website-files.com/61fd4eb76a8d78bc0676b47d/62a185326fcd73061ab9aaf9_Hero%20Image%20Final.svg" width="50%" height="50%"/>
</p>

<h2><p align="center">The most efficient recipes for training ResNets on ImageNet</p></h2>

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

# Mosaic ResNet
This folder contains starter code for training [torchvision ResNet architectures](https://pytorch.org/vision/stable/models.html) using our most efficient training recipes, see our [short blog](https://www.mosaicml.com/blog/mosaic-resnet) or [long blog](https://www.mosaicml.com/blog/mosaic-resnet-deep-dive) for details. These recipes were developed to hit baseline accuracy on [ImageNet](https://www.image-net.org/) 7x faster or to maximize ImageNet accuracy over long training durations. Although these recipes were developed for training ResNet on ImageNet, they could be used to train other image classification models on other datasets. Give it a try!

The specific files in this folder are:
* `model.py` - A function to create a [ComposerModel](https://docs.mosaicml.com/en/v0.11.0/composer_model.html) from a torchvision ResNet model
* `data.py` - A [MosaicML streaming dataset](https://docs.mosaicml.com/projects/streaming/en/latest/) for ImageNet and a PyTorch dataset for a local copy of ImageNet
* `main.py` - A script that builds a [Composer](https://github.com/mosaicml/composer) Trainer based on a configuration specified by a yaml
* `yamls/`
  * `resnet50.yaml` - Configuration for a ResNet50 training run to be used as the first argument to `main.py`
  * `mcloud_run.yaml` - yaml to use if running on the [MosaicML Cloud](https://www.mosaicml.com/blog/introducing-mosaicml-cloud)

Now that you've had a chance to explore the code, let's jump into the prerequisites for training:

# Prequisites

Here's what you need to train:

* Docker image with PyTorch 1.12+, e.g. [MosaicML's PyTorch image](https://hub.docker.com/r/mosaicml/pytorch/tags)
   * Recommended tag: `mosaicml/pytorch:1.12.1_cu116-python3.9-ubuntu20.04`
   * The image comes pre-configured with the following dependencies:
      * PyTorch Version: 1.12.1
      * CUDA Version: 11.6
      * Python Version: 3.9
      * Ubuntu Version: 20.04
* [Imagenet Dataset](http://www.image-net.org/) must be stored either locally or uploaded to an S3 bucket after converting to a streaming format using [this script](https://github.com/mosaicml/streaming/blob/86a9b95189e8b292a8c7880a1c49dc55d1895544/streaming/vision/convert/imagenet.py)
* System with NVIDIA GPUs
* Install requirements via `pip install -r requirements.txt` which installs:
  * `composer`
  * `streaming` - MosaicML's streaming dataset
  * `wandb` - Weights and Biases for experiment tracking
  * `omegaconf` - Configuration management

# Test Dataloader

This benchmark assumes that ImageNet is already stored on your local machine or stored in an S3 bucket after being processed into a streaming dataset. Information on downloading ImageNet can be found [here](https://www.image-net.org/download.php). Alternatively, you can train on other image classification datasets. The local dataset code assumes your data is in the [torchvision ImageFolder](https://pytorch.org/vision/main/generated/torchvision.datasets.ImageFolder.html) format.

The below command will test if your data is setup appropriately:
```bash
# Test locally stored dataset
python data.py path/to/data

# Test remote storage dataset
python data.py s3://my-bucket/my-dir/data /tmp/path/to/local
```

# How to start training

Now that you've installed dependencies and tested your dataset, let's start training!

**Please remember** to edit the `path` and (if streaming) `local` paths in the yaml to point to your data.

### Single-Node training
We run the `main.py` script using our `composer` launcher, which generates a process for each device.

If training on a single node, the `composer` launcher will autodetect the number of devices, so all you need to do is :

```bash
composer main.py yamls/resnet50.yaml
```

To train with high performance on multi-node clusters, the easiest way is with MosaicML Cloud ;)

But if you really must try this manually on your own cluster, then just provide a few variables to `composer`
either directly via CLI, or via environment variables that can be read. Then launch the appropriate command on each node:

### Multi-Node via CLI args
```bash
# Using 2 nodes with 8 devices each
# Total world size is 16
# IP Address for Node 0 = [0.0.0.0]

# Node 0
composer --world_size 16 --node_rank 0 --master_addr 0.0.0.0 --master_port 7501 main.py yamls/resnet50.yaml

# Node 1
composer --world_size 16 --node_rank 1 --master_addr 0.0.0.0 --master_port 7501 main.py main.py yamls/resnet50.yaml
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
composer main.py yamls/resnet50.yaml

# Node 1
# export WORLD_SIZE=16
# export NODE_RANK=1
# export MASTER_ADDR=0.0.0.0
# export MASTER_PORT=7501
composer main.py yamls/resnet50.yaml
```

### Results
You should see logs being printed to your terminal like so.
You can also easily enable other experiment trackers like Weights and Biases or CometML,
by using [Composer's logging integrations](https://docs.mosaicml.com/en/v0.10.0/trainer/logging.html).

```bash
logggggssssss
```
# Using Mosaic Recipes

As described in our [ResNet blog post](https://www.mosaicml.com/blog/mosaic-resnet), we cooked up three recipes to train ResNet faster and with higher accuracy:
- **Mild** recipe is for shorter training runs
- **Medium** recipe is for longer training runs
- **Hot** recipe is for the very longest training runs that maximize accuracy

<img src="https://assets.website-files.com/61fd4eb76a8d78bc0676b47d/62a188a808b39301a7c3550f_Recipe%20Final.svg" width="50%" height="50%"/>

To use a recipe, use the `recipe_name` argument to specify the recipe. Specifying a recipe will change several aspects of the training run:
1. Set the loss function to binary cross entropy instead of standard cross entropy since this has been shown to improve acurracy.
2. Set the train crop size to 176 instead of 224 and evaluation resize size to 232 from 256. This has been show to improve accuracy and the smaller train crop size increases throughput.
3.  Set the number of training epochs to the optimal value for each training recipe. Feel free to change these in `resnet50.yaml` to better suite your task.
4.  Specifies unique sets of speedup methods for model training.

Here is an example command to run the mild recipe on a single-node:
```bash
composer main.py yamls/resnet50.yaml recipe_name=mild
```

# Saving and Loading checkpoints

At the bottom of `yamls/resnet50.yaml`, we provide arguments for saving and loading model weights. Please uncomment and specify the arugments if you need to save or load weights!

# On memory constraints
In previous blogs ([1](https://www.mosaicml.com/blog/farewell-oom), [2](https://www.mosaicml.com/blog/billion-parameter-gpt-training-made-easy))
we demonstrated Auto Grad Accum, which allows Composer to determine `grad_accum` on its own. This means the same configuration can be run on different hardware or on a fewer number of devices without having to manually adjust the gradient accumulation! We have done extensive testing on this feature, but if there are any issues you can manually set `grad_accum` to your desired value.

# Contact Us
If you run into any problems with the code, please file Github issues directly to this repo.