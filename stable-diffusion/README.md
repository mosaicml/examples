

<h2><p align="center">Stable Diffusion Finetuning</p></h2>

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
    <a href="https://github.com/mosaicml/examples/blob/main/LICENSE">
        <img alt="License" src="https://img.shields.io/badge/License-Apache%202.0-green.svg?logo=slack">
    </a>
</p>
<br />

This folder contains starter code for finetuning Stable Diffusion on the lambda labs pokemon dataset. You can finetune Stable Diffusion v1 or 2, save the model and generate your own pokemon, or swap out the dataset to adapt stable diffusion to your desired domain.


# Quick Start

## Clone the repo
```bash
git clone mosaicml/examples
cd examples/stable-diffusion
```
## Install requirements

```
pip install -r requirements.txt 
```
## Train the model
```
composer main.py yamls/finetune.yaml
```

#





```

## Results

You should see logs printed to your terminal like below. You can also easily enable other experiment trackers like Weights and Biases or CometML using [Composer's logging integrations](https://docs.mosaicml.com/en/v0.11.0/trainer/logging.html).

```bash
[epoch=0][batch=16/625]: wall_clock/train: 17.1607
[epoch=0][batch=16/625]: wall_clock/val: 10.9666
[epoch=0][batch=16/625]: wall_clock/total: 28.1273
[epoch=0][batch=16/625]: lr-DecoupledSGDW/group0: 0.0061
[epoch=0][batch=16/625]: trainer/global_step: 16
[epoch=0][batch=16/625]: trainer/batch_idx: 16
[epoch=0][batch=16/625]: memory/alloc_requests: 38424
[epoch=0][batch=16/625]: memory/free_requests: 37690
[epoch=0][batch=16/625]: memory/allocated_mem: 6059054353408
[epoch=0][batch=16/625]: memory/active_mem: 1030876672
[epoch=0][batch=16/625]: memory/inactive_mem: 663622144
[epoch=0][batch=16/625]: memory/reserved_mem: 28137488384
[epoch=0][batch=16/625]: memory/alloc_retries: 3
[epoch=0][batch=16/625]: trainer/grad_accum: 2
[epoch=0][batch=16/625]: loss/train/total: 7.1292
[epoch=0][batch=16/625]: metrics/train/Accuracy: 0.0005
[epoch=0][batch=17/625]: wall_clock/train: 17.8836
[epoch=0][batch=17/625]: wall_clock/val: 10.9666
[epoch=0][batch=17/625]: wall_clock/total: 28.8502
[epoch=0][batch=17/625]: lr-DecoupledSGDW/group0: 0.0066
[epoch=0][batch=17/625]: trainer/global_step: 17
[epoch=0][batch=17/625]: trainer/batch_idx: 17
[epoch=0][batch=17/625]: memory/alloc_requests: 40239
[epoch=0][batch=17/625]: memory/free_requests: 39497
[epoch=0][batch=17/625]: memory/allocated_mem: 6278452575744
[epoch=0][batch=17/625]: memory/active_mem: 1030880768
[epoch=0][batch=17/625]: memory/inactive_mem: 663618048
[epoch=0][batch=17/625]: memory/reserved_mem: 28137488384
[epoch=0][batch=17/625]: memory/alloc_retries: 3
[epoch=0][batch=17/625]: trainer/grad_accum: 2
[epoch=0][batch=17/625]: loss/train/total: 7.1243
[epoch=0][batch=17/625]: metrics/train/Accuracy: 0.0010
train          Epoch   0:    3%|▋                        | 17/625 [00:17<07:23,  1.37ba/s, loss/train/total=7.1292]
```

# Saving and Loading checkpoints

At the bottom of `yamls/resnet56.yaml`, we provide arguments for saving and loading model weights. Please specify the `save_folder` or `load_path` arguments if you need to save or load checkpoints!

# Contact Us
If you run into any problems with the code, please file Github issues directly to this repo.
