

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

# Results
Finetuning for x iterations yeilds good results





```

## Results



# Saving and Loading checkpoints

At the bottom of `yamls/resnet56.yaml`, we provide arguments for saving and loading model weights. Please specify the `save_folder` or `load_path` arguments if you need to save or load checkpoints!

# Contact Us
If you run into any problems with the code, please file Github issues directly to this repo.
