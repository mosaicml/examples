# MosaicML platform end-to-end examples

This folder contains full examples of using the MosaicML platform to accomplish some task. Each example is self-contained and contains a series of steps to accomplish the task at hand.

Our current examples are:
- [`sec_10k_qa`](./sec_10k_qa/): A tutorial including processing SEC 10-K filings, finetuning MPT-7B on them, instruction finetuning MPT-7B, deploying an LLM and a text embedding model, and integrating the models into a LangChain and Gradio powered web application.
- [`stable_diffusion`](./stable_diffusion/): A tutorial including finetuning Stable Diffusion on a text-to-image dataset, and generating images from text prompts.
- [`stable_diffusion_dreambooth`](./stable_diffusion_dreambooth/): A tutorial including finetuning Stable Diffusion using Dreambooth.

# Before you start

These tutorials are written to be used with the MosaicML platform, which is only available to our customers. If you are signed up, make sure you've gone through the [getting started guide](https://docs.mosaicml.com/projects/mcli/en/latest/quick_start/getting_started.html). If you are not yet a customer but are interested in becoming one, please fill out our [interest form](https://forms.mosaicml.com/demo?utm_source=home&utm_medium=mosaicml.com&utm_campaign=always-on).

We also have multiple open source repositories that are used in these tutorials. If you are looking for documentation and repository-specific tutorials, please see each individual repository:

- [LLM-foundry](https://github.com/mosaicml/llm-foundry): An open-source PyTorch library of tools for training, finetuning, evaluating, and deploying LLMs for inference.
- [Diffusion](https://github.com/mosaicml/diffusion): An open-source PyTorch library for training Diffusion models.
- [Composer](https://github.com/mosaicml/composer): An open-source PyTorch library for easy large scale deep learning.
- [Streaming](https://github.com/mosaicml/streaming): An open-source PyTorch library for efficiently and accurately streaming data from the cloud to your training job, whereever that job is running.
