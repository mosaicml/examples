# MosaicML Inference Model Handlers

This repo contains model handler implementations for various models and can be used out-of-the box with the yamls provided using the MosaicML inference service. These are available for anyone to easily get started with the MosaicML inference service. If a model that you'd like to deploy isn't supported by one of the existing model handlers, you can flexibly configure the behavior of your models in the server by implementing your own model handler. The model handlers are expected to be a class that implements a `predict` function and optionally a `predict_stream` function if you'd like your deployment to support streaming outputs. For more details about the structure of the model handlers, please refer to the [mcli docs](https://docs.mosaicml.com/projects/mcli/en/latest/main_concepts/inference_schema.html)

## Getting Started

First, you'll need access to MosaicML's inference service. You can request access [here](https://forms.mosaicml.com/demo).

After getting started, follow the [MCLI Docs](https://docs.mosaicml.com/projects/mcli/en/latest/inference_getting_started/quick_start.html) for quick start instructions.

Once you have `mcli` set up, take a look at the README for each of the models in this repo for the command to deploy the model.
