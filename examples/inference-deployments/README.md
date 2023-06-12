# MosaicML Inference Model Handlers

The following instructions will guide you on how to get started with [MosaicML’s inference service](https://www.mosaicml.com/blog/inference-launch). 

In this folder, we provide examples of model handler implementations for various models. They can be used out-of-the box with the yamls provided using the MosaicML inference service. 

## Getting Started

Before using the inference service, you must request access [here](https://forms.mosaicml.com/demo).

Once you have access, 

1. Follow instructions [here](https://docs.mosaicml.com/projects/mcli/en/latest/quick_start/getting_started.html) to install mosaicml-cli, our command line interface that will allow you to deploy models and run inference on them.
2. Once you have `mcli` set up, the [Inference Docs](https://docs.mosaicml.com/projects/mcli/en/latest/quick_start/quick_start_inference.html) will give you a high level overview of how you can deploy your model and interact with it.
3. Now, you are ready to look at the README for each of the models in this repo to start running inference on them!


## Content
We have provided examples for 3 different model types in this repo:

1. MPT series
2. Instructor 
3. Mosaic diffusion

Each of these model directories have model handlers and yaml files.

### Model Handlers

Model handlers define how your model is loaded and how the model should be run when receiving a request. The model handlers are expected to be a class that implements a `predict` function and optionally a `predict_stream` function if you'd like your deployment to support streaming outputs. For more details about the structure of the model handlers, please refer to the [mcli docs](https://docs.mosaicml.com/projects/mcli/en/latest/inference/deployment_features.html#model-handlers)

If a model that you'd like to deploy isn't supported by one of the existing model handlers, you can flexibly configure the behavior of your models in the server by implementing your own model handler. 


### YAMLs

Deployment submissions to the MosaicML platform can be configured through a YAML file or using our Python API’s `InferenceDeploymentConfig` class. We have provided YAMLs in these examples which contain information like name, image, download path for the model, and more. Please see [this link](https://docs.mosaicml.com/projects/mcli/en/latest/inference/inference_schema.html) to understand what these parameters mean.