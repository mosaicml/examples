# Inference with Custom Models on the MosaicML Platform

In this folder, we provide an example of how to create and deploy your own text embedding model using MosaicML Inference.

You'll find in this folder:

- `src/hf_instructor_model_class.py` - a custom model class that implements the Hugging Face model [Instructor-Large](https://huggingface.co/hkunlp/instructor-large)
 - `yamls/instructor_model.yaml` - a configuration file that specifies information about the deployment, such as what Docker image to use

## Prerequisites

First, you'll need access to MosaicML's inference service. You can request access [here](https://forms.mosaicml.com/demo).

Once you have access, you just need to install the MosaicML client and command line interface:
```bash
pip install mosaicml-cli
```

## Configuring the Model

Currently, we offer support for any Hugging Face model and custom models that [adhere to a simple interface](https://docs.mosaicml.com/projects/mcli/en/latest/main_concepts/inference_schema.html#model).

The provided `.yaml` file is configured to deploy a custom model class. Information on the configuration options can be found [here](https://docs.mosaicml.com/projects/mcli/en/latest/main_concepts/inference_schema.html).

# Deploying the Model

Deploying the model is as simple as running `mcli deploy -f yamls/instructor_model.yaml`.

The logs for a successful deployment should look something like:

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./assets/instructor_model_logs.png">
  <img alt="Logs from server startup." src="./assets/instructor_model_logs.png">
</picture>

# Running Inference

Once the model has successfully been deployed, we can run inference by running `mcli predict <deployment_name> --inputs <input_dictionary>`.
