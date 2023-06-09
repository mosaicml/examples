## Inference with MPT-7B 

[MosaicML’s inference service](https://www.mosaicml.com/blog/inference-launch) allows users to deploy their ML models and run inference on them. In this folder, we provide an example of how to use MPT-7B, a state-of-the-art 6.7B parameter instruction fine-tuned language model trained by MosaicML for inference. Check out the [MPT-Instruct huggingface repo](https://huggingface.co/spaces/mosaicml/mpt-7b-instruct) for more information about the model.

You’ll find in this folder:

- Model YAMLS - read [docs](https://docs.mosaicml.com/projects/mcli/en/latest/inference/inference_schema.html) for an explanation of each field.
    - `mpt_7b.yaml` - a yaml to deploy [MPT-7B Base](https://huggingface.co/mosaicml/mpt-7b).
    - `mpt_7b_instruct.yaml` - a yaml to deploy [MPT-7B Intstruct](https://huggingface.co/mosaicml/mpt-7b-instruct).
    - `mpt_7b_storywriter.yaml` - a yaml to deploy [MPT-7B StoryWriter](https://huggingface.co/mosaicml/mpt-7b-storywriter).
- Model handlers - these define how your model should be loaded and what should happen in a forward pass. You can use the default handlers here or write your custom model handler as per instructions [here](https://docs.mosaicml.com/projects/mcli/en/latest/inference/deployment_features.html#custom-model-handlers).
    - `mpt_7b_handler.py` - a python script using DeepSpeed.
    - `mpt_7b_ft_handler.py` - a python script using FasterTransformer.
- `requirements.txt` - package requirements to be able to run these models.

## Deploying your model

To deploy, simply run `mcli deploy -f mpt_7b_instruct.yaml --cluster <your_cluster>`.

Once deployed, you can ping the deployment using
```
from mcli import ping
ping('deployment-name')
```
to check if it is ready (status 200).

More instructions can be found [here](https://docs.mosaicml.com/projects/mcli/en/latest/quick_start/quick_start_inference.html)



## Sending requests to your deployment

Once the deployment is ready, it's time to run inference! 



Using the Python SDK:
```
from mcli import predict

deployment = get_inference_deployment(<deployment-name>)
input = {
    {
        "input_strings": "Write 3 reasons why you should train an AI model on domain specific data set.",
        "temperature": 0.01
    }
}
predict(deployment, input)

```

You can also use [curl or command line](https://docs.mosaicml.com/projects/mcli/en/latest/quick_start/quick_start_inference.html#interacting-with-your-deployment) to send your requests.


```{csv-table}
:header: >
:    "Parameters", "Type", "Required", "Default", "Description"
:widths: 20, 20, 5, 10, 50

"input_string","List[str]","yes","N/A","The prompt to generate a completion for."
"top_p","float","no","0.95","Defines the tokens that are within the sample operation of text generation. Add tokens in the sample for more probable to least probable until the sum of the probabilities is greater than top_p."
"temperature","float","no","0.8","The temperature of the sampling operation. 1 means regular sampling, 0 means always take the highest score, 100.0 is getting closer to uniform probability."
"max_length","int","no","256","Defines the maximum length in tokens of the output summary."
"use_cache","bool","no","TRUE","Whether to use KV cacheing during autoregressive decoding. This will use more memory but improve speed."
"do_sample","bool","no","TRUE","Whether or not to use sampling, use greedy decoding otherwise."
```

## Output

```
{
    'data': [
        '1. The model will be more accurate.\n2. The model will be more efficient.\n3. The model will be more interpretable.'
    ]
}
```


## Before you go

Your deployments will be live and using resources until you manually shut them down. Remember to run:
```
mcli delete deployment --name <deployment_name>
```
if you want to shut down your deployment!
