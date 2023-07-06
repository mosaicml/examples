## Inference with MPT-7B

[MosaicML’s inference service](https://www.mosaicml.com/inference) allows users to deploy their ML models and run inference on them. In this folder, we provide an example of how to use MPT-7B, a family of 6.7B parameter large language models, including the base model, an instruction fine-tuned variant, and a variant fine-tuned on long context books.

Check out [this blog post](https://www.mosaicml.com/blog/mpt-7b) for more information!

You’ll find in this folder:

- Model YAMLS - read [docs](https://docs.mosaicml.com/projects/mcli/en/latest/inference/inference_schema.html) for an explanation of each field.
    - `mpt_30b_ft.yaml` - a yaml to deploy [MPT-30B Base](https://huggingface.co/mosaicml/mpt-30b).
    - `mpt_30b_instruct_ft.yaml` - a yaml to deploy [MPT-30B Instruct](https://huggingface.co/mosaicml/mpt-30b-instruct).
    - `mpt_7b.yaml` - a yaml to deploy [MPT-7B Base](https://huggingface.co/mosaicml/mpt-7b).
    - `mpt_7b_instruct.yaml` - a yaml to deploy [MPT-7B Intstruct](https://huggingface.co/mosaicml/mpt-7b-instruct).
    - `mpt_7b_storywriter.yaml` - a yaml to deploy [MPT-7B StoryWriter](https://huggingface.co/mosaicml/mpt-7b-storywriter).
- Model handlers - these define how your model should be loaded and how the model should be run when receiving a request. You can use the default handlers here or write your custom model handler as per instructions [here](https://docs.mosaicml.com/projects/mcli/en/latest/inference/deployment_features.html#custom-model-handlers).
    - `mpt_handler.py` - a model handler using DeepSpeed.
    - `mpt_ft_handler.py` - a model handler using FasterTransformer.
- `requirements.txt` - package requirements to be able to run these models.


## Setup

Please follow instructions in the Inference Deployments [README](https://github.com/mosaicml/examples/tree/main/examples/inference-deployments/README.md) and make sure
- You have access to our inference service.
- Your dev environment is set up with `mcli`.
- You have a cluster to work with.

## Deploying your model

To deploy, simply run `mcli deploy -f mpt_7b_instruct.yaml --cluster <your_cluster>`.

Run `mcli get deployments` on the command line or, using the Python SDK, `mcli.get_inference_deployments()` to get the name of your deployment.


Once deployed, you can ping the deployment using
```python
from mcli import ping
ping('deployment-name')
```
to check if it is ready (status 200).

More instructions can be found [here](https://docs.mosaicml.com/projects/mcli/en/latest/quick_start/quick_start_inference.html)

You can also check the deployment logs with `mcli get deployment logs <deployment name>`.

### Deploying from cloud storage
If your model exists on Amazon S3 or Hugging Face, you can edit the YAML's model params to deploy it:
```yaml
model:
    download_parameters:
        s3_path: <your-s3-path>
    model_parameters:
        ...
        model_name_or_path: my/local/s3_path
```

If your model exists on a different cloud storage, then you can follow instructions [here](https://docs.mosaicml.com/projects/mcli/en/latest/inference/deployment_features.html#id1) on writing your custom downloader function, and deploy the model.

## Sending requests to your deployment

Once the deployment is ready, it's time to run inference!

<details>
<summary> Using Python SDK </summary>


```python
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
</details>

<details>
<summary> Using MCLI </summary>

```bash
mcli predict <deployment-name> --input '{"input_strings": ["hello world!"]}'

```
</details>

<details>
<summary> Using Curl </summary>

```bash
curl https://<deployment-name>.inf.hosted-on.mosaicml.hosting/predict \
-H "Authorization: <your_api_key>" \
-d '{"input_strings": ["hello world!"]}'
```
</details>

<details>
<summary> Using Langchain </summary>

```python
from getpass import getpass

MOSAICML_API_TOKEN = getpass()
import os

os.environ["MOSAICML_API_TOKEN"] = MOSAICML_API_TOKEN
from langchain.llms import MosaicML
from langchain import PromptTemplate, LLMChain
template = """Question: {question}"""

prompt = PromptTemplate(template=template, input_variables=["question"])
llm = MosaicML(inject_instruction_format=True, model_kwargs={'do_sample': False})
llm_chain = LLMChain(prompt=prompt, llm=llm)
question = "Write 3 reasons why you should train an AI model on domain specific data set."

llm_chain.run(question)

```
</details>

### Input parameters
| Parameters | Type | Required | Default | Description |
| --- | --- | --- | --- | --- |
| input_string | List[str] | yes | N/A | The prompt to generate a completion for. |
| top_p | float | no | 0.95 | Defines the tokens that are within the sample operation of text generation. Add tokens in the sample for more probable to least probable until the sum of the probabilities is greater than top_p |
| temperature | float | no | 0.8 | The temperature of the sampling operation. 1 means regular sampling, 0 means always take the highest score, 100.0 is getting closer to uniform probability |
| max_length | int | no | 256 | Defines the maximum length in tokens of the output summary |
<<<<<<< HEAD
| use_cache | bool | no | true | Whether to use KV cacheing during autoregressive decoding. This will use more memory but improve speed |
=======
| use_cache | bool | no | true | Whether to use KV caching during autoregressive decoding. This will use more memory but improve speed |
>>>>>>> 679fd5b (small fix)
| do_sample | bool | no | true | Whether or not to use sampling, use greedy decoding otherwise |


## Output

```
{
    'data': [
        '1. The model will be more accurate.\n2. The model will be more efficient.\n3. The model will be more interpretable.'
    ]
}
```

## Before you go

<<<<<<< HEAD
Your deployments will be live and using resources until you manually shut them down. Remember to run:
```
mcli delete deployment --name <deployment_name>
```

## What's Next
 - Check out our [LLM foundry](https://github.com/mosaicml/llm-foundry), which contains code to train, finetune, evaluate and deploy LLMs.
=======
Your deployments will be live and using resources until you manually shut them down. In order to delete your deployment, remember to run:
```
mcli delete deployment --name <deployment_name>
```

## What's Next
 - Check out our [LLM foundry](https://github.com/mosaicml/llm-foundry), which contains code to train, fine-tune, evaluate and deploy LLMs.
>>>>>>> 679fd5b (small fix)
 - Check out the [Prompt Engineering Guide](https://www.promptingguide.ai) to better understand LLMs and how to use them.


## Additional Resources
- Check out the [MosaicML Blog](https://www.mosaicml.com/blog) to learn more about large scale AI
- Follow us on [Twitter](https://twitter.com/mosaicml) and [LinkedIn](https://www.linkedin.com/company/mosaicml)
- Join our community on [Slack](https://mosaicml.me/slack)
