## Inference with Llama2

[MosaicML’s inference service](https://www.mosaicml.com/inference) allows users to deploy their ML models and run inference on them. In this folder, we provide an example of how to deploy any Llama2 model, A state-of-the-art 70B parameter language model with a context length of 4096 tokens, trained by Meta. Llama 2 is licensed under the [LLAMA 2 Community License](https://github.com/facebookresearch/llama/blob/main/LICENSE), Copyright © Meta Platforms, Inc. All Rights Reserved. Customers are responsible for ensuring compliance with applicable model licenses.

Check out MosaicML's [Llama2 blog post](https://www.mosaicml.com/blog/llama2-inferenceb) for more information!

You’ll find in this folder:

- Model YAMLS - read [docs](https://docs.mosaicml.com/projects/mcli/en/latest/inference/inference_schema.html) for an explanation of each field.
    - `llama2_7b_chat.yaml` - an optimized yaml to deploy [Llama2 7B Chat](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf).
    - `llama2_13b.yaml` - an optimized yaml to deploy [Llama2 13B Base](https://huggingface.co/meta-llama/Llama-2-13b-hf).

## Setup

Please follow instructions in the Inference Deployments [README](https://github.com/mosaicml/examples/tree/main/examples/inference-deployments/README.md) and make sure
- You have access to our inference service.
- Your dev environment is set up with `mcli`.
- You have a cluster to work with.

## Deploying your model

To deploy, simply run `mcli deploy -f llama2_7b_chat.yaml --cluster <your_cluster>`.

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
If your model exists on Amazon S3, GCP, or Hugging Face, you can edit the YAML's `checkpoint_path` to deploy it. Keep in mind that the checkpoint_path sources are mutually exclusive, so you can only set one of `hf_path`, `s3_path`, or `gcp_path`:

```yaml
default_model:
  checkpoint_path:
    hf_path: meta-llama/Llama-2-13b-hf
    s3_path: s3://<your-s3-path>
    gcp_path: gs://<your-gcp-path>

```

If your model exists on a different cloud storage, then you can follow instructions [here](https://docs.mosaicml.com/projects/mcli/en/latest/inference/deployment_features.html#id1) on writing your custom downloader function, and deploy the model with the [custom yaml format](https://docs.mosaicml.com/projects/mcli/en/latest/inference/inference_schema.html#custom-model).

## Sending requests to your deployment

Once the deployment is ready, it's time to run inference! Detailed information about the Llama2 prompt format can be found [here](https://www.mosaicml.com/blog/llama2-inference).

<details>
<summary> Using Python SDK </summary>


```python
from mcli import predict

prompt = """[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. 
Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>
How do I make a customer support bot using my product docs? [/INST]"""

deployment = get_inference_deployment(<deployment-name>)
input = {
    {
        "inputs": prompt,
        "temperature": 0.01
    }
}
predict(deployment, input)

```
</details>

<details>
<summary> Using MCLI </summary>

```bash
mcli predict <deployment-name> --input '{"inputs": ["hello world!"]}'

```
</details>

<details>
<summary> Using Curl </summary>

```bash
curl https://<deployment-name>.inf.hosted-on.mosaicml.hosting/predict \
-H "Authorization: <your_api_key>" \
-d '{"inputs": ["hello world!"]}'
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
| max_new_tokens | int | no | 256 | Defines the maximum length in tokens of the output summary |
| use_cache | bool | no | true | Whether to use KV caching during autoregressive decoding. This will use more memory but improve speed |
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

Your deployments will be live and using resources until you manually shut them down. In order to delete your deployment, remember to run:
```
mcli delete deployment --name <deployment_name>
```

## What's Next
 - Check out our [LLM foundry](https://github.com/mosaicml/llm-foundry), which contains code to train, fine-tune, evaluate and deploy LLMs.
 - Check out the [Prompt Engineering Guide](https://www.promptingguide.ai) to better understand LLMs and how to use them.


## Additional Resources
- Check out the [MosaicML Blog](https://www.mosaicml.com/blog) to learn more about large scale AI
- Follow us on [Twitter](https://twitter.com/mosaicml) and [LinkedIn](https://www.linkedin.com/company/mosaicml)
- Join our community on [Slack](https://mosaicml.me/slack)
