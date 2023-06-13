## Inference with Instructor

[MosaicML’s inference service](https://www.mosaicml.com/blog/inference-launch) allows users to deploy their ML models and run inference on them. In this folder, we provide an example of how to use [Instructor-Large](https://huggingface.co/hkunlp/instructor-large) and [Instructor-XL](https://huggingface.co/hkunlp/instructor-xl), which are both instruction-finetuned text embedding models developed by the NLP Group of The University of Hong Kong. The instructor model series is state-of-the-art on the [Massive Text Embedding Benchmark (MTEB)](https://huggingface.co/blog/mteb).


You’ll find in this folder:

- Model YAMLS - read [docs](https://docs.mosaicml.com/projects/mcli/en/latest/inference/inference_schema.html) for an explanation of each field.
    - `instructor_large.yaml` - a yaml to deploy Instructor-large.
    - `instructor_xl.yaml` - a yaml to deploy Instructor-xl.
- Model handlers - these define how your model should be loaded and how the model should be run when receiving a request. You can use the default handlers here or write your custom model handler as per instructions [here](https://docs.mosaicml.com/projects/mcli/en/latest/inference/deployment_features.html#custom-model-handlers).
    - `instructor_handler.py` - a model handler for both Instructor models.
- `requirements.txt` - package requirements to be able to run these models.


## Setup

Please follow instructions in the Inference Deployments [README](https://github.com/mosaicml/examples/tree/main/examples/inference-deployments/README.md) and make sure
- You have access to our inference service.
- Your dev environment is set up with `mcli`.
- You have a cluster to work with.

## Deploying your model

To deploy the instructor-large model, run: `mcli deploy -f instructor_large.yaml --cluster <your_cluster>`

To deploy the instructor-xl model, run: `mcli deploy -f instructor_xl.yaml --cluster <your_cluster>`.

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
    "input_strings": [
        ["Represent the Science title:", "3D ActionSLAM: wearable person tracking in multi-floor environments"]
    ]
}

predict(deployment, input)

```
</details>

<details>
<summary> Using MCLI </summary>

```bash
mcli predict <deployment-name> --input '{"input_strings":  [["Represent the Science title:", "3D ActionSLAM: wearable person tracking"]]}'

```
</details>

<details>
<summary> Using Curl </summary>

```bash
curl https://<deployment-name>.inf.hosted-on.mosaicml.hosting/predict \
-H "Authorization: <your_api_key>" \
-d '{"input_strings":  [["Represent the Science title:", "3D ActionSLAM: wearable person tracking in multi-floor environments"]]}'
```
</details>


### Input parameters
| Parameters | Type | Required | Default | Description |
| --- | --- | --- | --- | --- |
| input_strings | List[Tuple[str, str]] | yes | N/A | A list of documents and instructions to embed. Each document is represented as tuple where the first item is the embedding instruction (e.g. "Represent the Science title:") and the second item is the document (e.g. "3D ActionSLAM: wearable person tracking in multi-floor environments"). |



## Output

```
{
    "data":[
        [
            -0.06155527010560036,0.010419987142086029,0.005884397309273481...-0.03766140714287758,0.010227023623883724,0.04394740238785744
        ]
    ]
}
```

## Before you go

Your deployments will be live and using resources until you manually shut them down. Remember to run:
```
mcli delete deployment --name <deployment_name>
```
if you want to shut down your deployment!

## What's Next
 - Check out the [Instructor paper](https://instructor-embedding.github.io) to better understand Instructor embedding models.
 - Check out our [LLM foundry](https://github.com/mosaicml/llm-foundry), which contains code to train, finetune, evaluate and deploy LLMs.
 


## Additional Resources
- Check out the [MosaicML Blog](https://www.mosaicml.com/blog) to learn more about large scale AI
- Follow us on [Twitter](https://twitter.com/mosaicml) and [LinkedIn](https://www.linkedin.com/company/mosaicml)
- Join our community on [Slack](https://mosaicml.me/slack)
