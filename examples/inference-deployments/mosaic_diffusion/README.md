:construction: This model is currently under construction. We'll update this README once it's good to go!


## Inference with Mosaic's Diffusion Model

[MosaicML’s inference service](https://www.mosaicml.com/blog/inference-launch) allows users to deploy their ML models and run inference on them. In this folder, we provide an example of how to use Mosaic's Stable Diffusion model, which we [trained for <$50k](https://www.mosaicml.com/blog/stable-diffusion-2) >. To learn more about Stable Diffusion itself, check out [the repo](https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines/stable_diffusion)!

You’ll find in this folder:

- Model YAMLS - read [docs](https://docs.mosaicml.com/projects/mcli/en/latest/inference/inference_schema.html) for an explanation of each field.
    - `mosaic_diffusion.yaml` - a yaml to deploy Mosaic's diffusion model.
- Model handlers - these define how your model should be loaded and how the model should be run when receiving a request. You can use the default handlers here or write your custom model handler as per instructions [here](https://docs.mosaicml.com/projects/mcli/en/latest/inference/deployment_features.html#custom-model-handlers).
    - `mosaic_diffusion_handler.py` - the model handler containing the default `predict` function.
- `requirements.txt` - package requirements to be able to run these models.


## Setup

Please follow instructions in the Inference Deployments [README](https://github.com/mosaicml/examples/tree/main/examples/inference-deployments/README.md) and make sure
- You have access to our inference service.
- Your dev environment is set up with `mcli`.
- You have a cluster to work with.

## Deploying your model

To deploy, simply run `mcli deploy -f mosaic_diffusion.yaml --cluster <your_cluster>`.

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
    "prompt": "a cute bunny"
}
predict(deployment, input)

```
</details>

<details>
<summary> Using MCLI </summary>

```bash
mcli predict <deployment-name> --input '{"prompt": "a cute bunny"s}'

```
</details>

<details>
<summary> Using Curl </summary>

```bash
curl https://<deployment-name>.inf.hosted-on.mosaicml.hosting/predict \
-H "Authorization: <your_api_key>" \
-d '{"prompt": "a cute bunny"}'
```
</details>


### Input parameters
| Parameters | Type | Required | Default | Description |
| --- | --- | --- | --- | --- |
| prompt | str or List[str] | yes | N/A | The prompt to generate an image for. |
| negative_prompt | str or List[str] | no | - | The prompt or prompts not to guide the image generation |
| height | int | no | 512 | Adjust height of generated images, best to have this value be a multiple of 8 |
| width | int | no | 512 | Adjust width of generated images, best to have this value be a multiple of 8 |
| num_inference_steps | int | no | 50 | The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference. |
| num_images_per_prompt | int | no | 1 | The number of images to generate per prompt. |
| seed | int | no | 0 | Used to generate images deterministically. |
| guidance_scale | float | no | 7.5 | Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).`guidance_scale` is defined as `w` of equation 2. of [Imagen Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`, usually at the expense of lower image quality. |


## Output

Generated images.

## Before you go

Your deployments will be live and using resources until you manually shut them down. Remember to run:
```
mcli delete deployment --name <deployment_name>
```
if you want to shut down your deployment!

## What's Next
 - Check out our [Diffusion model](https://github.com/mosaicml/diffusion), which contains code on how to train a Diffusion model on your own data.
 - Check out [our blogpost](https://www.mosaicml.com/blog/diffusion) about how we trained this Diffusion model.


## Additional Resources
- Check out the [MosaicML Blog](https://www.mosaicml.com/blog) to learn more about large scale AI.
- Follow us on [Twitter](https://twitter.com/mosaicml) and [LinkedIn](https://www.linkedin.com/company/mosaicml).
- Join our community on [Slack](https://mosaicml.me/slack).
