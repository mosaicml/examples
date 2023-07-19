# MosaicML Examples

This repo contains reference examples for using the MosaicML platform to train and deploy machine learning models at scale. It's designed to be easily forked/copied and modified.

It is structured with four different types of examples:

* [benchmarks](./examples/benchmarks/): Instructions for how to reproduce the cost estimates that we publish in our blogs. Start here if you are looking to verify or learn more about our cost estimates.
* [end-to-end-examples](./examples/end-to-end-examples/): Complete examples of using the MosaicML platform, starting from data processing and ending with model deployment. Start here if you are looking full MosaicML platform usage examples.
* [inference-deployments](./examples/inference-deployments/): Example model handlers and deployment yamls for deploying a model with MosaicML inference. Start here if you are looking to deploy a model.
* [third-party](./examples/third-party/): Example usages of the MosaicML platform with third-party distributed training libraries. Start here if you are looking to try out the MosaicML platform with non-MosaicML training software.

Please see the README in each folder for more information about each type of example.

## Tests and Linting

To run the lint and test suites for a specific folder, you can use the `lint_subdirectory.sh` and `test_subdirectory.sh` scripts:
```bash
bash ./scripts/lint_subdirectory.sh benchmarks/bert
bash ./scripts/test_subdirectory.sh benchmarks/bert
```

## Other MosaicML repositories and documentation
- [MosaicML platform sign up](https://forms.mosaicml.com/demo?utm_source=home&utm_medium=mosaicml.com&utm_campaign=always-on)
- [LLM-foundry](https://github.com/mosaicml/llm-foundry)
- [Diffusion](https://github.com/mosaicml/diffusion)
- [Composer](https://github.com/mosaicml/composer)
- [Streaming](https://github.com/mosaicml/streaming)
- [MosaicML docs](https://docs.mosaicml.com/en/latest/)
- [MosaicML blog](https://www.mosaicml.com/blog)
