### :construction: :construction: :construction: THIS REPO IS UNDER CONSTRUCTION :construction: :construction: :construction:
We are restructuring this repo to focus on examples usages of the MosaicML platform, and more clear examples of end-to-end usages of MosaicML's tools. If you were previously using the examples repo, our most recent stable release is [v0.0.4](https://github.com/mosaicml/examples/releases/tag/v0.0.4), and the commit before this restructuring began is [e37d79874dc9f7c2409e076a5155ff7d4c9d445c](https://github.com/mosaicml/examples/tree/e37d79874dc9f7c2409e076a5155ff7d4c9d445c).

If you are looking for:
- our LLM training stack, see [llm-foundry](https://github.com/mosaicml/llm-foundry)
- our diffusion training stack, see [diffusion](https://github.com/mosaicml/diffusion)
- our inference handlers and deployment yamls see [inference](https://github.com/mosaicml/examples/tree/main/examples/inference-deployments)
- our docs, see [docs](https://docs.mosaicml.com/en/latest/)
- our BERT training code, see [the released version](https://github.com/mosaicml/examples/tree/v0.0.4/examples/bert)

# MosaicML Examples

This repo contains reference examples for using the MosaicML platform to train and deploy machine learning models at scale. It's designed to be easily forked/copied and modified.

It is structured with four different types of examples:

* [benchmarks](./examples/benchmarks/): Instructions for how to reproduce the cost estimates that we publish in our blogs. Start here if you are looking to verify or learn more about our cost estimates.
* [end-to-end-examples](./examples/end-to-end-examples/): Complete examples of using the MosaicML platform, starting from data processing and ending with model deployment. Start here if you are looking to get something up and running that you can hack on using the MosaicML platform.
* [inference-deployments](./examples/inference-deployments/): Example model handlers and deployment yamls for deploying a model with MosaicML inference. Start here if you are looking to deploy a model.
* [third-party](./examples/third-party/): Example usages of the MosaicML platform with third-party distributed training libraries. Start here if you are looking to try out the MosaicML platform with non-MosaicML training software.

Please see the README in each folder for more information about each type of example.

## Tests and Linting

To run the lint and test suites for a specific folder, you can use the `lint_subdirectory.sh` and `test_subdirectory.sh` scripts:
```bash
bash ./scripts/lint_subdirectory.sh benchmarks/bert
bash ./scripts/test_subdirectory.sh benchmarks/bert
```
