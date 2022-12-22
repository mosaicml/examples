# Using MosaicML Cloud to train LLMs

This folder contains some examples of how to use [MosaicML Cloud](https://www.mosaicml.com/cloud) to launch LLM training runs.

Full documentation on MosaicML Cloud can be found at https://mcli.docs.mosaicml.com/.

## Using MCLI to launch runs

In this folder, we provide two MCLI examples, [`mcli-1b.yaml`](./mcli-1b.yaml) and [`mcli-1b-custom.yaml`](./mcli-1b-custom.yaml) that demonstrate how to configure and launch training runs using our command-line tool, `mcli`.

The first example, `mcli-1b.yaml`, is minimal, and simply clones this repo (https://github.com/mosaicml/examples), checks out a particular tag, and runs the `main.py` training script. The workload configuration is read from a YAML sitting in the repo (`yamls/mosaic_gpt/1b.yaml`).

The second example, `mcli-1b-custom.yaml`, demonstrates how to inject a custom YAML at runtime (`/mnt/config/parameters.yaml`) and instead pass that file to `main.py`. This enables users to quickly customize training runs without needing to check in their edits to the repository.

Here's how easy it is to launch an LLM training run with MCLI:
```bash
mcli run -f mcli-1b.yaml --cluster CLUSTER --gpus GPUS --name NAME --follow
```

All the details of multi-gpu and multi-node orchestration are handled automatically by MosaicML Cloud. Try it out yourself!
## Using the Python API to launch runs

TBD...
