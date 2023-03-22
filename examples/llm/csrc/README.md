# Mosaic CUDA Extensions
This folder contains a customized CUDA extension for Mosaic's GPT models. This extension, along with the requirement listed in `examples/examples/llm/requirements_optimized_perf.txt`, are designed to speed up Mosaic's GPT models. These extensions should work on any modern NVIDIA GPU, but performance improvements will vary based on your GPU type and model size.

# Prerequisites
 Then, from the `examples/llm` folder, run:
```bash
pip install -r requirements_optimized_perf.txt
```
This will install a CUDA module from [HazyResearch](https://github.com/HazyResearch/).

# Installation
These extensions must be installed after the main [LLM requirements](../README.md#installation). After that, run the following command from this `csrc` folder:
```bash
python -m build --wheel --no-isolation  # may take a long time (~10 minutes)
pip install dist/*.whl
```

# Enabling the Optimizations
After installing the module in this folder, enable the optimizations by setting
```
model.gpt_block: optimized
```
in your YAML, or via the CLI like so: `composer main.py ... model.gpt_block=optimized`.

# Disabling the Optimizations
To disable the optimizations, set
```
model.gpt_block: standard
```
or, equivalently, omit `model.gpt_block` from your config entirely.

# Technical Details
A kernel fusion is designed to improve the performance of memory-bound operations. The CUDA module in this folder is a kernel fusion, [originally written by HazyResearch](https://github.com/HazyResearch/flash-attention/tree/eb33e587e95ec29a13c58f76dadca04b64122784/csrc/layer_norm), which we have modified to work on larger sizes (30B and 70B parameter models). This module fuses the dropout, addition, and LayerNorm pattern that occurs inside of each GPT block.

# Expected Performance
We have seen improvements of 5 - 15% for 1B - 13B parameter models on A100-40GB and A100-80GB nodes. Performance gains may be smaller if your batch size is very small or your model is extremely large.

# Credit
[HazyResearch](https://github.com/HazyResearch/).
