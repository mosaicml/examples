# Mosaic CUDA Extensions
This folder contains CUDA extensions for Mosaic's GPT models. These extensions should work on any modern NVIDIA GPU, but performance improvements will vary based on your GPU type and model size.

These extensions must be installed after the main LLM folder. To install, run the following command from this `csrc` folder:
```bash
pip install . # may take a long time (up to 20 minutes)
```

After installing, enable the optimizations by setting
```
model.gpt_block: optimized
```
in your YAML, or via the CLI like so: `composer main.py ... model.gpt_block=optimized`.

To disable the optimizations, set
```
model.gpt_block: standard
```

# Expected Performance
We have seen improvements of 5 - 15% for on 1B - 13B parameter models on A100-40GB and A100-80GB nodes. Performance gains may be smaller if your batch size is very small or your model is extremely large. These CUDA extensions do not currently support 30B+ models.

# Credit
Most of these optimizations are adapted or copied from [HazyResearch](https://github.com/HazyResearch/).
