
from src.mosaic_gpt import ComposerMosaicGPT
from src.hf_gpt import ComposerHFCausalLM


MODEL_REGISTRY = {
    "mosaic_gpt": lambda cfg, device: ComposerMosaicGPT(cfg),
    "hf_causal_lm": lambda cfg, device: ComposerHFCausalLM(cfg)
}

