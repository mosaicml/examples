import warnings

from .mosaic_gpt import ComposerMosaicGPT


def build_composer_model(cfg):
    warnings.filterwarnings(action='ignore', message='Torchmetrics v0.9 introduced a new argument class property')

    if cfg.name == 'mosaic_gpt':
        return ComposerMosaicGPT(cfg)
    else:
        raise ValueError(f'Not sure how to build model with name={cfg.name}')
