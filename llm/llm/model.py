from .gpt import ComposerGPT


def build_composer_model(cfg):
    if cfg.name == 'gpt':
        return ComposerGPT(cfg)
    else:
        raise ValueError(f'Not sure how to build model with name={name}')