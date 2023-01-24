# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0
"""Prompt and image visualization callback for diffusion models"""

from composer import Callback, Logger, State
from torchvision.utils import make_grid
from composer.utils import ensure_tuple
from composer.loggers import WandBLogger


class LogDiffusionImages(Callback):
    """Logs eval prompts and generated images to a w&b table after every batch, logs all generated images at the end of evaluation.
    requires weights and biases.
    """
    def eval_batch_end(self, state: State, logger: Logger):
        prompts = state.batch
        outputs = state.outputs.cpu()
        num_images_per_prompt = state.model.module.num_images_per_prompt
        for destination in ensure_tuple(logger.destinations):
            if isinstance(destination, WandBLogger):
                if num_images_per_prompt > 1:
                    outputs = [make_grid(out, nrow=1) for out in outputs.chunk(num_images_per_prompt)]
                for prompt, output in zip(prompts, outputs):
                    destination.log_images(images=output, name=prompt, step=state.timestamp.batch.value)    