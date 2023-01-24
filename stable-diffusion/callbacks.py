# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0
"""Prompt and image visualization callback for diffusion models"""

from composer import Callback, Logger, State
from torchvision.utils import make_grid
import wandb


class LogDiffusionImages(Callback):
    """Logs eval prompts and generated images to a w&b table after every batch, logs all generated images at the end of evaluation.
    requires weights and biases.
    """
    def __init__(self):
        self.table = None

    def eval_start(self, state: State, logger: Logger) -> None:
        self.table = wandb.Table(columns=["prompt", "images"])

    def eval_batch_end(self, state: State, logger: Logger):
        prompts = state.batch_get_item(key=0)
        outputs = state.outputs.cpu()
        num_images_per_prompt = state.model.module.num_images_per_prompt
        if num_images_per_prompt > 1:
            outputs = [make_grid(out, nrow=1) for out in outputs.chunk(num_images_per_prompt)]
            for prompt, output in zip(prompts, outputs):
                self.table.add_data(prompt, output)

    def eval_end(self, state: State, logger: Logger) -> None:
        step = state.timestamp.batch.value
        logger.log_metrics(self.table, step)
         