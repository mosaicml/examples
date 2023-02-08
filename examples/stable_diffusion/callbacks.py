# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

"""Prompt and image visualization callback for diffusion models."""

from composer import Callback, Logger, State
from composer.loggers import WandBLogger
from composer.utils import ensure_tuple
from torchvision.utils import make_grid


class LogDiffusionImages(Callback):
    """Logs eval prompts and generated images to a Weights and Biases table at
    the end of an evaluation batch.

    Requires Weights and Biases to be installed and setup.
    """
    def eval_batch_end(self, state: State, logger: Logger):
        prompts = state.batch  # batch_size
        outputs = state.outputs.cpu()  # Tensor of shape [len(prompts) * num_images_per_prompt, 3, 512, 512]
        num_images_per_prompt = state.model.module.num_images_per_prompt
        for destination in ensure_tuple(logger.destinations):
            if isinstance(destination, WandBLogger):
                if num_images_per_prompt > 1: 
                    outputs = [make_grid(out, nrow=num_images_per_prompt) for out in outputs.chunk(num_images_per_prompt)]
                    
                for prompt, output in zip(prompts, outputs):
                    destination.log_images(images=output, name=prompt, step=state.timestamp.batch.value)
                    
                # # 1 prompt 1 image
                # if len(prompts) == 1 and num_images_per_prompt == 1:
                #     destination.log_images(images=output[0],
                #                            name=prompt[0],
                #                            step=state.timestamp.batch.value)

                # # multiple prompts, 1 image per prompt
                # if len(prompts) > 1 and num_images_per_prompt == 1:
                #     for prompt, output in zip(prompts, outputs):
                #         destination.log_images(images=output,
                #                                name=prompt,
                #                                step=state.timestamp.batch.value)

                # # one prompt multiple images
                # if len(prompts) == 1 and num_images_per_prompt > 1:
                #     outputs = make_grid(outputs, nrow=num_images_per_prompt)
                #     destination.log_images(images=outputs,
                #                            name=prompts[0],
                #                            step=state.timestamp.batch.value)

                # # multiple prompts multiple images
                # if len(prompts) > 1 and num_images_per_prompt > 1:
                #     outputs = [
                #         make_grid(out, nrow=num_images_per_prompt)
                #         for out in outputs.chunk(num_images_per_prompt)
                #     ]
                #     for prompt, output in zip(prompts, outputs):
                #         destination.log_images(images=outputs,
                #                                name=prompt,
                #                                step=state.timestamp.batch.value)
