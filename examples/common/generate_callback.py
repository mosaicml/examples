# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Monitor rate of change of loss."""
from typing import List

import wandb
from composer.core import Callback, State
from composer.loggers import Logger, WandBLogger
from composer.utils import dist, ensure_tuple


class Generate(Callback):

    def __init__(self, prompts: List[str], **kwargs):
        self.prompts = prompts
        self.generate_kwargs = kwargs

    def eval_end(self, state: State, logger: Logger):
        model = state.model
        tokenizer = state.model.tokenizer
        device = state.device

        # stash the original original value of padding_side because generation requires left padding
        original_padding_side = tokenizer.padding_side
        tokenizer.padding_side = 'left'
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenized_input = tokenizer(self.prompts,
                                    return_tensors='pt',
                                    padding=True)

        for k, v in tokenized_input.items():
            tokenized_input[k] = device.tensor_to_device(v)

        output_token_ids = model.model.generate(
            input_ids=tokenized_input['input_ids'],
            attention_mask=tokenized_input['attention_mask'],
            synced_gpus=True,
            **self.generate_kwargs,
        )

        if dist.get_global_rank() == 0:
            wandb_artifact = wandb.Artifact('generate_samples_' +
                                            str(wandb.run.id),
                                            type='predictions')
            for destination in ensure_tuple(logger.destinations):
                if isinstance(destination, WandBLogger):
                    rows = []
                    for i in range(len(self.prompts)):
                        prompt = self.prompts[i]
                        output_tokens = output_token_ids[i][
                            tokenized_input['input_ids'].shape[1]:]
                        output_text = tokenizer.decode(output_tokens,
                                                       skip_special_tokens=True)

                        rows.append([prompt, output_text])

                    text_table = wandb.Table(data=rows,
                                             columns=['prompt', 'generation'])
                    wandb_artifact.add(text_table, 'predictions')
                    wandb.log_artifact(wandb_artifact)

        tokenizer.padding_side = original_padding_side
