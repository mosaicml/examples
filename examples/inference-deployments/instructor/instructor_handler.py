# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Dict, List

import torch
from InstructorEmbedding import INSTRUCTOR


class HFInstructorHandler():
    """Custom Hugging Face Instructor Model handler class."""

    MODEL_NAME = ''

    def __init__(self, model_name: str):
        self.device = torch.cuda.current_device()
        self.model_name = model_name

        self.model = INSTRUCTOR(self.model_name)
        self.model.to(self.device)

    def predict(self, inputs: List[Dict[str, Any]]):
        """Runs a forward pass with the given inputs.

        Input format: {"input_strings": ["<instruction>", "<sentence>"]}
        """
        input_list = []
        for input in inputs:
            if 'input_strings' not in input:
                raise KeyError('input_strings key not in inputs')
            input_list.append(input['input_strings'])

        embeddings = self.model.encode(input_list)
        return embeddings.tolist()
