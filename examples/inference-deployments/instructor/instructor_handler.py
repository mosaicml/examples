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

    def predict(self, model_requests: List[Dict[str, Any]]):
        """Runs a forward pass with the given inputs.

        Model request format: {"input": ["<instruction>", "<sentence>"]}
        """
        input_list = []
        for req in model_requests:
            if 'input' not in req:
                raise KeyError('input key not in model_requests')
            input_list.append(req['input'])

        embeddings = self.model.encode(input_list)
        return embeddings.tolist()
