from InstructorEmbedding import INSTRUCTOR
import torch
from typing import Dict, Any

class BaseModelHandler:

    def __init__(self) -> None:
        self.device = torch.cuda.current_device()
        self.setup()
        print(f"DSInferenceModel initialized with device: {self.device}")

    def setup(self):
        raise NotImplementedError

    def predict(self, **inputs: Dict[str, Any]):
        raise NotImplementedError

    def predict_stream(self, **inputs: Dict[str, Any]):
        raise NotImplementedError


class HFInstructorLargeModel(BaseModelHandler):
    MODEL_NAME = 'hkunlp/instructor-large'

    def setup(self):
        self.model = INSTRUCTOR(self.MODEL_NAME)

    def predict(self, **inputs):
        # input_strings in format of [[<instruction>, <sentence>]]
        if "input_strings" not in inputs:
            raise KeyError("input_strings key not in inputs")

        embeddings = self.model.encode(inputs["input_strings"])
        return embeddings.tolist()
