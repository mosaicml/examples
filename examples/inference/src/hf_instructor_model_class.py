from InstructorEmbedding import INSTRUCTOR
import torch
import numpy as np
import base64
from typing import Dict, Any
from utils import BaseModelHandler

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

    def predict(self, **inputs: Dict[str, Any]):
        if "instruction" not in inputs:
            raise KeyError("Instruction key not in inputs")

        if "sentence" not in inputs:
            raise KeyError("Sentence key not in inputs")

        embeddings = self.model.encode([[inputs["instruction"], inputs["sentence"]]])
        np_out = np.asarray(embeddings, dtype=np.float32)
        output_bytes = bytes(base64.b64encode(np_out).decode('utf8'),
                                encoding='utf8')
        return output_bytes
