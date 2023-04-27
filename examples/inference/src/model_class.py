import copy
import warnings
from contextlib import nullcontext
from threading import Thread
from typing import Optional

import torch
from examples.llm.src.models.mosaic_gpt.configuration_mosaic_gpt import \
    MosaicGPTConfig
from examples.llm.src.models.mosaic_gpt.mosaic_gpt import MosaicGPT
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          TextGenerationPipeline, TextIteratorStreamer)
from huggingface_hub import snapshot_download
from utils import get_torch_dtype


class BaseModel:

    def __init__(self) -> None:
        self.device = torch.cuda.current_device()
        self.setup()
        print(f"DSInferenceModel initialized with device: {self.device}")

    def setup(self):
        raise NotImplementedError

    def predict(self, **inputs):
        raise NotImplementedError

    def predict_stream(self, **inputs):
        raise NotImplementedError


class HFModel(BaseModel):

    DEFAULT_GENERATE_KWARGS = {
        'max_length': 256,
        'use_cache': True,
        'do_sample': True,
        'top_p': 0.95,
        'temperature': 0.8,
    }

    INPUT_STRINGS_KEY = 'input_strings'

    def __init__(self,
                 model_name_or_path: str,
                 dtype: str = 'fp16',
                 autocast_dtype: Optional[str] = None):
        self.model_name_or_path = model_name_or_path
        self.dtype = dtype
        self.autocast_dtype = autocast_dtype
        super().__init__()

    def _register_models(self):
        AutoConfig.register('mosaic_gpt', MosaicGPTConfig)
        AutoModelForCausalLM.register(MosaicGPTConfig, MosaicGPT)

    def setup(self):
        self._register_models()
        pretrained_model_name_or_path = self.model_name_or_path
        print(f"Downloading HF model with name: {pretrained_model_name_or_path}")
        snapshot_download(repo_id=pretrained_model_name_or_path)
        print(f"Loading HF Model with name: {pretrained_model_name_or_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path)
        if tokenizer.pad_token_id is None:
            warnings.warn(
                'pad_token_id is not set for the tokenizer. Using eos_token_id as pad_token_id.'
            )
            tokenizer.pad_token_id = tokenizer.eos_token_id
            tokenizer.padding_side = 'left'
        self.tokenizer = tokenizer

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path,
            torch_dtype=get_torch_dtype(self.dtype))
        self.model = model.eval()

    def _parse_inputs(self, inputs):
        if 'input_strings' not in inputs:
            raise RuntimeError(
                "Input strings must be provided as a list to generate call")

        generate_input = inputs[self.INPUT_STRINGS_KEY]

        # Set default generate kwargs
        generate_kwargs = copy.deepcopy(self.DEFAULT_GENERATE_KWARGS)
        generate_kwargs['eos_token_id'] = self.tokenizer.eos_token_id

        # If request contains any additional kwargs, add them to generate_kwargs
        for k, v in inputs.items():
            if k not in [self.INPUT_STRINGS_KEY]:
                generate_kwargs[k] = v

        return generate_input, generate_kwargs

    def predict(self, **inputs):
        print("Logging input to generate: ", inputs)
        generate_input, generate_kwargs = self._parse_inputs(inputs)

        self.generator = TextGenerationPipeline(task='text-generation',
                                                model=self.model,
                                                tokenizer=self.tokenizer,
                                                device=self.device)

        with torch.cuda.amp.autocast(
                enabled=True, dtype=get_torch_dtype(self.autocast_dtype)
        ) if self.autocast_dtype else nullcontext():
            outputs = self.generator(generate_input, **generate_kwargs)

        return [
            bytes(output[0]['generated_text'], encoding='utf8')
            for output in outputs
        ]

    def predict_stream(self, **inputs):
        print("Logging input to generate streaming: ", inputs)
        generate_input, generate_kwargs = self._parse_inputs(inputs)

        # TextGenerationPipeline passes streamer to generate as a kwarg
        streamer = TextIteratorStreamer(self.tokenizer)
        generate_kwargs["streamer"] = streamer
        self.generator = TextGenerationPipeline(task='text-generation',
                                                model=self.model,
                                                tokenizer=self.tokenizer,
                                                device=self.device)
        thread = Thread(target=self.generator,
                        args=(generate_input),
                        kwargs=generate_kwargs)
        thread.start()

        for new_text in streamer:
            yield new_text

        thread.join()
