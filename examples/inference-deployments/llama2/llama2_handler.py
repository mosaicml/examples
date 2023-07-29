import copy
import warnings
from contextlib import nullcontext
from enum import Enum
from threading import Thread
from typing import Any, Dict, List
import os

import deepspeed
import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          LlamaForCausalLM, LlamaConfig, LlamaTokenizer,
                          TextIteratorStreamer, pipeline)

class Llama2ModelHandler:

    DEFAULT_GENERATE_KWARGS = {
        'max_length': 256,
        'use_cache': True,
        'do_sample': True,
        'top_p': 0.95,
        'temperature': 0.8,
    }

    INPUT_KEY = 'input'
    PARAMETERS_KEY = 'parameters'
    MODEL_DTYPE = 'fp16'

    def __init__(
        self,
        model_name_or_path: str,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.setup()

    def setup(self):
        print(f"Loading Llama2 Model with name: {self.model_name_or_path}")
        hf_model_name = self.model_name_or_path

        dtype = torch.float16
        world_size = int(os.getenv("WORLD_SIZE", "1"))

        inf_config = {
            "dtype": dtype,
            "tensor_parallel": {
                "tp_size": world_size
            }
        }

        with deepspeed.OnDevice(dtype=dtype, device='meta'):
            model = LlamaForCausalLM.from_pretrained(hf_model_name, low_cpu_mem_usage=True)
            hf_config = LlamaConfig.from_pretrained(hf_model_name, low_cpu_mem_usage=True)
        
            print("Loaded model!")

            model.eval()
        
            self.tokenizer = LlamaTokenizer.from_pretrained(hf_model_name, low_cpu_mem_usage=True)
        
            # Deepspeed's init_inference takes in a huggingface model, which is the .model
            # object of our ComposerModel object.
            ds_engine = deepspeed.init_inference(model, config=inf_config)
            ds_model = ds_engine.module

        # For some reason, we have to set device after the the deepspeed.onDevice block
        self.device = torch.cuda.current_device()
        self.generator = pipeline(task='text-generation',
                                  model=ds_model,
                                  tokenizer=self.tokenizer,
                                  device=self.device)

    def _parse_model_request(self, model_request: Dict[str, Any]):
        if self.INPUT_KEY not in model_request:
            raise RuntimeError(
                f"{self.INPUT_KEY} must be provided to generate call")

        generate_input = model_request[self.INPUT_KEY]

        # Set default generate kwargs
        generate_kwargs = copy.deepcopy(self.DEFAULT_GENERATE_KWARGS)
        generate_kwargs['eos_token_id'] = self.tokenizer.eos_token_id

        # If request contains any additional kwargs, add them to generate_kwargs
        for k, v in model_request.get(self.PARAMETERS_KEY, {}).items():
            generate_kwargs[k] = v

        return generate_input, generate_kwargs

    def _extract_output(self, outputs: List[Any]):
        output_bytes_list = []
        for output in outputs:
            output_item = output[0]['generated_text']
            output_bytes_list.append(output_item)
        return output_bytes_list

    def predict(self, model_requests: List[Dict[str, Any]]):
        """
        model_requests: List of dictionaries that contain forward pass inputs as well
          as other parameters, such as generate kwargs.

          ex. [{'input': 'hello world!', 'parameters': {'max_length': 10}}]
        """
        generate_inputs = []
        # Note: this assumes the same generate_kwargs for the entire batch.
        for req in model_requests:
            generate_input, generate_kwargs = self._parse_model_request(req)
            generate_inputs += [generate_input]

        outputs = self.generator(generate_inputs, **generate_kwargs)

        return self._extract_output(outputs)

    def predict_stream(self, **inputs):
        generate_input, generate_kwargs = self._parse_model_request(inputs)

        # TextGenerationPipeline passes streamer to generate as a kwarg
        streamer = TextIteratorStreamer(self.tokenizer)
        generate_kwargs["streamer"] = streamer

        thread = Thread(target=self.generator,
                        args=(generate_input,),
                        kwargs=generate_kwargs)
        thread.start()

        for new_text in streamer:
            yield new_text

        thread.join()