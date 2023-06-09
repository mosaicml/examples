# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

import copy
from threading import Thread
from typing import Any, Dict, List

import torch
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          TextIteratorStreamer, pipeline)


class MPTModelHandler():

    DEFAULT_GENERATE_KWARGS = {
        'max_length': 256,
        'use_cache': True,
        'do_sample': True,
        'top_p': 0.95,
        'temperature': 0.8,
    }

    INPUT_KEY = 'input'
    PARAMETERS_KEY = 'parameters'

    def __init__(self, model_name: str):
        self.device = torch.cuda.current_device()
        self.model_name = model_name

        config = AutoConfig.from_pretrained(self.model_name,
                                            trust_remote_code=True)
        config.attn_config['attn_impl'] = 'torch'

        model = AutoModelForCausalLM.from_pretrained(self.model_name,
                                                     config=config,
                                                     torch_dtype=torch.bfloat16,
                                                     trust_remote_code=True)

        tokenizer = AutoTokenizer.from_pretrained(self.model_name,
                                                  trust_remote_code=True)
        self.tokenizer = tokenizer

        model = model.eval()
        self.generator = pipeline(task='text-generation',
                                  model=model,
                                  tokenizer=tokenizer,
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
        output_list = []
        for output in outputs:
            output_bytes = output[0]['generated_text']
            output_list.append(output_bytes)
        return output_list

    def predict(self, model_requests: List[Dict[str, Any]]):
        """Runs forward pass with the given inputs.

        model_requests: List of dictionaries that contain forward pass inputs as well
        as other parameters, such as generate kwargs.

        ex. [{'input': 'hello world!', 'parameters': {'max_length': 10}]
        """
        generate_inputs = []
        generate_kwargs = {}
        # Currently assumes the same generate_kwargs for the entire batch.
        for req in model_requests:
            generate_input, generate_kwargs = self._parse_model_request(req)
            generate_inputs += generate_input

        print('Logging input to generate: ', generate_inputs)
        outputs = self.generator(generate_inputs, **generate_kwargs)
        return self._extract_output(outputs)

    def predict_stream(self, **inputs: Dict[str, Any]):
        generate_input, generate_kwargs = self._parse_model_request(inputs)

        # TextGenerationPipeline passes streamer to generate as a kwarg
        streamer = TextIteratorStreamer(self.tokenizer)
        generate_kwargs['streamer'] = streamer

        thread = Thread(target=self.generator,
                        args=(generate_input),
                        kwargs=generate_kwargs)
        thread.start()

        for new_text in streamer:
            yield new_text

        thread.join()
