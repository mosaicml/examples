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

    INPUT_STRINGS_KEY = 'input_strings'

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

    def _parse_inputs(self, inputs: Dict[str, Any]):
        if 'input_strings' not in inputs:
            raise RuntimeError(
                'Input strings must be provided as a list to generate call')

        generate_input = inputs[self.INPUT_STRINGS_KEY]

        # Set default generate kwargs
        generate_kwargs = copy.deepcopy(self.DEFAULT_GENERATE_KWARGS)
        generate_kwargs['eos_token_id'] = self.tokenizer.eos_token_id

        # If request contains any additional kwargs, add them to generate_kwargs
        for k, v in inputs.items():
            if k not in [self.INPUT_STRINGS_KEY]:
                generate_kwargs[k] = v

        return generate_input, generate_kwargs

    def _extract_output(self, outputs: List[Any]):
        output_list = []
        for output in outputs:
            output_bytes = output[0]['generated_text']
            output_list.append(output_bytes)
        return output_list

    def predict(self, input_dicts: List[Dict[str, Any]]):
        generate_inputs = []
        generate_kwargs = {}
        for input_dict in input_dicts:            
            generate_input_list, generate_kwarg = self._parse_inputs(
                input_dict
            )
            # Flatten the list of inputs into a single list
            # 2 cases for batching 
            # 1. Multiple requests single input string
            # 2. Single request multiple input strings
            generate_inputs += generate_input_list

            for k, v in generate_kwarg.items(): 
                if k in generate_kwargs and generate_kwargs[k] != v:
                    raise RuntimeError(f"Request has conflicting values for kwarg {k}")
                generate_kwargs[k] = v 


        print("Logging input to generate: ", generate_inputs)
        outputs = self.generator(generate_inputs, **generate_kwargs)
        return self._extract_output(outputs)

    def predict_stream(self, **inputs: Dict[str, Any]):
        generate_input, generate_kwargs = self._parse_inputs(inputs)

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
