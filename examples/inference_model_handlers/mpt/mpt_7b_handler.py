import copy
from threading import Thread
from typing import Any, Dict

import torch
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          TextIteratorStreamer, pipeline)
from examples.inference_model_handlers.utils.input_utils import parse_generate_inputs

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

        config = AutoConfig.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        config.attn_config['attn_impl'] = 'triton'

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            config=config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True)
        self.tokenizer = tokenizer

        model = model.eval()
        self.generator = pipeline(task='text-generation',
                                  model=model,
                                  tokenizer=tokenizer,
                                  device=self.device)

    def predict(self, **inputs: Dict[str, Any]):
        generate_input, generate_kwargs = parse_generate_inputs(
            inputs, self.INPUT_STRINGS_KEY, self.DEFAULT_GENERATE_KWARGS)
        outputs = self.generator(generate_input, **generate_kwargs)
        return [output[0]['generated_text'] for output in outputs]

    def predict_stream(self, **inputs: Dict[str, Any]):
        generate_input, generate_kwargs = self._parse_inputs(inputs)

        # TextGenerationPipeline passes streamer to generate as a kwarg
        streamer = TextIteratorStreamer(self.tokenizer)
        generate_kwargs["streamer"] = streamer

        thread = Thread(target=self.generator,
                        args=(generate_input),
                        kwargs=generate_kwargs)
        thread.start()

        for new_text in streamer:
            yield new_text

        thread.join()
