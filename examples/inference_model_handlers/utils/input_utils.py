import copy
from typing import Any, Dict


def parse_generate_inputs(
        inputs: Dict[str, Any],
        input_strings_key: str,
        default_generate_kwargs: Dict[str, Any]
):
    if input_strings_key not in inputs:
        raise RuntimeError(
            "Input strings must be provided as a list to generate call")

    generate_input = inputs[input_strings_key]

    # Set default generate kwargs
    generate_kwargs = copy.deepcopy(default_generate_kwargs)

    # If request contains any additional kwargs, add them to generate_kwargs
    for k, v in inputs.items():
        if k not in [input_strings_key]:
            generate_kwargs[k] = v

    return generate_input, generate_kwargs
