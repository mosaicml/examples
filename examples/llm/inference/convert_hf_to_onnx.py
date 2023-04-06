# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
from pathlib import Path
from typing import Optional

import torch
from composer.utils import (maybe_create_object_store_from_uri, parse_uri,
                            reproducibility)
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from examples.llm import MosaicGPT, MosaicGPTConfig


def gen_random_batch(batch_size: int, vocab_size: int, max_seq_len: int):
    # generate input batch of random data
    batch = {
        'input_ids':
            torch.randint(
                low=0,
                high=vocab_size,
                size=(batch_size, max_seq_len),
                dtype=torch.int64,
            ),
        'attention_mask':
            torch.ones(size=(batch_size, max_seq_len), dtype=torch.bool)
    }
    return batch


def export_to_onnx(
    pretrained_model_name_or_path: str,
    output_folder: str,
    export_batch_size: int,
    max_seq_len: Optional[int],
    verify_export: bool,
):
    reproducibility.seed_all(42)
    save_object_store = maybe_create_object_store_from_uri(output_folder)
    _, _, parsed_save_path = parse_uri(output_folder)

    AutoConfig.register('mosaic_gpt', MosaicGPTConfig)
    AutoModelForCausalLM.register(MosaicGPTConfig, MosaicGPT)

    print('Loading HF config/model/tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
    config = AutoConfig.from_pretrained(pretrained_model_name_or_path,
                                        attn_impl='torch')
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path,
                                                 config=config)
    model.eval()

    if max_seq_len is None and not hasattr(model.config, 'max_seq_len'):
        raise ValueError(
            'max_seq_len must be specified in either the model config or as an argument to this function.'
        )
    elif max_seq_len is None:
        max_seq_len = model.config.max_seq_len

    print('Creating random batch...')
    sample_input = gen_random_batch(
        export_batch_size,
        len(tokenizer),
        max_seq_len,
    )

    with torch.no_grad():
        model(**sample_input)

    output_file = Path(parsed_save_path) / 'model.onnx'
    os.makedirs(parsed_save_path, exist_ok=True)
    print('Exporting the model with ONNX...')
    torch.onnx.export(
        model,
        sample_input,
        str(output_file),
        input_names=['input_ids', 'attention_mask'],
        output_names=['output'],
        opset_version=16,
    )

    if verify_export:
        with torch.no_grad():
            orig_out = model(**sample_input)

        import onnx  # type: ignore
        import onnx.checker  # type: ignore
        import onnxruntime as ort  # type: ignore

        _ = onnx.load(str(output_file))

        onnx.checker.check_model(str(output_file))

        ort_session = ort.InferenceSession(str(output_file))

        for key, value in sample_input.items():
            sample_input[key] = value.cpu().numpy()

        loaded_model_out = ort_session.run(None, sample_input)

        torch.testing.assert_close(
            orig_out.logits.detach().numpy(),
            loaded_model_out[0],
            rtol=1e-2,
            atol=1e-2,
            msg=f'output mismatch between the orig and onnx exported model',
        )
        print('exported model ouptut matches with unexported model!!')

    if save_object_store is not None:
        save_object_store.upload_object(output_file, output_file)


def parse_args():
    parser = argparse.ArgumentParser(description='Convert HF model to ONNX',)
    parser.add_argument(
        '--pretrained_model_name_or_path',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--output_folder',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--export_batch_size',
        type=int,
        default=8,
    )
    parser.add_argument(
        '--max_seq_len',
        type=int,
        default=None,
    )
    parser.add_argument(
        '--verify_export',
        action='store_true',
    )
    return parser.parse_args()


def main(args: argparse.Namespace):
    export_to_onnx(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        output_folder=args.output_folder,
        export_batch_size=args.export_batch_size,
        max_seq_len=args.max_seq_len,
        verify_export=args.verify_export,
    )


if __name__ == '__main__':
    main(parse_args())
