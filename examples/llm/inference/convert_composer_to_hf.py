# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

"""TBD."""

import json
import tempfile
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Dict, Optional, Union

import sentencepiece as spm
import torch
from composer.utils import get_file, safe_torch_load
from transformers import (AutoConfig, AutoTokenizer, PretrainedConfig,
                          PreTrainedTokenizer)

from examples.llm import MosaicGPTConfig


def get_hf_config_from_composer_state_dict(
        state_dict: Dict[str, Any]) -> PretrainedConfig:
    hf_config_dict = state_dict['state']['integrations']['huggingface'][
        'model']['config']['content']

    AutoConfig.register('mosaic_gpt', MosaicGPTConfig)
    return AutoConfig.for_model(**hf_config_dict)


def get_hf_tokenizer_from_comopser_state_dict(
        state_dict: Dict[str, Any]) -> Optional[PreTrainedTokenizer]:
    hf_tokenizer_state = state_dict['state']['integrations']['huggingface'][
        'tokenizer']
    hf_tokenizer = None
    if hf_tokenizer_state != {}:
        with tempfile.TemporaryDirectory() as _tmp_dir:
            for filename, saved_content in hf_tokenizer_state.items():
                tokenizer_file_path = Path(
                    _tmp_dir) / f'{filename}{saved_content["file_extension"]}'
                if saved_content['file_extension'] == '.json':
                    with open(tokenizer_file_path, 'w') as _tmp_file:
                        json.dump(saved_content['content'], _tmp_file)
                elif saved_content['file_extension'] == '.txt':
                    with open(tokenizer_file_path, 'w') as _tmp_file:
                        for line in saved_content['content']:
                            _tmp_file.write(line)
                            _tmp_file.write('\n')
                elif saved_content['file_extension'] == '.model':
                    s = spm.SentencePieceProcessor()
                    s.load_from_serialized_proto(saved_content['content'])
                    with open(tokenizer_file_path, 'wb') as _tmp_file:
                        _tmp_file.write(s.serialized_model_proto())
            hf_tokenizer = AutoTokenizer.from_pretrained(_tmp_dir)

            # we need to set the name_or_path back because otherwise it is the tmp dir we are loading from here
            hf_tokenizer.name_or_path = None
            hf_tokenizer.init_kwargs['name_or_path'] = None

            # # for an unknown reason this key is missing when loading the saved tokenizer, but present with a value of None
            # # for the original tokenizer, so we default it to None
            # hf_tokenizer.init_kwargs['tokenizer_file'] = hf_tokenizer.init_kwargs.get('tokenizer_file', None)

    return hf_tokenizer


def write_huggingface_pretrained_from_composer_checkpoint(
        checkpoint_path: Union[Path, str],
        output_folder: Union[Path, str],
        local_checkpoint_save_location: Optional[Union[Path,
                                                       str]] = None) -> None:
    """Convert a Composer checkpoint to a pretrained HF checkpoint folder.

    Write a ``config.json`` and ``pytorch_model.bin``, like
    :meth:`transformers.PreTrainedModel.from_pretrained` expects, from a
    composer checkpoint.

    .. note:: This function will not work properly if you used surgery algorithms when you trained your model. In that case you will want to
        load the model weights using the Composer :class:`~composer.Trainer` with the ``load_path`` argument.
    .. testsetup::
        import torch
        dataset = RandomTextClassificationDataset(size=16, use_keys=True)
        train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=8)
        eval_dataloader = torch.utils.data.DataLoader(dataset, batch_size=8)
        import transformers
        from composer.models import HuggingFaceModel
        from composer.trainer import Trainer
        hf_model = transformers.AutoModelForSequenceClassification.from_pretrained('prajjwal1/bert-tiny', num_labels=2)
        hf_tokenizer = transformers.AutoTokenizer.from_pretrained('prajjwal1/bert-tiny')
        composer_model = HuggingFaceModel(hf_model, tokenizer=hf_tokenizer, metrics=[], use_logits=True)
        trainer = Trainer(model=composer_model,
                            train_dataloader=train_dataloader,
                            save_filename='composer-hf-checkpoint.pt',
                            max_duration='1ep',
                            save_folder='./')
        trainer.fit()
        trainer.close()

    Example:
    .. testcode::
        from composer.models import write_huggingface_pretrained_from_composer_checkpoint
        write_huggingface_pretrained_from_composer_checkpoint('composer-hf-checkpoint.pt', './hf-save-pretrained-output')
        loaded_model = transformers.AutoModelForSequenceClassification.from_pretrained('./hf-save-pretrained-output')

    Args:
        checkpoint_path (Union[Path, str]): Path to the composer checkpoint, can be a local path, or a remote path beginning with ``s3://``, or another backend
            supported by :meth:`composer.utils.maybe_create_object_store_from_uri`.
        output_folder (Union[Path, str]): Path to the folder to write the output to. Must be a local path.
        local_checkpoint_save_location (Optional[Union[Path, str]], optional): If specified, where to save the checkpoint file to locally.
                                                                                If the input ``checkpoint_path`` is already a local path, this will be a symlink.
                                                                                Defaults to None, which will use a temporary file.
    """
    # default local path to a tempfile if path is not provided
    if local_checkpoint_save_location is None:
        tmp_dir = tempfile.TemporaryDirectory()
        local_checkpoint_save_location = Path(
            tmp_dir.name) / 'local-composer-checkpoint.pt'

    # download the checkpoint file
    get_file(str(checkpoint_path), str(local_checkpoint_save_location))

    # Load the Composer checkpoint state dict
    composer_state_dict = safe_torch_load(local_checkpoint_save_location)

    # Build and save HF Config
    print('Saving HF Model Config...')
    hf_config = get_hf_config_from_composer_state_dict(composer_state_dict)
    hf_config.save_pretrained(output_folder)
    print(hf_config)

    # Extract and save the HF tokenizer
    print('#' * 30)
    print('Saving HF Tokenizer Config...')
    hf_tokenizer = get_hf_tokenizer_from_comopser_state_dict(
        composer_state_dict)
    hf_tokenizer.save_pretrained(output_folder)
    print(hf_tokenizer)

    # # Extract and save the HF model weights
    print('#' * 30)
    print('Saving HF Model Weights...')
    weights_state_dict = composer_state_dict['state']['model']
    weights_state_dict = {
        k: v for k, v in weights_state_dict.items() if k.startswith('model.')
    }
    torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(
        weights_state_dict, prefix='model.')
    torch.save(weights_state_dict, Path(output_folder) / 'pytorch_model.bin')

    print(f'HF checkpoint folder successfully created at {output_folder}.')
    print('#' * 30)


def parse_args() -> Namespace:
    """Parse commandline arguments."""
    parser = ArgumentParser(
        description=
        'Convert Composer checkpoint and Omegaconf model config into a standard HuggingFace checkpoint folder.'
    )
    parser.add_argument('--composer_ckpt', type=str, required=True)
    parser.add_argument('--hf_folder', type=str, required=True)
    parser.add_argument('--local_checkpoint_save_location',
                        type=str,
                        default=None)

    return parser.parse_args()


def main(args: Namespace) -> None:
    """Main: convert a Composer checkpoint to a HuggingFace pretrained model folder

    Args:
        args (Namespace): Commandline arguments.
    """
    write_huggingface_pretrained_from_composer_checkpoint(
        checkpoint_path=args.composer_ckpt,
        output_folder=args.hf_folder,
        local_checkpoint_save_location=args.local_checkpoint_save_location)


if __name__ == '__main__':
    main(parse_args())
