# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

"""Implements a Hugging Prefix LM wrapped inside a :class:`.ComposerModel`."""

from __future__ import annotations

import torch
from accelerate import init_empty_weights
from composer.metrics.nlp import LanguageCrossEntropy, MaskedAccuracy
from omegaconf import DictConfig
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from examples.llm.src.models.hf.model_wrapper import HuggingFaceModelWithZLoss

__all__ = ['ComposerHFPrefixLM']

# HuggingFace hardcodes the ignore index to -100
_HF_IGNORE_INDEX = -100


class ComposerHFPrefixLM(HuggingFaceModelWithZLoss):
    """Configures a :class:`.HuggingFaceModel` around a Prefix LM.

    Note: HuggingFace does not natively support Prefix LM-style models. This function uses
    `transformers.AutoModelForCausalLM` to instantiate a Causal LM, then uses a conversion utility
    to turn the model into a Prefix LM. Currently, that conversion utility only supports the
    following HuggingFace Causal LM types:
        - `GPT2LMHeadModel`
        - `GPTNeoForCausalLM`
        - `GPTNeoXForCausalLM`
        - `GPTJForCausalLM`
        - `BloomForCausalLM`
        - `OPTForCausalLM`

    Args:
        cfg (DictConfig): An omegaconf dictionary used to configure the model:
            cfg.pretrained_model_name_or_path (str): The name of or local path to
                the HF model (e.g., `gpt2` to instantiate a GPT2LMHeadModel). The model
                will be converted to a Prefix LM during initialization.
            cfg.config_overrides (dict, optional): An optional dictionary of keyword
                arguments that override the default configuration associated with
                cfg.pretrained_model_name_or_path. Default: ``{}``.
            cfg.pretrained (bool): Whether to instantiate the model with pre-trained
                weights coming from cfg.pretrained_model_name_or_path. If ``True``,
                cfg.config_overrides must be compatible with the pre-trained weights.
            cfg.init_device ('cpu' | 'meta'): Which device, 'cpu' or 'meta', to
                initialize the model on. Currently, `meta` is only supported when
                cfg.pretrained is ``False``. Default: ``'cpu'``.
            cfg.z_loss (float, optional): The coefficient of the z-loss. If >0.0, this
                the z-loss will be multiplied by this value before being added to the
                standard loss term. Default: ``0.0``.
            cfg.adapt_vocab_for_denoising (bool, optional):  Whether to adapt the vocab
                of the model/tokenizer to include sentinel tokens that are used in denoising
                tasks like Span Corruption. If you intend to load from an existing Composer
                checkpoint that was trained on such a task, set this to ``True`` to ensure
                that the model vocab size matches your checkpoint's vocab size when loading
                the weights. Default: ``False``.
            cfg.add_exact_match (bool, optional): CURRENTLY UNUSED. Whether to add ExactMatch metric used
                in some fine-tuning settings. Default: ``False``.
            cfg.add_rouge (bool, optional): CURRENTLY UNUSED. Whether to add RougeWithDetokenizer metric
                to validation metrics. Default: ``False``.
    """

    def __init__(self, cfg: DictConfig):
        config = AutoConfig.from_pretrained(cfg.pretrained_model_name_or_path,
                                            **cfg.get('config_overrides', {}))

        # Set up the tokenizer (add tokens for denoising sentinels if needed)
        if cfg.get('adapt_vocab_for_denoising', False):
            tokenizer = utils.AutoTokenizerForMOD.from_pretrained(
                cfg.pretrained_model_name_or_path)
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                cfg.pretrained_model_name_or_path)
        vocab_size = len(tokenizer)

        init_device = cfg.get('init_device', 'cpu')
        if init_device == 'cpu':
            if cfg.pretrained:
                model = AutoModelForCausalLM.from_pretrained(
                    cfg.pretrained_model_name_or_path, config=config)
            else:
                model = AutoModelForCausalLM.from_config(config)
        elif init_device == 'meta':
            if cfg.pretrained:
                raise ValueError(
                    'Setting cfg.pretrained=True is not supported when init_device="meta".'
                )
            with init_empty_weights(include_buffers=False):
                model = AutoModelForCausalLM.from_config(config)
        else:
            raise ValueError(
                f'init_device="{init_device}" must be either "cpu" or "meta".')

        # Convert the Causal LM into a Prefix LM via our custom wrapper
        model = utils.convert_hf_causal_lm_to_prefix_lm(model)

        # Expand the embeddings/vocab size to match the tokenizer
        if model.config.vocab_size != vocab_size:
            model.resize_token_embeddings(new_num_tokens=vocab_size)

        metrics = [
            LanguageCrossEntropy(vocab_size=vocab_size,
                                 ignore_index=_HF_IGNORE_INDEX),
            MaskedAccuracy(ignore_index=_HF_IGNORE_INDEX)
        ]

        # if cfg.add_exact_match:
        #     metrics.append(ExactMatch(ignore_index=_HF_IGNORE_INDEX))

        composer_model = super().__init__(model=model,
                                          tokenizer=tokenizer,
                                          metrics=metrics,
                                          z_loss=cfg.get('z_loss', 0.0))

        # if cfg.add_rouge:
        #     rouge_metric = RougeWithDetokenizer(detokenizer=tokenizer)
        #     composer_model.val_metrics[RougeWithDetokenizer.__name__] = rouge_metric

        return composer_model

    def forward(self, batch):
        # Add bidirectional_mask if it is missing and can be constructed
        if 'bidirectional_mask' not in batch:
            if batch.get('mode', None) == 'icl_task':
                batch['bidirectional_mask'] = batch['attention_mask'].clone()
                for i, continuation_indices in enumerate(
                        batch['continuation_indices']):
                    batch['bidirectional_mask'][i, continuation_indices] = 0
            elif 'labels' in batch:
                batch['bidirectional_mask'] = torch.logical_and(
                    torch.equal(batch['attention_mask'], 1),
                    torch.equal(batch['labels'], _HF_IGNORE_INDEX),
                ).type_as(batch['attention_mask'])
            else:
                raise KeyError(
                    'No bidirectional_mask in batch and not sure how to construct one.'
                )
        return super().forward(batch)


# def create_hf_prefix_lm(pretrained_model_name: str = 'gpt2',
#                         tokenizer_name: str = 'gpt2',
#                         use_pretrained: Optional[bool] = False,
#                         model_config: Optional[dict] = None,
#                         gradient_checkpointing: Optional[bool] = False,
#                         z_loss: Optional[float] = 0.0,
#                         task_finetuning: Optional[bool] = False,
#                         generation_eval: bool = False,
#                         adapt_vocab_for_denoising: bool = False):
#     """Prefix-LM model based on |:hugging_face:| Transformers.

#     For more information, see `Transformers <https://huggingface.co/transformers/>`_.

#     Note: HuggingFace does not natively support Prefix-LM-style models. This function uses
#     `transformers.AutoModelForCausalLM` to instantiate a Causal-LM, then uses a conversion utility
#     to turn the model into a Prefix-LM. Currently, that conversion utility only supports the
#     following HuggingFace Causal-LM types:
#         - `GPT2LMHeadModel`
#         - `GPTNeoForCausalLM`
#         - `GPTNeoXForCausalLM`
#         - `GPTJForCausalLM`
#         - `BloomForCausalLM`
#         - `OPTForCausalLM`

#     Args:
#         pretrained_model_name (str): Name of the Hugging Face model to instantiate. Default: ``'gpt2'``.
#         tokenizer_name (str): Tokenizer name used to preprocess the dataset and validate the models inputs. To be compatible
#             with denoising tasks, special sentinel tokens ("<extra_id_0>", "<extra_id_1>", etc.) will be added if you
#             set ``adapt_vocab_size=True``.
#             The model's vocab_size will be hardcoded to match the resulting vocab_size of the tokenizer.
#         use_pretrained (bool, optional): Whether to initialize the model with the pretrained weights. Default: ``False``.
#         model_config (dict, optional): The settings used to create a Hugging Face CausalLM. Used to specify/modify
#             the architecture of the Hugging Face model.
#         gradient_checkpointing (bool, optional): Use gradient checkpointing. Default: ``False``.
#         z_loss (float, optional): Coefficient of `z-loss` term to use during training. Default: ``0.0``.
#         task_finetuning (bool, optional): Whether to add ExactMatch metric used in fine-tuning. Default: ``False``.
#         generation_eval (bool, optional): Whether evaluation requires generation, in which case RougeWithDetokenizer
#             is used for for the validation metric. Default: ``False``.
#         adapt_vocab_for_denoising (bool, optional): Whether to adapt the vocab of the model/tokenizer to include
#             sentinel tokens that are used in denoising tasks like Span Corruption. If you intend to load from an
#             existing Composer checkpoint that was trained on such a task, set this to ``True`` to ensure that the model
#             vocab size matches your checkpoint's vocab size when loading the weights. Default: ``False``.

#     To create a |:hugging_face:| Prefix-LM model for UL2 pretraining:

#     .. testcode::

#         from examples.ul2.src.hf_prefix_lm import create_prefix_lm
#         model = create_prefix_lm('gpt2')
#     """
#     try:
#         import transformers
#     except ImportError as e:
#         raise MissingConditionalImportError(extra_deps_group='nlp',
#                                             conda_package='transformers') from e

#     # Set up the tokenizer (add tokens for denoising sentinels if needed)
#     if adapt_vocab_for_denoising:
#         tokenizer = utils.AutoTokenizerForMOD.from_pretrained(tokenizer_name)
#     else:
#         tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
#     vocab_size = len(tokenizer)

#     if not model_config:
#         model_config = {}

#     if not pretrained_model_name:
#         pretrained_model_name = 'gpt2'

#     if use_pretrained:
#         assert transformers.AutoModelForCausalLM.from_pretrained is not None, 'AutoModelForCausalLM has from_pretrained method'
#         model = transformers.AutoModelForCausalLM.from_pretrained(
#             pretrained_model_name_or_path=pretrained_model_name, **model_config)
#     else:
#         config = transformers.AutoConfig.from_pretrained(
#             pretrained_model_name, **model_config)
#         assert transformers.AutoModelForCausalLM.from_config is not None, 'AutoModelForCausalLM has from_config method'
#         model = transformers.AutoModelForCausalLM.from_config(config)

#     # Convert the Causal LM into a Prefix LM via our custom wrapper
#     model = utils.convert_hf_causal_lm_to_prefix_lm(model)

#     # Expand the embeddings/vocab size to match the tokenizer
#     if model.config.vocab_size != vocab_size:
#         model.resize_token_embeddings(new_num_tokens=vocab_size)

#     # Super charge
#     prepare_hf_causal_lm_model_for_fsdp(model)

#     if gradient_checkpointing:
#         model.gradient_checkpointing_enable()  # type: ignore

#     metrics = [
#         LanguageCrossEntropy(ignore_index=_HF_IGNORE_INDEX,
#                              vocab_size=model.config.vocab_size),
#         MaskedAccuracy(ignore_index=_HF_IGNORE_INDEX)
#     ]
#     if task_finetuning:
#         metrics.append(ExactMatch(ignore_index=_HF_IGNORE_INDEX))
#     model = HuggingFaceModelWithZLoss(model=model,
#                                       tokenizer=tokenizer,
#                                       metrics=metrics,
#                                       z_loss=z_loss)
#     if generation_eval:
#         model.val_metrics = {
#             RougeWithDetokenizer.__name__:
#                 RougeWithDetokenizer(detokenizer=tokenizer, rouge_keys='rougeL')
#         }
#     return model

# class HuggingFaceModelWithZLoss(HuggingFaceModel):

#     def __init__(self, model, tokenizer, metrics, z_loss=0.0):
#         super().__init__(model=model,
#                          tokenizer=tokenizer,
#                          metrics=metrics,
#                          use_logits=True)
#         self.z_loss = float(z_loss)
#         assert self.z_loss >= 0.0

#         self.pad_token_id = self.model.config.pad_token_id
#         if self.pad_token_id is None:
#             self.pad_token_id = self.model.config.eos_token_id

#         self.model_forward_args = inspect.getfullargspec(
#             self.model.forward).args

#     def forward(self, batch):
#         if 'bidirectional_mask' not in batch:
#             if batch.get('mode', None) == 'icl_task':
#                 batch['bidirectional_mask'] = batch['attention_mask'].clone()
#                 for i, continuation_indices in enumerate(
#                         batch['continuation_indices']):
#                     batch['bidirectional_mask'][i, continuation_indices] = 0
#             elif 'labels' in batch:
#                 batch['bidirectional_mask'] = torch.logical_and(
#                     torch.equal(batch['attention_mask'], 1),
#                     torch.equal(batch['labels'], _HF_IGNORE_INDEX),
#                 ).type_as(batch['attention_mask'])
#             else:
#                 raise KeyError(
#                     'No bidirectional_mask in batch and not sure how to construct one.'
#                 )
#         if isinstance(batch, dict) or isinstance(batch, UserDict):
#             # Further input validation is left to the huggingface forward call
#             batch = {
#                 k: v for k, v in batch.items() if k in self.model_forward_args
#             }
#             output = self.model(**batch)  # type: ignore (thirdparty)
#         else:
#             raise ValueError(
#                 'Unexpected batch type. Expected a dictionary with keys corresponding to the inputs to the forward function of the Huggingface model'
#             )
#         return output

#     def loss(self, outputs, batch):
#         if self.config.use_return_dict:
#             loss, logits = outputs['loss'], outputs['logits']
#         else:
#             # loss is at index 0 in the output tuple, logits are at index 1
#             loss, logits = outputs[:2]
#         if self.z_loss == 0.0:
#             return loss

#         # Shift the labels since this is for decoder-only models
#         labels_shifted = torch.full_like(batch['labels'], _HF_IGNORE_INDEX)
#         labels_shifted[..., :-1] = batch['labels'][..., 1:]
#         labels_flat = labels_shifted.contiguous().view(-1)
#         # Add a z_loss to the standard loss
#         logits_flat = logits.view(-1, logits.size(-1))
#         log_z = torch.logsumexp(logits_flat[labels_flat != _HF_IGNORE_INDEX],
#                                 dim=1)
#         log_z2 = log_z**2
#         z_loss = log_z2.mean() * self.z_loss
#         if self.config.use_return_dict:
#             outputs['loss'] += z_loss
#             return outputs['loss']
#         else:
#             outputs[0] += z_loss
#             return outputs[0]

#     def eval_forward(self, batch, outputs: Optional[Any] = None):
#         if 'generate_output' in batch:
#             self.labels = batch.pop('labels')
#             output_tokens = self.model.generate(
#                 batch['input_ids'],
#                 attention_mask=batch['attention_mask'],
#                 max_new_tokens=512,
#                 do_sample=True,
#                 top_p=0.90,
#                 top_k=0,
#                 no_repeat_ngram_size=3,
#                 pad_token_id=self.pad_token_id,
#             )
#             # Remove the input_ids from output_tokens
#             inp_seq_len = batch['input_ids'].shape[1]
#             outputs = output_tokens[:, inp_seq_len:]
#             return outputs

#         return super().eval_forward(batch, outputs)
