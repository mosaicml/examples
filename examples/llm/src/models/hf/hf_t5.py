# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Implements a Hugging Face T5 wrapped inside a :class:`.ComposerModel`."""

from __future__ import annotations

from accelerate import init_empty_weights
from composer.metrics.nlp import LanguageCrossEntropy, MaskedAccuracy
from omegaconf import DictConfig
from transformers import AutoConfig, AutoTokenizer, T5ForConditionalGeneration

from examples.llm.src.models.hf.model_wrapper import HuggingFaceModelWithZLoss

__all__ = ['ComposerHFT5']

# HuggingFace hardcodes the ignore index to -100
_HF_IGNORE_INDEX = -100


class ComposerHFT5(HuggingFaceModelWithZLoss):
    """Configures a :class:`.HuggingFaceModel` around a T5.

    Note: This function uses `transformers.T5ForConditionalGenration`. Future releases
        will expand support to more general classes of HF Encoder-Decoder models.

    Args:
        cfg (DictConfig): An omegaconf dictionary used to configure the model:
            cfg.pretrained_model_name_or_path (str): The name of or local path to
                the HF model (e.g., `t5-base` to instantiate a T5 using the base config).
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
            cfg.add_exact_match (bool, optional): CURRENTLY UNUSED. Whether to add ExactMatch metric used
                in some fine-tuning settings. Default: ``False``.
            cfg.add_rouge (bool, optional): CURRENTLY UNUSED. Whether to add RougeWithDetokenizer metric
                to validation metrics. Default: ``False``.
    """

    def __init__(self, cfg: DictConfig):
        config = AutoConfig.from_pretrained(cfg.pretrained_model_name_or_path,
                                            **cfg.get('config_overrides', {}))

        tokenizer = AutoTokenizer.from_pretrained(
            cfg.pretrained_model_name_or_path)
        vocab_size = len(tokenizer)

        init_device = cfg.get('init_device', 'cpu')
        if init_device == 'cpu':
            if cfg.pretrained:
                model = T5ForConditionalGeneration.from_pretrained(
                    cfg.pretrained_model_name_or_path, config=config)
            else:
                model = T5ForConditionalGeneration.from_config(config)
        elif init_device == 'meta':
            if cfg.pretrained:
                raise ValueError(
                    'Setting cfg.pretrained=True is not supported when init_device="meta".'
                )
            with init_empty_weights(include_buffers=False):
                model = T5ForConditionalGeneration.from_config(config)
        else:
            raise ValueError(
                f'init_device="{init_device}" must be either "cpu" or "meta".')

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


# def create_hf_t5(
#     pretrained_model_name: str = 't5-base',
#     use_pretrained: Optional[bool] = False,
#     model_config: Optional[dict] = None,
#     tokenizer_name: Optional[str] = None,
#     z_loss: float = 0.0,
#     task_finetuning: Optional[bool] = False,
#     generation_eval: bool = False,
# ):
#     """T5 model based on |:hugging_face:| Transformers.

#     For more information, see `Transformers <https://huggingface.co/transformers/>`_.

#     Args:
#         pretrained_model_name (str): Name of the Hugging Face model to instantiate. Default: ``'t5-base'``.
#         use_pretrained (bool, optional): Whether to initialize the model with the pretrained weights. Default: ``False``.
#         model_config (dict): The settings used to create a Hugging Face T5Config. T5Config is used to specify the
#             architecture of a Hugging Face model.
#         tokenizer_name (str, optional): Tokenizer name used to preprocess the dataset and validate the models inputs.
#         z_loss (float, optional): Coefficient of `z-loss` term to use during training. Default: ``0.0``.
#         task_finetuning (bool, optional): Whether to add ExactMatch metric used in fine-tuning. Default: ``False``.
#         generation_eval (bool, optional): Whether evaluation requires generation, in which case RougeWithDetokenizer
#             is used for for the validation metric. Default: ``False``.

#     To create a |:hugging_face:| T5 model for Masked Language Model pretraining:

#     .. testcode::

#         from examples.ul2.src.hf_t5 import create_hf_t5
#         model = create_hf_t5('t5-base')
#     """
#     try:
#         from transformers import T5Config, T5ForConditionalGeneration
#     except ImportError as e:
#         raise MissingConditionalImportError(extra_deps_group='nlp',
#                                             conda_package='transformers') from e

#     if not pretrained_model_name:
#         pretrained_model_name = 't5-base'

#     # setup the tokenizer
#     tokenizer_name = tokenizer_name or pretrained_model_name
#     tokenizer = AutoTokenizerForMOD.from_pretrained(tokenizer_name)
#     vocab_size = tokenizer.vocab_size

#     if not model_config:
#         model_config = {}

#     if use_pretrained:
#         model = T5ForConditionalGeneration.from_pretrained(
#             pretrained_model_name_or_path=pretrained_model_name, **model_config)
#     else:
#         config = T5Config.from_pretrained(pretrained_model_name, **model_config)
#         model = T5ForConditionalGeneration(config)

#     # Expand the embeddings/vocab size to match the tokenizer
#     if model.config.vocab_size != vocab_size:
#         model.resize_token_embeddings(new_num_tokens=vocab_size)

#     # Super charge
#     prepare_hf_enc_dec_model_for_fsdp(model)

#     # We use `len(tokenizer) instead of `model.config.vocab_size` because
#     # `HuggingFaceModel` will set the latter to the former
#     metrics = [
#         # Note: The HF code is hardcoded to use -100 as the ignore index
#         LanguageCrossEntropy(ignore_index=_HF_IGNORE_INDEX,
#                              vocab_size=len(tokenizer)),
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

#         self.model_forward_args = inspect.getfullargspec(
#             self.model.forward).args

#     def forward(self, batch):
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

#         # Add a z_loss to the standard loss
#         logits_flat = logits.view(-1, logits.size(-1))
#         labels_flat = batch['labels'].view(-1)
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
#             return self.model.generate(
#                 batch['input_ids'],
#                 attention_mask=batch['attention_mask'],
#                 max_new_tokens=512,
#                 do_sample=True,
#                 top_p=0.90,
#                 top_k=0,
#                 no_repeat_ngram_size=3,
#             )

#         return super().eval_forward(batch, outputs)
