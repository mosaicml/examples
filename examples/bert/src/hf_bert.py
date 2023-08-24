# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

"""Implements a Hugging Face BERT wrapped inside a :class:`.ComposerModel`."""

from __future__ import annotations

from typing import Optional, Union, Tuple

import torch
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from composer.metrics.nlp import (BinaryF1Score, LanguageCrossEntropy,
                                  MaskedAccuracy)
from composer.models.huggingface import HuggingFaceModel
from composer.utils.import_helpers import MissingConditionalImportError
from torchmetrics import MeanSquaredError
from torchmetrics.classification.accuracy import MulticlassAccuracy
from torchmetrics.classification.matthews_corrcoef import MatthewsCorrCoef
from torchmetrics.regression.spearman import SpearmanCorrCoef
from transformers.models.bert import BertForSequenceClassification
from transformers.models.bert import BertOnlyMLMHead
from transformers.modeling_outputs import SequenceClassifierOutput

__all__ = ['create_hf_bert_mlm', 'create_hf_bert_classification']


def create_hf_bert_mlm(pretrained_model_name: str = 'bert-base-uncased',
                       use_pretrained: Optional[bool] = False,
                       model_config: Optional[dict] = None,
                       tokenizer_name: Optional[str] = None,
                       gradient_checkpointing: Optional[bool] = False):
    """BERT model based on |:hugging_face:| Transformers.

    For more information, see `Transformers <https://huggingface.co/transformers/>`_.

    Args:
        pretrained_model_name (str): Name of the Hugging Face model to instantiate. Default: ``'bert-base-uncased'``.
        use_pretrained (bool, optional): Whether to initialize the model with the pretrained weights. Default: ``False``.
        model_config (dict): The settings used to create a Hugging Face BertConfig. BertConfig is used to specify the
            architecture of a Hugging Face model.
        tokenizer_name (str, optional): Tokenizer name used to preprocess the dataset and validate the models inputs.
        gradient_checkpointing (bool, optional): Use gradient checkpointing. Default: ``False``.

        .. code-block::

            {
              "_name_or_path": "bert-base-uncased",
              "architectures": ["BertForMaskedLM"],
              "attention_probs_dropout_prob": 0.1,
              "classifier_dropout": null,
              "gradient_checkpointing": false,
              "hidden_act": "gelu",
              "hidden_dropout_prob": 0.1,
              "hidden_size": 768,
              "initializer_range": 0.02,
              "intermediate_size": 3072,
              "layer_norm_eps": 1e-12,
              "max_position_embeddings": 512,
              "model_type": "bert",
              "num_attention_heads": 12,
              "num_hidden_layers": 12,
              "pad_token_id": 0,
              "position_embedding_type": "absolute",
              "transformers_version": "4.16.0",
              "type_vocab_size": 2,
              "use_cache": true,
              "vocab_size": 30522
            }

    To create a |:hugging_face:| BERT model for Masked Language Model pretraining:

     .. testcode::

         from examples.bert.src.hf_bert import create_hf_bert_mlm
         model = create_hf_bert_mlm()
    """
    try:
        import transformers
    except ImportError as e:
        raise MissingConditionalImportError(extra_deps_group='nlp',
                                            conda_package='transformers') from e

    if not model_config:
        model_config = {}

    if not pretrained_model_name:
        pretrained_model_name = 'bert-base-uncased'

    if use_pretrained:
        assert transformers.AutoModelForMaskedLM.from_pretrained is not None, 'AutoModelForMaskedLM has from_pretrained method'
        model = transformers.AutoModelForMaskedLM.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name, **model_config)
    else:
        config = transformers.AutoConfig.from_pretrained(
            pretrained_model_name, **model_config)
        assert transformers.AutoModelForMaskedLM.from_config is not None, 'AutoModelForMaskedLM has from_config method'
        model = transformers.AutoModelForMaskedLM.from_config(config)

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()  # type: ignore

    # setup the tokenizer
    if tokenizer_name:
        tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
    else:
        tokenizer = None

    metrics = [
        LanguageCrossEntropy(ignore_index=-100,
                             vocab_size=model.config.vocab_size),
        MaskedAccuracy(ignore_index=-100)
    ]
    return HuggingFaceModel(model=model,
                            tokenizer=tokenizer,
                            use_logits=True,
                            metrics=metrics)

class BertForRTS(BertForSequenceClassification):

    def __init__(self, config):
        super().__init__(config)

        config.vocab_size = 2
        self.cls = BertOnlyMLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()


    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[0]
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    #def forward(
        #self,
        #input_ids: Optional[torch.Tensor] = None,
        #attention_mask: Optional[torch.Tensor] = None,
        #token_type_ids: Optional[torch.Tensor] = None,
        #position_ids: Optional[torch.Tensor] = None,
        #head_mask: Optional[torch.Tensor] = None,
        #inputs_embeds: Optional[torch.Tensor] = None,
        #encoder_hidden_states: Optional[torch.Tensor] = None,
        #encoder_attention_mask: Optional[torch.Tensor] = None,
        #labels: Optional[torch.Tensor] = None,
        #output_attentions: Optional[bool] = None,
        #output_hidden_states: Optional[bool] = None,
        #return_dict: Optional[bool] = None,
    #):
        #r"""
        #labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            #Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            #config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            #loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        #"""

        #return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        #outputs = self.bert(
            #input_ids,
            #attention_mask=attention_mask,
            #token_type_ids=token_type_ids,
            #position_ids=position_ids,
            #head_mask=head_mask,
            #inputs_embeds=inputs_embeds,
            #encoder_hidden_states=encoder_hidden_states,
            #encoder_attention_mask=encoder_attention_mask,
            #output_attentions=output_attentions,
            #output_hidden_states=output_hidden_states,
            #return_dict=return_dict,
        #)

        #sequence_output = outputs[0]
        #prediction_scores = self.cls(sequence_output)

        #masked_lm_loss = None
        #if labels is not None:
            #loss_fct = CrossEntropyLoss()  # -100 index = padding token
            #masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        #if not return_dict:
            #output = (prediction_scores,) + outputs[2:]
            #return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        #return MaskedLMOutput(
            #loss=masked_lm_loss,
            #logits=prediction_scores,
            #hidden_states=outputs.hidden_states,
            #attentions=outputs.attentions,
        #)

def create_hf_bert_rts(pretrained_model_name: str = 'bert-base-uncased',
                       use_pretrained: Optional[bool] = False,
                       model_config: Optional[dict] = None,
                       tokenizer_name: Optional[str] = None,
                       gradient_checkpointing: Optional[bool] = False):
    """BERT model based on |:hugging_face:| Transformers.

    For more information, see `Transformers <https://huggingface.co/transformers/>`_.

    Args:
        pretrained_model_name (str): Name of the Hugging Face model to instantiate. Default: ``'bert-base-uncased'``.
        use_pretrained (bool, optional): Whether to initialize the model with the pretrained weights. Default: ``False``.
        model_config (dict): The settings used to create a Hugging Face BertConfig. BertConfig is used to specify the
            architecture of a Hugging Face model.
        tokenizer_name (str, optional): Tokenizer name used to preprocess the dataset and validate the models inputs.
        gradient_checkpointing (bool, optional): Use gradient checkpointing. Default: ``False``.

        .. code-block::

            {
              "_name_or_path": "bert-base-uncased",
              "architectures": ["BertForMaskedLM"],
              "attention_probs_dropout_prob": 0.1,
              "classifier_dropout": null,
              "gradient_checkpointing": false,
              "hidden_act": "gelu",
              "hidden_dropout_prob": 0.1,
              "hidden_size": 768,
              "initializer_range": 0.02,
              "intermediate_size": 3072,
              "layer_norm_eps": 1e-12,
              "max_position_embeddings": 512,
              "model_type": "bert",
              "num_attention_heads": 12,
              "num_hidden_layers": 12,
              "pad_token_id": 0,
              "position_embedding_type": "absolute",
              "transformers_version": "4.16.0",
              "type_vocab_size": 2,
              "use_cache": true,
              "vocab_size": 30522
            }

    To create a |:hugging_face:| BERT model for Masked Language Model pretraining:

     .. testcode::

         from examples.bert.src.hf_bert import create_hf_bert_mlm
         model = create_hf_bert_mlm()
    """
    try:
        import transformers
    except ImportError as e:
        raise MissingConditionalImportError(extra_deps_group='nlp',
                                            conda_package='transformers') from e

    if not model_config:
        model_config = {}

    model_config['num_labels'] = 2

    if not pretrained_model_name:
        pretrained_model_name = 'bert-base-uncased'

    if use_pretrained:
        assert BertForRTS.from_pretrained is not None, 'AutoModelForSequenceClassification has from_pretrained method'
        model = BertForRTS.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name, **model_config)
    else:
        config = transformers.AutoConfig.from_pretrained(
            pretrained_model_name, **model_config)
        #assert BertForRTS.from_config is not None, 'AutoModelForSequenceClassification has from_config method'
        model = BertForRTS(config)

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()  # type: ignore

    # setup the tokenizer
    if tokenizer_name:
        tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
    else:
        tokenizer = None

    metrics = [
        LanguageCrossEntropy(ignore_index=-100, vocab_size=2),
        MaskedAccuracy(ignore_index=-100)
    ]
    return HuggingFaceModel(model=model,
                            tokenizer=tokenizer,
                            use_logits=True,
                            metrics=metrics)


def create_hf_bert_classification(
        num_labels: int,
        pretrained_model_name: str = 'bert-base-uncased',
        use_pretrained: Optional[bool] = False,
        model_config: Optional[dict] = None,
        tokenizer_name: Optional[str] = None,
        gradient_checkpointing: Optional[bool] = False):
    """BERT model based on |:hugging_face:| Transformers.

    For more information, see `Transformers <https://huggingface.co/transformers/>`_.

    Args:
        num_labels (int): The number of classes in the task (``1`` indicates regression). Default: ``2``.
        pretrained_model_name (str): Name of the Hugging Face model to instantiate. Default: ``'bert-base-uncased'``.
        use_pretrained (bool, optional): Whether to initialize the model with the pretrained weights. Default: ``False``.
        model_config (dict, optional): The settings used to create a Hugging Face BertConfig. BertConfig is used to specify the
            architecture of a Hugging Face model.
        tokenizer_name (str, optional): Tokenizer name used to preprocess the dataset and validate the models inputs.
        gradient_checkpointing (bool, optional): Use gradient checkpointing. Default: ``False``.

        .. code-block::

            {
              "_name_or_path": "bert-base-uncased",
              "architectures": [
                "BertForSequenceClassification
              ],
              "attention_probs_dropout_prob": 0.1,
              "classifier_dropout": null,
              "gradient_checkpointing": false,
              "hidden_act": "gelu",
              "hidden_dropout_prob": 0.1,
              "hidden_size": 768,
              "id2label": {
                "0": "LABEL_0",
                "1": "LABEL_1",
                "2": "LABEL_2"
              },
              "initializer_range": 0.02,
              "intermediate_size": 3072,
              "label2id": {
                "LABEL_0": 0,
                "LABEL_1": 1,
                "LABEL_2": 2
              },
              "layer_norm_eps": 1e-12,
              "max_position_embeddings": 512,
              "model_type": "bert",
              "num_attention_heads": 12,
              "num_hidden_layers": 12,
              "pad_token_id": 0,
              "position_embedding_type": "absolute",
              "transformers_version": "4.16.0",
              "type_vocab_size": 2,
              "use_cache": true,
              "vocab_size": 30522
            }

    Note:
        This function can be used to construct a BERT model for regression by setting ``num_labels == 1``.
        This will have two noteworthy effects. First, it will switch the training loss to :class:`~torch.nn.MSELoss`.
        Second, the returned :class:`.ComposerModel`'s train/validation metrics will be :class:`~torchmetrics.MeanSquaredError` and :class:`~torchmetrics.SpearmanCorrCoef`.

        For the classifcation case (when ``num_labels > 1``), the training loss is :class:`~torch.nn.CrossEntropyLoss`, and the train/validation
        metrics are :class:`~torchmetrics.Accuracy` and :class:`~torchmetrics.MatthewsCorrCoef`, as well as :class:`.BinaryF1Score` if ``num_labels == 2``.
    """
    try:
        import transformers
    except ImportError as e:
        raise MissingConditionalImportError(extra_deps_group='nlp',
                                            conda_package='transformers') from e

    if not model_config:
        model_config = {}

    model_config['num_labels'] = num_labels

    if not pretrained_model_name:
        pretrained_model_name = 'bert-base-uncased'

    if use_pretrained:
        assert transformers.AutoModelForSequenceClassification.from_pretrained is not None, 'AutoModelForSequenceClassification has from_pretrained method'
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name, **model_config)
    else:
        config = transformers.AutoConfig.from_pretrained(
            pretrained_model_name, **model_config)
        assert transformers.AutoModelForSequenceClassification.from_config is not None, 'AutoModelForSequenceClassification has from_config method'
        model = transformers.AutoModelForSequenceClassification.from_config(
            config)

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # setup the tokenizer
    if tokenizer_name:
        tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
    else:
        tokenizer = None

    if num_labels == 1:
        # Metrics for a regression model
        metrics = [MeanSquaredError(), SpearmanCorrCoef()]
    else:
        # Metrics for a classification model
        metrics = [
            MulticlassAccuracy(num_classes=num_labels, average='micro'),
            MatthewsCorrCoef(task="multiclass",
                             num_classes=model.config.num_labels)
        ]
        if num_labels == 2:
            metrics.append(BinaryF1Score())

    return HuggingFaceModel(model=model,
                            tokenizer=tokenizer,
                            use_logits=True,
                            metrics=metrics)
