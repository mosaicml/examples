# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

""""""

from __future__ import annotations

import torch
import torch.nn as nn

from typing import Optional, Mapping, Any

from composer.metrics.nlp import LanguageCrossEntropy, MaskedAccuracy
from composer.models.huggingface import HuggingFaceModel
from composer.utils.import_helpers import MissingConditionalImportError

# __all__ = ['create_mosaic_bert_mlm']

class BERTEmbeddings(nn.Module):
    """Construct the embeddings from just word embeddings (no position or token type used)."""

    def __init__(self, cfg: Mapping[str, Any]):
        super().__init__()
        self.cfg = cfg
        self.word_embeddings = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=0)
        self.layernorm = nn.LayerNorm(cfg.d_model)
        self.dropout = nn.Dropout(cfg.emb_pdrop)

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        embeddings = self.word_embeddings(input_ids)
        embeddings = self.layernorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class BERTSelfAttention(nn.Module):
    pass

class BERTSelfOutput(nn.Module):
    def __init__(self, cfg: Mapping[str, Any]):
        super().__init__()
        self.cfg = cfg
        self.dense = nn.Linear(cfg.d_model, cfg.d_model)
        self.layernorm = nn.LayerNorm(cfg.d_model)
        self.dropout = nn.Dropout(cfg.hidden_pdrob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layernorm(hidden_states + input_tensor)
        return hidden_states

class BERTMLP(nn.Module):
    def __init__(self, cfg: Mapping[str, Any]):
        super().__init__()
        self.cfg = cfg
        self.intermediate_dim = cfg.d_model * cfg.mlp_ratio
        self.gated_layers = nn.Linear(cfg.d_model, self.intermediate_dim*2, bias=False)
        self.act = nn.GELU(approximate='none')
        self.wo = nn.Linear(self.intermediate_dim, cfg.d_model)
        self.dropout = nn.Dropout(cfg.hidden_pdrop)
        self.layernorm = torch.nn.LayerNorm(cfg.d_model)

    def forward(self, hidden_states: torch.Tensor, residual_connection: torch.Tensor):
        """
        Args:
            hidden_states (torch.Tensor): The hidden states from the attention matrix.
            residual_connection (torch.Tensor): The residual connection to add before the LayerNorm operator.
        """
        # compute the activation
        hidden_states = self.gated_layers(hidden_states)
        gated = hidden_states[:, :self.intermediate_dim]
        non_gated = hidden_states[:, self.intermediate_dim:]
        hidden_states = self.act(gated) * non_gated
        hidden_states = self.dropout(hidden_states)
        # multiply by the second matrix
        hidden_states = self.wo(hidden_states)
        # add the residual connection and post-LN
        hidden_states = self.layernorm(hidden_states + residual_connection)
        return hidden_states

class MosaicBERT(nn.Module):
    def __init__(self, cfg: Mapping[str, Any]):
        super().__init__()
        assert cfg.name == 'mosaic_bert', f'Tried to build MosaicBERT model with cfg.name={cfg.name}'
        self.cfg = cfg
        self.embedding = BERTEmbedding(cfg)
        self.layers = nn.ModuleList([BERTBlock(cfg) for _ in range(cfg.n_layers)])
        self.lm_head = BERTHeadMLM(cfg)
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(cfg.vocab_size, cfg.d_model, device=cfg.device),
                wpe=nn.Embedding(cfg.max_seq_len, cfg.d_model, device=cfg.device),
                emb_drop=nn.Dropout(cfg.emb_pdrop),
                blocks=nn.ModuleList([
                    GPTBlock(cfg, device=cfg.device) for _ in range(cfg.n_layers)
                ]),
                ln_f=nn.LayerNorm(cfg.d_model, device=cfg.device),
            ))
        self.lm_head = nn.Linear(cfg.d_model,
                                 cfg.vocab_size,
                                 bias=False,
                                 device=cfg.device)

# def create_mosaic_bert_mlm(pretrained_model_name: str = 'bert-base-uncased',
#                            use_pretrained: Optional[bool] = False,
#                            model_config: Optional[dict] = None,
#                            tokenizer_name: Optional[str] = None,
#                            gradient_checkpointing: Optional[bool] = False):