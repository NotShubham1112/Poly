"""
models.polychain.pretraining.sub_smiles_mask

(Optional) BERT-style atom masking on the SMILES tokens.

Used as an auxiliary SSL objective. We randomly replace SMILES tokens
with a [MASK] token and train the model to recover them. This is
the standard molecular SSL pretext; PolyChain combines it with the
polymer-specific asterisk-mask task.
"""
from __future__ import annotations

import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

from models.chemberta import ChemBERTaRegressor


# Use the same tokenizer as ChemBERTa for consistency
_TOKENIZER_NAME = "DeepChem/ChemBERTa-77M-MTR"
MASK_TOKEN = "[MASK]"


class SubSmilesMaskHead(nn.Module):
    """Predict masked SMILES tokens from ChemBERTa's hidden states."""

    def __init__(self, hidden_dim: int = 768, vocab_size: int = 800,
                 cache_dir: str | None = None):
        super().__init__()
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.head(hidden_states)


def sub_smiles_mask_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Standard MLM cross-entropy loss with -100 for non-masked positions."""
    return F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1),
                           ignore_index=-100)


def make_masked_tokens(smiles_list: list[str], mask_prob: float = 0.15,
                       cache_dir: str | None = None):
    """Tokenize SMILES and randomly mask tokens.

    Returns
    -------
    input_ids      : (B, L) LongTensor
    masked_labels  : (B, L) LongTensor, -100 for unmasked
    """
    tokenizer = AutoTokenizer.from_pretrained(_TOKENIZER_NAME, cache_dir=cache_dir)
    enc = tokenizer(smiles_list, padding=True, truncation=True,
                    return_tensors="pt")
    input_ids = enc["input_ids"].clone()
    labels = torch.full_like(input_ids, -100)

    # Build the mask: 15% of non-special tokens get masked
    prob = torch.full(input_ids.shape, mask_prob)
    # Don't mask special tokens
    special_mask = (input_ids == tokenizer.cls_token_id) | \
                   (input_ids == tokenizer.sep_token_id) | \
                   (input_ids == tokenizer.pad_token_id)
    prob.masked_fill_(special_mask, 0.0)
    masked_indices = torch.bernoulli(prob).bool()
    labels[masked_indices] = input_ids[masked_indices]

    # 80% [MASK], 10% random, 10% unchanged
    mask_token_id = tokenizer.mask_token_id
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    input_ids[indices_replaced] = mask_token_id

    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & \
                    masked_indices & ~indices_replaced
    random_words = torch.randint(tokenizer.vocab_size, labels.shape, dtype=torch.long)
    input_ids[indices_random] = random_words[indices_random]

    return input_ids, labels
