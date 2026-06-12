"""
models/chemberta.py
ChemBERTa embedding extractor + lightweight regression head.

ChemBERTa is frozen by default; the head is trained.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


class ChemBERTaRegressor(nn.Module):
    def __init__(self, model_name: str = "DeepChem/ChemBERTa-77M-MTR",
                 cache_dir: str | None = None,
                 hidden_dim: int = 768, out_dim: int = 1,
                 freeze_encoder: bool = True, dropout: float = 0.2):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.encoder = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, out_dim),
        )

    def forward(self, smiles_list: list[str], device: str = "cuda") -> torch.Tensor:
        tokens = self.tokenizer(smiles_list, padding=True, truncation=True,
                                return_tensors="pt").to(device)
        out = self.encoder(**tokens)
        # Use CLS-token embedding
        cls = out.last_hidden_state[:, 0, :]
        return self.head(cls).squeeze(-1)
