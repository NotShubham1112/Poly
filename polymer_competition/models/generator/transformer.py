from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.polychain.backbone import GINBackbone


@dataclass
class GeneratorConfig:
    vocab_size: int
    d_model: int = 512
    n_head: int = 8
    n_layer: int = 6
    d_ff: int = 2048
    dropout: float = 0.1
    max_seq_len: int = 256
    use_graph_encoder: bool = True
    graph_dim: int = 256
    n_pseudo_slots: int = 64


class GraphEncoder(nn.Module):
    def __init__(self, polychain_config: dict):
        super().__init__()
        cfg = polychain_config
        self.backbone = GINBackbone(
            in_dim=cfg.get("in_atom_dim", 50),
            edge_dim=cfg.get("in_edge_dim", 8),
            hidden_dim=cfg.get("hidden_dim", 256),
            n_layers=cfg.get("n_backbone_layers", 4),
            dropout=cfg.get("dropout", 0.1),
        )
        self.out_dim = self.backbone.out_dim

    def _encode_single(self, data) -> torch.Tensor:
        g, _ = self.backbone(data)
        return g

    def forward(self, batch: dict) -> torch.Tensor:
        h1 = self._encode_single(batch["monomer"])
        has_dimer = "dimer" in batch and batch["dimer"] is not None
        has_trimer = "trimer" in batch and batch["trimer"] is not None
        if has_dimer and has_trimer:
            h2 = self._encode_single(batch["dimer"])
            h3 = self._encode_single(batch["trimer"])
            stacked = torch.stack([h1, h2, h3], dim=0)
            graph_emb = stacked.mean(dim=0)
        else:
            graph_emb = h1
        return graph_emb


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_head: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, n_head, dropout=dropout, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_head, dropout=dropout, batch_first=True
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor | None = None,
        causal_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.norm1(x)
        h = self.self_attn(h, h, h, attn_mask=causal_mask, need_weights=False)[0]
        x = x + self.dropout(h)

        if memory is not None:
            h = self.norm2(x)
            h = self.cross_attn(h, memory, memory, need_weights=False)[0]
            x = x + self.dropout(h)

        h = self.norm3(x)
        h = self.ff(h)
        x = x + self.dropout(h)
        return x


class GeneratorDecoder(nn.Module):
    def __init__(self, config: GeneratorConfig):
        super().__init__()
        self.config = config

        self.token_embed = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, config.max_seq_len, config.d_model)
        )
        self.dropout = nn.Dropout(config.dropout)

        self.layers = nn.ModuleList([
            DecoderLayer(config.d_model, config.n_head, config.d_ff, config.dropout)
            for _ in range(config.n_layer)
        ])
        self.norm = nn.LayerNorm(config.d_model)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.property_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.ReLU(),
            nn.Linear(config.d_model // 2, 1),
        )

        self.property_proj = nn.Sequential(
            nn.Linear(1, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, config.d_model),
        )

        self.graph_proj: nn.Module | None = None
        if config.use_graph_encoder:
            self.graph_proj = nn.Sequential(
                nn.Linear(config.graph_dim, config.d_model * config.n_pseudo_slots),
            )
            self.pseudo_pos = nn.Parameter(
                torch.zeros(1, config.n_pseudo_slots, config.d_model)
            )

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_embed.weight, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)
        if self.graph_proj is not None:
            for m in self.graph_proj:
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.02)
                    nn.init.zeros_(m.bias)
            nn.init.normal_(self.pseudo_pos, std=0.02)
        for layer in self.layers:
            for m in layer.ff:
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        for m in self.property_head:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for m in self.property_proj:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _build_pseudo_sequence(self, graph_emb: torch.Tensor) -> torch.Tensor:
        B = graph_emb.size(0)
        h = self.graph_proj(graph_emb)
        h = h.view(B, self.config.n_pseudo_slots, self.config.d_model)
        h = h + self.pseudo_pos
        return h

    def _make_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device),
            diagonal=1,
        )
        return mask

    def forward(
        self,
        tokens: torch.LongTensor,
        graph_emb: torch.Tensor | None = None,
        property_cond: torch.Tensor | None = None,
        return_hidden: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, S = tokens.shape
        device = tokens.device

        token_emb = self.token_embed(tokens)
        pos_emb = self.pos_embed[:, :S, :]
        x = token_emb + pos_emb

        if property_cond is not None:
            property_cond = property_cond.view(-1)
            prop_emb = self.property_proj(property_cond.unsqueeze(-1).float())
            x = x + prop_emb.unsqueeze(1)

        x = self.dropout(x)

        memory = None
        if graph_emb is not None and self.graph_proj is not None:
            memory = self._build_pseudo_sequence(graph_emb)

        causal_mask = self._make_causal_mask(S, device)

        for layer in self.layers:
            x = layer(x, memory=memory, causal_mask=causal_mask)

        x = self.norm(x)
        token_logits = self.lm_head(x)

        if return_hidden:
            return token_logits, x
        return token_logits, x

    @torch.no_grad()
    def generate(
        self,
        prefix: torch.LongTensor,
        max_len: int,
        temperature: float = 1.0,
        top_k: int = 40,
        top_p: float = 0.9,
        graph_emb: torch.Tensor | None = None,
        property_cond: torch.Tensor | None = None,
        mask_fn: callable | None = None,
        eos_token_id: int = 2,
    ) -> dict:
        self.eval()
        batch_size = prefix.size(0)
        device = prefix.device
        tokens = prefix.clone()
        was_training = self.training

        for _ in range(max_len - prefix.size(1)):
            logits, _ = self.forward(
                tokens, graph_emb=graph_emb, property_cond=property_cond,
                return_hidden=False,
            )
            next_logits = logits[:, -1, :] / temperature

            if mask_fn is not None:
                next_logits = mask_fn(next_logits, tokens)

            if top_k > 0:
                k = min(top_k, next_logits.size(-1))
                indices_to_remove = next_logits < torch.topk(next_logits, k)[0][..., -1, None]
                next_logits[indices_to_remove] = float("-inf")

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                for b in range(batch_size):
                    indices_to_remove = sorted_indices[b][sorted_indices_to_remove[b]]
                    next_logits[b, indices_to_remove] = float("-inf")

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, next_token], dim=1)

            if (next_token == eos_token_id).all():
                break

        if mask_fn is not None:
            tokens = mask_fn.finalize(tokens, eos_token_id)

        final_logits, decoder_hidden = self.forward(
            tokens, graph_emb=graph_emb, property_cond=property_cond,
            return_hidden=True,
        )
        cls_hidden = decoder_hidden[:, 0, :]
        pred_property = self.property_head(cls_hidden).squeeze(-1)

        log_probs = self._compute_log_probs(final_logits, tokens)

        if was_training:
            self.train()

        return {
            "tokens": tokens,
            "logits": final_logits,
            "pred_property": pred_property,
            "log_probs": log_probs,
        }

    def _compute_log_probs(
        self, logits: torch.Tensor, tokens: torch.LongTensor
    ) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(dim=-1, index=tokens.unsqueeze(-1)).squeeze(-1)
        return token_log_probs.sum(dim=-1)
