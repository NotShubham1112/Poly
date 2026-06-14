"""
inference.generator.py

PolymerGenerator – load a trained generative checkpoint and produce
novel polymer SMILES with various sampling strategies.

Usage:
    from inference.generator import PolymerGenerator
    gen = PolymerGenerator("outputs/checkpoints/anon_generator_best.pt")
    samples = gen.generate(n_samples=100, temperature=1.0, top_k=40)
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Callable

import numpy as np
import torch

os.environ["RDKIT_SKIP_VALIDATION_WARNINGS"] = "1"
logging.getLogger("rdkit").setLevel(logging.ERROR)

from models.generator import (
    SELFIESTokenizer,
    GeneratorConfig,
    GeneratorDecoder,
    GraphEncoder,
    MoleculeValidator,
    SELFIESMask,
)

DECODE_CONFIG = {
    "temperature": 1.0,
    "top_k": 40,
    "top_p": 0.9,
    "max_length": 256,
}


class PolymerGenerator:
    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.checkpoint_path = Path(checkpoint_path)
        self.validator = MoleculeValidator()
        self._load_checkpoint()

    def _load_checkpoint(self):
        ckpt = torch.load(
            self.checkpoint_path, map_location=self.device, weights_only=False,
        )

        cfg = ckpt.get("config", {})
        vocab = ckpt.get("vocab", {})
        self.tokenizer = SELFIESTokenizer()
        self.tokenizer._vocab = vocab
        self.tokenizer._inv_vocab = {v: k for k, v in vocab.items()}

        config = GeneratorConfig(
            vocab_size=cfg.get("vocab_size", len(vocab)),
            d_model=cfg.get("d_model", 512),
            n_head=cfg.get("n_head", 8),
            n_layer=cfg.get("n_layer", 6),
            use_graph_encoder=cfg.get("use_graph_encoder", True),
            graph_dim=256,
        )

        self.graph_encoder = None
        if config.use_graph_encoder:
            polychain_cfg = {
                "in_atom_dim": 50,
                "in_edge_dim": 8,
                "hidden_dim": 256,
                "n_backbone_layers": 4,
                "dropout": 0.0,
            }
            self.graph_encoder = GraphEncoder(polychain_cfg).to(self.device)
            if "graph_state" in ckpt and ckpt["graph_state"] is not None:
                self.graph_encoder.load_state_dict(ckpt["graph_state"])
            self.graph_encoder.eval()

        self.model = GeneratorDecoder(config).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

        self.config = config
        self.mask_fn = SELFIESMask(self.tokenizer)
        self.prop_scaler: PropertyScaler | None = None
        if "prop_scaler" in ckpt and ckpt["prop_scaler"] is not None:
            from training.train_generator import PropertyScaler
            from pathlib import Path
            self.prop_scaler = PropertyScaler()
            self.prop_scaler.load_state_dict(ckpt["prop_scaler"])
        print(f"Loaded generator from {self.checkpoint_path}")

    @torch.no_grad()
    def generate(
        self,
        n_samples: int = 100,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        max_length: int | None = None,
        property_target: float | None = None,
        filter_valid: bool = True,
        batch_size: int = 32,
    ) -> list[dict]:
        temp = temperature if temperature is not None else DECODE_CONFIG["temperature"]
        k = top_k if top_k is not None else DECODE_CONFIG["top_k"]
        p = top_p if top_p is not None else DECODE_CONFIG["top_p"]
        max_len = max_length if max_length is not None else DECODE_CONFIG["max_length"]

        prop_cond = None
        if property_target is not None and self.prop_scaler is not None:
            prop_cond = torch.tensor(
                self.prop_scaler.transform([property_target]),
                dtype=torch.float, device=self.device,
            )
        elif property_target is not None:
            prop_cond = torch.tensor(
                [property_target], dtype=torch.float, device=self.device,
            )

        results = []
        n_remaining = n_samples

        while n_remaining > 0:
            current_bs = min(batch_size, n_remaining)
            prefix = torch.full(
                (current_bs, 1),
                self.tokenizer.bos_token_id,
                dtype=torch.long,
                device=self.device,
            )

            graph_emb = None
            if self.graph_encoder is not None:
                pass

            batch_prop = prop_cond.expand(current_bs, 1) if prop_cond is not None else None
            outputs = self.model.generate(
                prefix=prefix,
                max_len=max_len,
                temperature=temp,
                top_k=k,
                top_p=p,
                graph_emb=graph_emb,
                property_cond=batch_prop,
                mask_fn=self.mask_fn.apply,
                eos_token_id=self.tokenizer.eos_token_id,
            )

            tokens = outputs["tokens"].cpu().tolist()
            log_probs = outputs["log_probs"].cpu().tolist()
            pred_norm = outputs["pred_property"].cpu().numpy()
            if self.prop_scaler is not None and self.prop_scaler.fitted:
                pred_norm = self.prop_scaler.inverse(pred_norm)
            pred_properties = pred_norm.tolist()

            for i in range(current_bs):
                smi = self.tokenizer.decode(tokens[i])
                is_valid, reason = self.validator.validate(smi) if smi else (False, "empty decode")

                sample = {
                    "smiles": smi,
                    "selfies": "",
                    "valid": is_valid,
                    "property_pred": pred_properties[i] if isinstance(pred_properties[i], (int, float)) else pred_properties[i],
                    "log_prob": log_probs[i] if isinstance(log_probs[i], (int, float)) else log_probs[i],
                    "validation_reason": reason,
                }

                if filter_valid and not is_valid:
                    continue

                results.append(sample)
                n_remaining -= 1

                if len(results) >= n_samples:
                    break

            if len(results) >= n_samples:
                break

        return results[:n_samples]

    @torch.no_grad()
    def generate_beam(
        self,
        beam_width: int = 5,
        max_length: int = 256,
        n_best: int = 5,
    ) -> list[dict]:
        prefix = torch.LongTensor([[self.tokenizer.bos_token_id]]).to(self.device)
        beams = [(prefix, 0.0)]

        for step in range(max_length - 1):
            candidates = []
            for tokens, score in beams:
                logits, _ = self.model(tokens, graph_emb=None)
                next_logits = logits[:, -1, :]
                probs = torch.softmax(next_logits, dim=-1)
                top_probs, top_indices = torch.topk(probs, beam_width, dim=-1)

                for i in range(beam_width):
                    next_token = top_indices[0, i].unsqueeze(0).unsqueeze(0)
                    new_tokens = torch.cat([tokens, next_token], dim=1)
                    new_score = score + torch.log(top_probs[0, i] + 1e-10).item()
                    candidates.append((new_tokens, new_score))

            candidates.sort(key=lambda x: x[1], reverse=True)
            beams = candidates[:beam_width]

            if all(tokens[0, -1].item() == self.tokenizer.eos_token_id for tokens, _ in beams):
                break

        beams.sort(key=lambda x: x[1], reverse=True)
        results = []
        for tokens, score in beams[:n_best]:
            smi = self.tokenizer.decode(tokens[0].tolist())
            is_valid, reason = self.validator.validate(smi) if smi else (False, "empty decode")
            results.append({
                "smiles": smi,
                "selfies": "",
                "valid": is_valid,
                "property_pred": 0.0,
                "log_prob": score,
                "validation_reason": reason,
            })
        return results
