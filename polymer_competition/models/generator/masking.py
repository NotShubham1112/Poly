from __future__ import annotations

import torch
import torch.nn.functional as F


class SELFIESMask:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_id = tokenizer.pad_token_id
        self.bos_id = tokenizer.bos_token_id
        self.eos_id = tokenizer.eos_token_id
        self.mask_id = tokenizer.mask_token_id

    def apply(
        self, logits: torch.Tensor, prefix_tokens: list[int]
    ) -> torch.Tensor:
        logits = logits.clone()

        if self.pad_id < logits.size(-1):
            if self.eos_id < logits.size(-1):
                has_eos = (prefix_tokens == self.eos_id).any().item() if isinstance(prefix_tokens, torch.Tensor) else (self.eos_id in prefix_tokens)
                if not has_eos:
                    logits[..., self.pad_id] = float("-inf")

        if self.bos_id < logits.size(-1):
            logits[..., self.bos_id] = float("-inf")

        return logits

    def finalize(
        self, tokens: torch.LongTensor, eos_token_id: int
    ) -> torch.LongTensor:
        final = tokens.clone()
        for b in range(final.size(0)):
            eos_positions = (final[b] == eos_token_id).nonzero(as_tuple=True)
            if len(eos_positions[0]) > 0:
                first_eos = eos_positions[0][0].item()
                if first_eos + 1 < final.size(1):
                    final[b, first_eos + 1:] = self.pad_id
        return final
