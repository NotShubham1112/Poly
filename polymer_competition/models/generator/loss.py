from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GenerativeLoss(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        property_weight: float = 0.5,
        label_smoothing: float = 0.0,
        ignore_index: int = 0,
    ):
        super().__init__()
        self.property_weight = property_weight
        self.ignore_index = ignore_index
        self.ce_loss = nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
        )
        self.mse_loss = nn.MSELoss()

    def forward(
        self,
        token_logits: torch.Tensor,
        targets: torch.Tensor,
        pred_property: torch.Tensor | None = None,
        true_property: torch.Tensor | None = None,
    ) -> torch.Tensor:
        token_logits = token_logits[:, :-1, :].contiguous()
        targets = targets[:, 1:].contiguous()

        B, S, V = token_logits.shape
        ce = self.ce_loss(token_logits.view(B * S, V), targets.view(B * S))

        loss = ce

        if pred_property is not None and true_property is not None:
            prop_loss = self.mse_loss(
                pred_property.view(-1), true_property.view(-1)
            )
            loss = loss + self.property_weight * prop_loss

        return loss
