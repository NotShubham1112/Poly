import torch
import torch.nn as nn


class MultiTaskModel(nn.Module):
    """Multi-task model with uncertainty-weighted loss (Kendall et al. 2018).
    
    Uses masking instead of zero-padding to handle mismatched dataset sizes
    (Tg: ~4143 samples, Egc: ~2028 samples).
    """
    
    def __init__(self, n_features: int, hidden_dims: list = [256, 128, 64], dropout: float = 0.2):
        super().__init__()
        
        layers = []
        in_dim = n_features
        for h_dim in hidden_dims[:-1]:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = h_dim
        self.shared_encoder = nn.Sequential(*layers)
        
        self.tg_head = nn.Sequential(
            nn.Linear(hidden_dims[-2], hidden_dims[-1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[-1], 1)
        )
        
        self.egc_head = nn.Sequential(
            nn.Linear(hidden_dims[-2], hidden_dims[-1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[-1], 1)
        )
        
        self.log_var_tg = nn.Parameter(torch.zeros(1))
        self.log_var_egc = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        h = self.shared_encoder(x)
        tg_pred = self.tg_head(h).squeeze(-1)
        egc_pred = self.egc_head(h).squeeze(-1)
        return tg_pred, egc_pred
    
    def loss(self, tg_pred, egc_pred, tg_true, egc_true, tg_mask, egc_mask):
        """Uncertainty-weighted multi-task loss.
        
        Only computes loss on masked (non-padded) samples.
        """
        if tg_mask.any():
            loss_tg = nn.functional.mse_loss(tg_pred[tg_mask], tg_true[tg_mask])
        else:
            loss_tg = torch.tensor(0.0, device=tg_pred.device)
        
        if egc_mask.any():
            loss_egc = nn.functional.mse_loss(egc_pred[egc_mask], egc_true[egc_mask])
        else:
            loss_egc = torch.tensor(0.0, device=egc_pred.device)
        
        precision_tg = torch.exp(-self.log_var_tg)
        precision_egc = torch.exp(-self.log_var_egc)
        
        total_loss = (
            precision_tg * loss_tg + self.log_var_tg +
            precision_egc * loss_egc + self.log_var_egc
        )
        
        return total_loss, {
            'loss_tg': loss_tg.item(),
            'loss_egc': loss_egc.item(),
            'log_var_tg': self.log_var_tg.item(),
            'log_var_egc': self.log_var_egc.item(),
        }
