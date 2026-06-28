"""Multi-task learning model for joint Tg and Egc prediction."""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class MultiTaskModel:
    """
    Multi-task neural network for joint Tg and Egc prediction.
    
    Architecture:
        Input → Shared Encoder → Tg Head → Tg Prediction
                              → Egc Head → Egc Prediction
    
    Loss: L = γ_egc * L_egc + L_tg
    """
    
    def __init__(self, n_features: int, hidden_dims: list = [128, 64, 32],
                 dropout: float = 0.2, gamma_egc: float = 100.0):
        """
        Initialize multi-task model.
        
        Args:
            n_features: Number of input features
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout rate
            gamma_egc: Scaling factor for Egc loss (Egc values are ~10x smaller)
        """
        self.n_features = n_features
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.gamma_egc = gamma_egc
        
        # Build model
        self.model = self._build_model()
        self.scaler_X = None
        self.scaler_y_tg = None
        self.scaler_y_egc = None
        self.tg_model = None
        self.egc_model = None
    
    def _build_model(self):
        """Build PyTorch model."""
        try:
            import torch
            import torch.nn as nn
            
            class MultiTaskNet(nn.Module):
                def __init__(self, n_features, hidden_dims, dropout):
                    super().__init__()
                    
                    # Shared encoder
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
                    
                    # Tg head
                    self.tg_head = nn.Sequential(
                        nn.Linear(in_dim, hidden_dims[-1]),
                        nn.ReLU(),
                        nn.Linear(hidden_dims[-1], 1)
                    )
                    
                    # Egc head
                    self.egc_head = nn.Sequential(
                        nn.Linear(in_dim, hidden_dims[-1]),
                        nn.ReLU(),
                        nn.Linear(hidden_dims[-1], 1)
                    )
                
                def forward(self, x):
                    shared = self.shared_encoder(x)
                    tg = self.tg_head(shared)
                    egc = self.egc_head(shared)
                    return tg.squeeze(), egc.squeeze()
            
            return MultiTaskNet(self.n_features, self.hidden_dims, self.dropout)
            
        except ImportError:
            print("Warning: PyTorch not available, using sklearn fallback")
            return None
    
    def fit(self, X: np.ndarray, y_tg: np.ndarray, y_egc: np.ndarray,
            epochs: int = 100, batch_size: int = 32, lr: float = 0.001):
        """
        Train multi-task model.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y_tg: Tg targets (n_samples,)
            y_egc: Egc targets (n_samples,)
            epochs: Number of training epochs
            batch_size: Batch size
            lr: Learning rate
        """
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset
            
            # Scale inputs
            from sklearn.preprocessing import StandardScaler
            self.scaler_X = StandardScaler()
            X_scaled = self.scaler_X.fit_transform(X)
            
            # Scale targets
            self.scaler_y_tg = StandardScaler()
            y_tg_scaled = self.scaler_y_tg.fit_transform(y_tg.reshape(-1, 1)).flatten()
            
            self.scaler_y_egc = StandardScaler()
            y_egc_scaled = self.scaler_y_egc.fit_transform(y_egc.reshape(-1, 1)).flatten()
            
            # Convert to tensors
            X_tensor = torch.FloatTensor(X_scaled)
            y_tg_tensor = torch.FloatTensor(y_tg_scaled)
            y_egc_tensor = torch.FloatTensor(y_egc_scaled)
            
            dataset = TensorDataset(X_tensor, y_tg_tensor, y_egc_tensor)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            # Move model to device
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = self.model.to(device)
            self.model.train()
            
            # Optimizer
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
            
            # Loss function
            mse_loss = nn.MSELoss()
            
            # Training loop
            for epoch in range(epochs):
                total_loss = 0
                for X_batch, tg_batch, egc_batch in loader:
                    X_batch = X_batch.to(device)
                    tg_batch = tg_batch.to(device)
                    egc_batch = egc_batch.to(device)
                    
                    optimizer.zero_grad()
                    tg_pred, egc_pred = self.model(X_batch)
                    
                    # Multi-task loss
                    loss_tg = mse_loss(tg_pred, tg_batch)
                    loss_egc = mse_loss(egc_pred, egc_batch)
                    loss = loss_tg + self.gamma_egc * loss_egc
                    
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                
                # Learning rate scheduling
                avg_loss = total_loss / len(loader)
                scheduler.step(avg_loss)
                
                if (epoch + 1) % 20 == 0:
                    print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
            
            self.model.eval()
            
        except ImportError:
            # Fallback to sklearn
            from sklearn.ensemble import GradientBoostingRegressor
            
            self.tg_model = GradientBoostingRegressor(n_estimators=100)
            self.egc_model = GradientBoostingRegressor(n_estimators=100)
            
            self.tg_model.fit(X, y_tg)
            self.egc_model.fit(X, y_egc)
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict Tg and Egc.
        
        Args:
            X: Feature matrix (n_samples, n_features)
        
        Returns:
            Tuple of (tg_predictions, egc_predictions)
        """
        if self.model is None and self.tg_model is not None:
            tg_pred = self.tg_model.predict(X)
            egc_pred = self.egc_model.predict(X)
            return tg_pred, egc_pred
        
        try:
            import torch
            
            if self.scaler_X is not None:
                X_scaled = self.scaler_X.transform(X)
            else:
                X_scaled = X
            X_tensor = torch.FloatTensor(X_scaled)
            
            device = next(self.model.parameters()).device
            X_tensor = X_tensor.to(device)
            
            # Predict
            self.model.eval()
            with torch.no_grad():
                tg_pred, egc_pred = self.model(X_tensor)
            
            # Inverse scale
            tg_np = tg_pred.cpu().numpy()
            egc_np = egc_pred.cpu().numpy()
            if self.scaler_y_tg is not None:
                tg_np = self.scaler_y_tg.inverse_transform(tg_np.reshape(-1, 1)).flatten()
            if self.scaler_y_egc is not None:
                egc_np = self.scaler_y_egc.inverse_transform(egc_np.reshape(-1, 1)).flatten()
            
            return tg_np, egc_np
            
        except Exception as e:
            print(f"Warning: PyTorch prediction failed: {e}")
            if self.tg_model is not None:
                tg_pred = self.tg_model.predict(X)
                egc_pred = self.egc_model.predict(X)
                return tg_pred, egc_pred
            raise RuntimeError("Model not fitted. Call fit() before predict().")
