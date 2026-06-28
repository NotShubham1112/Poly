import pytest
import numpy as np
import pandas as pd
from models.multitask import MultiTaskModel

def test_multitask_model_creation():
    model = MultiTaskModel(n_features=100, hidden_dims=[64, 32])
    assert model is not None

def test_multitask_model_forward():
    model = MultiTaskModel(n_features=100, hidden_dims=[64, 32])
    X = np.random.randn(10, 100).astype(np.float32)
    tg_pred, egc_pred = model.predict(X)
    assert tg_pred.shape == (10,)
    assert egc_pred.shape == (10,)

def test_multitask_model_train():
    model = MultiTaskModel(n_features=100, hidden_dims=[64, 32])
    X = np.random.randn(100, 100).astype(np.float32)
    y_tg = np.random.randn(100).astype(np.float32)
    y_egc = np.random.randn(100).astype(np.float32)
    
    model.fit(X, y_tg, y_egc, epochs=10, batch_size=32)
    tg_pred, egc_pred = model.predict(X)
    assert tg_pred.shape == (100,)
    assert egc_pred.shape == (100,)
