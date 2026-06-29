import pytest
import torch
from models.multitask import MultiTaskModel

def test_multitask_forward():
    model = MultiTaskModel(n_features=100)
    x = torch.randn(32, 100)
    tg, egc = model(x)
    assert tg.shape == (32,)
    assert egc.shape == (32,)

def test_multitask_loss_with_masking():
    model = MultiTaskModel(n_features=100)
    x = torch.randn(32, 100)
    tg_pred, egc_pred = model(x)
    
    tg_true = torch.randn(32)
    egc_true = torch.randn(32)
    
    tg_mask = torch.arange(32) < 16
    egc_mask = torch.arange(32) >= 16
    
    loss, logs = model.loss(tg_pred, egc_pred, tg_true, egc_true, tg_mask, egc_mask)
    assert loss.requires_grad
    assert 'loss_tg' in logs
    assert 'loss_egc' in logs

def test_multitask_loss_all_masked():
    model = MultiTaskModel(n_features=100)
    x = torch.randn(32, 100)
    tg_pred, egc_pred = model(x)
    
    tg_true = torch.randn(32)
    egc_true = torch.randn(32)
    tg_mask = torch.ones(32, dtype=torch.bool)
    egc_mask = torch.ones(32, dtype=torch.bool)
    
    loss, logs = model.loss(tg_pred, egc_pred, tg_true, egc_true, tg_mask, egc_mask)
    assert loss.requires_grad

def test_multitask_log_var_values():
    model = MultiTaskModel(n_features=100)
    assert model.log_var_tg.item() == pytest.approx(0.0)
    assert model.log_var_egc.item() == pytest.approx(0.0)
