import pytest
from features.build_features import build_periodic_graph_features

def test_periodic_features_shape():
    df = build_periodic_graph_features(["*CC(=O)OCCO*", "*CC*"], n_repeats=3)
    assert df.shape[0] == 2
    assert df.shape[1] == 14
    assert not df.isna().all().all()

def test_periodic_features_nonzero():
    df = build_periodic_graph_features(["*CC(=O)OCCO*"], n_repeats=3)
    assert df['periodic_mw'].iloc[0] > 0
    assert df['periodic_chain_length'].iloc[0] == 3

def test_periodic_features_invalid():
    df = build_periodic_graph_features(["invalid"], n_repeats=3)
    assert df.shape[0] == 1
    assert df['periodic_mw'].iloc[0] == 0.0
