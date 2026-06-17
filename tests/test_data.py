import numpy as np
from src import data, metric


def test_build_targets_shapes(synthetic_train):
    perf, cost = data.build_targets(synthetic_train)
    assert perf.shape == (len(synthetic_train), 11)
    assert cost.shape == (len(synthetic_train), 11)
    assert set(np.unique(perf)).issubset({0.0, 1.0})


def test_cost_constants(synthetic_train):
    _, cost = data.build_targets(synthetic_train)
    cc = data.cost_constants(cost)
    assert cc.shape == (11,)
    assert np.allclose(cc, cost.mean(0), atol=1e-6)


def test_validate_columns_real(real_train):
    # should not raise
    data.validate_train_columns(real_train)
