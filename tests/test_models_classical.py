import numpy as np
import pytest
from src import models_classical as mc, data, cv


def test_lgbm_oof_runs(synthetic_train):
    pytest.importorskip("lightgbm")
    perf, _ = data.build_targets(synthetic_train)
    X = np.random.RandomState(0).randn(len(perf), 12).astype(np.float32)
    folds = cv.make_folds_from_arrays(perf, synthetic_train["query"], 3, 42)
    oof = mc.lgbm_oof(X, perf, folds, n_estimators=50, lr=0.1, leaves=7, min_child=2)
    assert oof.shape == perf.shape
    assert (oof >= 0).all() and (oof <= 1).all()


def test_linear_oof_runs(synthetic_train):
    perf, _ = data.build_targets(synthetic_train)
    from scipy.sparse import csr_matrix
    X = csr_matrix(np.random.RandomState(1).rand(len(perf), 20))
    folds = cv.make_folds_from_arrays(perf, synthetic_train["query"], 3, 42)
    oof = mc.linear_oof(X, perf, folds)
    assert oof.shape == perf.shape
    assert (oof >= 0).all() and (oof <= 1).all()
