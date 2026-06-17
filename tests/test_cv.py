import numpy as np
from src import cv, data


def test_folds_partition(synthetic_train):
    perf, _ = data.build_targets(synthetic_train)
    folds = cv.make_folds_from_arrays(perf, synthetic_train["query"], n_splits=5, seed=42)
    assert len(folds) == 5
    all_val = np.concatenate([va for _, va in folds])
    assert sorted(all_val.tolist()) == list(range(len(perf)))  # exact partition
    for tr, va in folds:
        assert set(tr).isdisjoint(set(va))


def test_folds_deterministic(synthetic_train):
    perf, _ = data.build_targets(synthetic_train)
    q = synthetic_train["query"]
    f1 = cv.make_folds_from_arrays(perf, q, 5, 42)
    f2 = cv.make_folds_from_arrays(perf, q, 5, 42)
    for (a, b), (c, d) in zip(f1, f2):
        assert np.array_equal(b, d)
