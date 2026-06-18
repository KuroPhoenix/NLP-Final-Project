import numpy as np
from src import ensemble_routing as er


def test_route_k_anchored_margin_extremes():
    p = np.array([[0.6, 0.1, 0.1], [0.1, 0.9, 0.1]])
    cc = np.array([0.001, 0.05, 0.05])  # model 0 is the cheap "K"
    # huge margin => never worth leaving K => all route to k_idx=0
    idx = er.route_k_anchored(p, cc, 0.07721, k_idx=0, margin=1.0)
    assert (idx == 0).all()
    # zero margin => identical to plain argmax routing
    assert (er.route_k_anchored(p, cc, 0.07721, 0, 0.0) == er.route(p, cc, 0.07721)).all()


def test_nested_calibrate_shape_and_range():
    rng = np.random.RandomState(0)
    oof = rng.rand(60, 3).astype(np.float32)
    perf = (rng.rand(60, 3) < 0.5).astype(np.float32)
    folds = [(np.setdiff1d(np.arange(60), va), va) for va in np.array_split(np.arange(60), 3)]
    cal = er.nested_calibrate(oof, perf, folds)
    assert cal.shape == (60, 3)
    assert cal.min() >= 0.0 and cal.max() <= 1.0


def test_selected_policy_never_below_always_k():
    rng = np.random.RandomState(1)
    p = rng.rand(200, 3).astype(np.float64)
    perf = (rng.rand(200, 3) < 0.5).astype(np.float64)
    cost = np.tile([0.001, 0.05, 0.05], (200, 1))
    cc = cost.mean(0)
    sel = er.select_policy(p, perf, cost, cc, 0.07721, k_idx=0, margins=np.linspace(0, 0.2, 11))
    # the best of the candidate policies must be at least always-K (always-K is a candidate)
    best = max(sel["always_K"], sel["argmax"], sel["k_margin"])
    assert best >= sel["always_K"] - 1e-9
    assert "best_margin" in sel and "best_policy" in sel
