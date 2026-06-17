import numpy as np
from src import metric


def test_constants():
    assert metric.MODEL_NAMES[0] == "Model_A"
    assert metric.MODEL_NAMES[-1] == "Model_K"
    assert len(metric.MODEL_NAMES) == 11
    assert metric.PERF_COLS[0] == "Model_A_performance"
    assert metric.COST_COLS[10] == "Model_K_cost"


def test_route_reward_simple():
    perf = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    cost = np.array([[0.1, 0.2], [0.3, 0.1]], dtype=np.float32)
    denom = 0.2
    # route to col 0 then col 1 => perf mean 1.0, cost mean (0.1+0.1)/2=0.1
    r = metric.route_reward(np.array([0, 1]), perf, cost, denom)
    assert abs(r - (0.85 * 1.0 - 0.15 * (0.1 / 0.2))) < 1e-6


def test_expected_reward_matrix_shape():
    p = np.full((4, 11), 0.5, dtype=np.float32)
    c = np.full(11, 0.01, dtype=np.float32)
    m = metric.expected_reward_matrix(p, c, denom=0.07721)
    assert m.shape == (4, 11)


def test_real_anchors(real_train):
    perf = real_train[metric.PERF_COLS].to_numpy(np.float64)
    cost = real_train[metric.COST_COLS].to_numpy(np.float64)
    denom = metric.cost_denominator(cost)
    assert abs(denom - 0.07721) < 1e-3
    k = metric.MODEL_NAMES.index("Model_K")
    always_k = metric.route_reward(np.full(len(perf), k), perf, cost, denom)
    assert abs(always_k - 0.450376) < 2e-3
    reward_mat = metric.expected_reward_matrix_from_truth(perf, cost, denom)
    oracle = metric.route_reward(reward_mat.argmax(1), perf, cost, denom)
    assert abs(oracle - 0.673071) < 2e-3
