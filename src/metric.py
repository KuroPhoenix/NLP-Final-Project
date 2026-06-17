from __future__ import annotations
import numpy as np

LETTERS = list("ABCDEFGHIJK")
MODEL_NAMES = [f"Model_{L}" for L in LETTERS]
PERF_COLS = [f"{m}_performance" for m in MODEL_NAMES]
COST_COLS = [f"{m}_cost" for m in MODEL_NAMES]

PERF_WEIGHT = 0.85
COST_WEIGHT = 0.15


def cost_denominator(cost: np.ndarray) -> float:
    d = float(np.asarray(cost, np.float64).max(axis=1).mean())
    if d <= 0:
        raise ValueError("cost denominator must be positive")
    return d


def route_reward(pred_idx, perf, cost, denom,
                 perf_weight=PERF_WEIGHT, cost_weight=COST_WEIGHT) -> float:
    pred_idx = np.asarray(pred_idx, np.int64)
    rows = np.arange(len(pred_idx))
    mp = float(np.asarray(perf, np.float64)[rows, pred_idx].mean())
    mc = float(np.asarray(cost, np.float64)[rows, pred_idx].mean())
    return perf_weight * mp - cost_weight * (mc / denom)


def expected_reward_matrix(p_hat, cost_const, denom,
                           perf_weight=PERF_WEIGHT, cost_weight=COST_WEIGHT) -> np.ndarray:
    p_hat = np.asarray(p_hat, np.float64)
    cost_const = np.asarray(cost_const, np.float64).reshape(1, -1)
    return perf_weight * p_hat - cost_weight * (cost_const / denom)


def expected_reward_matrix_from_truth(perf, cost, denom,
                                      perf_weight=PERF_WEIGHT, cost_weight=COST_WEIGHT) -> np.ndarray:
    perf = np.asarray(perf, np.float64)
    cost = np.asarray(cost, np.float64)
    return perf_weight * perf - cost_weight * (cost / denom)
