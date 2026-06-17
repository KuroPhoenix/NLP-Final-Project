from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

from .metric import MODEL_NAMES, PERF_COLS, COST_COLS


def validate_train_columns(df: pd.DataFrame) -> None:
    expected = ["ID", "query"] + [c for pair in zip(PERF_COLS, COST_COLS) for c in pair]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"missing train columns: {missing}")


def load_data(cfg):
    train = pd.read_csv(cfg.data_dir / "train.csv")
    test = pd.read_csv(cfg.data_dir / "test.csv")
    sample = pd.read_csv(cfg.data_dir / "sample_submission.csv")
    validate_train_columns(train)
    assert list(test.columns) == ["ID", "query"], test.columns.tolist()
    assert list(sample.columns) == ["ID", "pred_model"], sample.columns.tolist()
    train["query"] = train["query"].fillna("").astype(str)
    test["query"] = test["query"].fillna("").astype(str)
    if cfg.smoke:
        train = train.sample(min(cfg.smoke_rows, len(train)),
                             random_state=cfg.seed).sort_values("ID").reset_index(drop=True)
        test = test.head(min(200, len(test))).copy()
        sample = sample.head(len(test)).copy()  # keep submission aligned with truncated test
    return train, test, sample


def build_targets(df: pd.DataFrame):
    perf = df[PERF_COLS].to_numpy(np.float32)
    cost = df[COST_COLS].to_numpy(np.float32)
    return perf, cost


def cost_constants(cost: np.ndarray) -> np.ndarray:
    return np.asarray(cost, np.float32).mean(axis=0)
