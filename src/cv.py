from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, KFold

from .features import mcq_flag


def make_strata(perf: np.ndarray, queries) -> np.ndarray:
    n_correct = np.asarray(perf, np.float32).sum(1).astype(int)   # 0..11
    is_mcq = mcq_flag(pd.Series(list(queries))).to_numpy().astype(int)
    return n_correct * 2 + is_mcq


def make_folds_from_arrays(perf, queries, n_splits=5, seed=42):
    """Stratify on difficulty x mcq. Merge strata with < n_splits members into one
    catch-all bucket; if still infeasible, fall back to plain KFold. Always returns
    an exact, disjoint partition."""
    n = len(perf)
    strata = make_strata(perf, queries).astype(int)
    vals, counts = np.unique(strata, return_counts=True)
    rare = vals[counts < n_splits]
    if len(rare):
        strata = np.where(np.isin(strata, rare), -1, strata)
    _, counts = np.unique(strata, return_counts=True)
    if counts.min() < n_splits:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return [(tr.copy(), va.copy()) for tr, va in kf.split(np.arange(n))]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [(tr.copy(), va.copy()) for tr, va in skf.split(np.arange(n), strata)]


def get_folds(cfg, perf, queries):
    path = cfg.cache_dir / f"folds_{cfg.n_splits}_{cfg.seed}_{len(perf)}.npz"
    if path.exists():
        z = np.load(path, allow_pickle=True)
        return [(z[f"tr{i}"], z[f"va{i}"]) for i in range(cfg.n_splits)]
    folds = make_folds_from_arrays(perf, queries, cfg.n_splits, cfg.seed)
    save = {}
    for i, (tr, va) in enumerate(folds):
        save[f"tr{i}"] = tr
        save[f"va{i}"] = va
    np.savez(path, **save)
    return folds
