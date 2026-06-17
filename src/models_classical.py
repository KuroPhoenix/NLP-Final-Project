from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression


def _lgbm_factory(n_estimators, lr, leaves, min_child, seed=42):
    from lightgbm import LGBMClassifier
    return lambda: LGBMClassifier(
        objective="binary", n_estimators=n_estimators, learning_rate=lr,
        num_leaves=leaves, min_child_samples=min_child, subsample=0.9,
        colsample_bytree=0.85, reg_alpha=0.05, reg_lambda=0.2,
        random_state=seed, n_jobs=-1, verbosity=-1)


def _const_or_proba(model, X):
    # robust when a fold's label column is single-class
    if hasattr(model, "classes_") and len(model.classes_) == 1:
        return np.full(X.shape[0], float(model.classes_[0]), dtype=np.float32)
    return model.predict_proba(X)[:, 1].astype(np.float32)


def _oof_multilabel(factory, X, Y, folds):
    oof = np.zeros_like(Y, dtype=np.float32)
    for tr, va in folds:
        for j in range(Y.shape[1]):
            ytr = Y[tr, j]
            if len(np.unique(ytr)) == 1:
                oof[va, j] = float(ytr[0])
                continue
            m = factory()
            m.fit(X[tr], ytr)
            oof[va, j] = _const_or_proba(m, X[va])
    return oof


def _full_multilabel(factory, X, Y, X_test):
    test = np.zeros((X_test.shape[0], Y.shape[1]), dtype=np.float32)
    for j in range(Y.shape[1]):
        y = Y[:, j]
        if len(np.unique(y)) == 1:
            test[:, j] = float(y[0])
            continue
        m = factory()
        m.fit(X, y)
        test[:, j] = _const_or_proba(m, X_test)
    return test


def lgbm_oof(X, Y, folds, n_estimators=1200, lr=0.03, leaves=31, min_child=30, seed=42):
    return _oof_multilabel(_lgbm_factory(n_estimators, lr, leaves, min_child, seed), X, Y, folds)


def lgbm_full(X, Y, X_test, n_estimators=1200, lr=0.03, leaves=31, min_child=30, seed=42):
    return _full_multilabel(_lgbm_factory(n_estimators, lr, leaves, min_child, seed), X, Y, X_test)


def _linear_factory(seed=42):
    return lambda: LogisticRegression(C=1.0, max_iter=2000, solver="liblinear", random_state=seed)


def linear_oof(X, Y, folds, seed=42):
    return _oof_multilabel(_linear_factory(seed), X, Y, folds)


def linear_full(X, Y, X_test, seed=42):
    return _full_multilabel(_linear_factory(seed), X, Y, X_test)
