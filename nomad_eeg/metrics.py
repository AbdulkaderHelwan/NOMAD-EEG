"""Metrics and bootstrap confidence intervals, as reported in the paper.

Binary tasks: AUROC, PR-AUC, balanced accuracy, macro F1 (plus accuracy and weighted F1).
Multi-class tasks: balanced accuracy, Cohen's kappa, weighted F1 (plus macro F1).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             cohen_kappa_score, f1_score, roc_auc_score)


def scores(y: np.ndarray, prob: np.ndarray) -> dict:
    """y: [N] labels, prob: [N, n_classes] class probabilities."""
    n_classes = prob.shape[1]
    p = prob.argmax(1)
    s = {"balanced_accuracy": float(balanced_accuracy_score(y, p)),
         "f1_macro": float(f1_score(y, p, average="macro", zero_division=0)),
         "f1_weighted": float(f1_score(y, p, average="weighted", zero_division=0))}
    if n_classes == 2:
        s["auroc"] = float(roc_auc_score(y, prob[:, 1]))
        s["pr_auc"] = float(average_precision_score(y, prob[:, 1]))
        s["accuracy"] = float(accuracy_score(y, p))
    else:
        s["cohen_kappa"] = float(cohen_kappa_score(y, p))
    return s


def bootstrap_ci(y: np.ndarray, prob: np.ndarray, n: int = 1000, seed: int = 0) -> dict:
    """95% percentile intervals from n resamples of the test samples."""
    rng = np.random.default_rng(seed)
    acc: dict[str, list] = {}
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        for k, v in scores(y[i], prob[i]).items():
            acc.setdefault(k, []).append(v)
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in acc.items()}


def primary(n_classes: int) -> str:
    """Model-selection metric: AUROC for binary tasks, balanced accuracy otherwise."""
    return "auroc" if n_classes == 2 else "balanced_accuracy"
