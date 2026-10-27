#!/usr/bin/env python3
"""Frozen linear probe, the paper's main protocol.

    python scripts/linear_probe.py --features features/my_task

Features are standardised with training mean and std. A class-balanced, L2-regularised
logistic regression is fitted on train for each C in 1e-6 ... 1e3; C is chosen on the
validation split (AUROC for binary, balanced accuracy for multi-class tasks), and the test
split is scored once, with 95% CIs from 1,000 bootstrap resamples of the test samples.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nomad_eeg.metrics import bootstrap_ci, primary, scores   # noqa: E402

C_GRID = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help="folder with train/val/test .npz (X, y)")
    ap.add_argument("--out", default=None, help="result json (default: <features>/probe.json)")
    a = ap.parse_args()

    root = Path(a.features)
    load = lambda sp: (lambda d: (d["X"], d["y"].astype(np.int64)))(np.load(root / f"{sp}.npz"))
    (Xtr, ytr), (Xva, yva), (Xte, yte) = load("train"), load("val"), load("test")
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Ztr, Zva, Zte = (Xtr - mu) / sd, (Xva - mu) / sd, (Xte - mu) / sd
    n_classes = int(max(ytr.max(), yte.max())) + 1
    key = primary(n_classes)

    val = {}
    for C in C_GRID:
        clf = LogisticRegression(max_iter=2000, C=C, class_weight="balanced").fit(Ztr, ytr)
        val[C] = scores(yva, clf.predict_proba(Zva))[key]
        print(f"  C={C:<8g} val {key} {val[C]:.4f}", flush=True)
    best = max(val, key=val.get)
    if best in (C_GRID[0], C_GRID[-1]):
        print(f"  [warn] best C={best} is at the edge of the grid", flush=True)
    clf = LogisticRegression(max_iter=2000, C=best, class_weight="balanced").fit(Ztr, ytr)
    prob = clf.predict_proba(Zte)
    res = {"best_C": best, "val_by_C": {str(k): v for k, v in val.items()},
           "n": {"train": len(ytr), "val": len(yva), "test": len(yte)}, "dim": int(Xtr.shape[1]),
           "test": scores(yte, prob), "ci95": bootstrap_ci(yte, prob)}
    out = Path(a.out) if a.out else root / "probe.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"\nTEST (C={best}): " + "  ".join(
        f"{k} {v:.4f} [{res['ci95'][k][0]:.3f}, {res['ci95'][k][1]:.3f}]" for k, v in res["test"].items()))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
