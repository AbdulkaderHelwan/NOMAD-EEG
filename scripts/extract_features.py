#!/usr/bin/env python3
"""Frozen NOMAD-EEG features for every split of a dataset folder.

    python scripts/extract_features.py --data data/my_task --out features/my_task

Writes features/my_task/{train,val,test}.npz with X [N, 10240] and y [N].
--weights uses a local nomad_eeg_base.pt instead of downloading it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nomad_eeg import extract_features, load_pretrained   # noqa: E402
from nomad_eeg.data import SPLITS, load_meta, load_split  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--n-windows", type=int, default=5)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_pretrained(a.weights, device=dev)
    meta = load_meta(a.data)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(meta['channels'])} electrodes, {meta['n_samples']} samples per window, device {dev}", flush=True)
    for sp in SPLITS:
        X, y = load_split(a.data, sp)
        F = extract_features(model, X, meta["channels"], batch_size=a.batch_size, n_windows=a.n_windows)
        assert np.isfinite(F).all()
        np.savez_compressed(out / f"{sp}.npz", X=F, y=y)
        print(f"  {sp:<5} {F.shape}", flush=True)


if __name__ == "__main__":
    main()
