#!/usr/bin/env python3
"""Turn raw EEG windows into the dataset folder the other scripts read.

    python scripts/prepare_dataset.py --raw raw/ --out data/my_task --fs 256 \
        --channels FP1,FP2,F3,F4,C3,C4,P3,P4,O1,O2,F7,F8,T3,T4,T5,T6,FZ,CZ,PZ \
        --classes healthy,patient --bandpass 0.3 75 --notch 50

Input: raw/train.npz, raw/val.npz, raw/test.npz, each with
    X  [N, E, T]  windows in MICROVOLTS at --fs (MNE reads volts: multiply by 1e6)
    y  [N]        integer labels
    subjects [N]  optional subject ids
Keep the splits subject-disjoint. Robust-scaling statistics are fitted on train only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nomad_eeg import FS, RobustScaler, channel_coords, preprocess   # noqa: E402
from nomad_eeg.data import SPLITS                                    # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fs", type=float, required=True)
    ap.add_argument("--channels", required=True, help="comma-separated names, in the order of X's rows")
    ap.add_argument("--classes", required=True, help="comma-separated class names, index = label")
    ap.add_argument("--bandpass", type=float, nargs=2, default=None, metavar=("LO", "HI"))
    ap.add_argument("--notch", type=float, default=None)
    a = ap.parse_args()

    channels = [c.strip().upper() for c in a.channels.split(",")]
    channel_coords(channels)                 # fails early on a channel without a 10-20/10-05 position
    raw, out = Path(a.raw), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    data = {}
    for sp in SPLITS:
        d = np.load(raw / f"{sp}.npz")
        X = d["X"]
        assert X.ndim == 3 and X.shape[1] == len(channels), f"{sp}: X {X.shape} vs {len(channels)} channels"
        X = np.concatenate([preprocess(X[i:i + 512], a.fs, a.bandpass, a.notch)
                            for i in range(0, len(X), 512)])
        subj = d["subjects"] if "subjects" in d else np.full(len(X), "unknown")
        data[sp] = (X, d["y"].astype(np.int64), np.asarray(subj).astype(str))
        print(f"{sp}: {X.shape} at {FS} Hz, labels {np.bincount(data[sp][1])}", flush=True)

    scaler = RobustScaler().fit(data["train"][0])
    scaler.save(out / "scaler.json")
    for sp, (X, y, subj) in data.items():
        np.save(out / f"{sp}_X.npy", scaler.transform(X).astype(np.float16))
        np.save(out / f"{sp}_y.npy", y)
        np.save(out / f"{sp}_subjects.npy", subj)
    meta = {"channels": channels, "classes": a.classes.split(","), "fs": FS,
            "n_samples": int(data["train"][0].shape[-1])}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
