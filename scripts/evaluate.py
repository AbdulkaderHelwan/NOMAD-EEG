#!/usr/bin/env python3
"""Score a fine-tuned NOMAD-EEG checkpoint on one split of a dataset folder.

    python scripts/evaluate.py --data data/my_task --checkpoint runs/my_task/best.pt --split test

The dataset may have a different montage from the one used for fine-tuning: electrodes are
located by name, so only the channel names in meta.json need to be valid 10-20 / 10-05 names.
Writes <checkpoint dir>/eval_<split>.json and, with --save-probs, the class probabilities.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nomad_eeg import Classifier, channel_coords, load_pretrained                         # noqa: E402
from nomad_eeg.data import EEGSplit, load_meta                                               # noqa: E402
from nomad_eeg.metrics import bootstrap_ci, scores                                           # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from finetune import predict                                                                 # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--checkpoint", required=True, help="best.pt written by finetune.py")
    ap.add_argument("--split", default="test")
    ap.add_argument("--weights", default=None, help="local nomad_eeg_base.pt (only its config is used)")
    ap.add_argument("--save-probs", action="store_true")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=True)
    meta = load_meta(a.data)
    model = Classifier(load_pretrained(a.weights), n_classes=len(ck["classes"]))
    model.load_state_dict(ck["state_dict"])
    model.to(dev)
    coords = channel_coords(meta["channels"]).to(dev)
    ds = EEGSplit(a.data, a.split)
    y, prob = predict(model, DataLoader(ds, batch_size=128, num_workers=2), coords, dev)
    res = {"checkpoint": a.checkpoint, "data": a.data, "split": a.split, "n": len(y),
           "scores": scores(y, prob), "ci95": bootstrap_ci(y, prob)}
    out = Path(a.checkpoint).parent / f"eval_{a.split}.json"
    out.write_text(json.dumps(res, indent=2))
    if a.save_probs:
        np.savez(out.with_suffix(".npz"), y=y, prob=prob)
    print(f"{a.split} ({len(y):,} samples): " + "  ".join(
        f"{k} {v:.4f} [{res['ci95'][k][0]:.3f}, {res['ci95'][k][1]:.3f}]" for k, v in res["scores"].items()))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
