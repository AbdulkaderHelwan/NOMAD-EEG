#!/usr/bin/env python3
"""Fine-tune NOMAD-EEG end to end on a dataset folder (the paper's schedule).

    python scripts/finetune.py --data data/my_task --out runs/my_task

Head: per-electrode attention pooling (nomad_eeg.Classifier). Schedule: --frozen-epochs
with the encoder frozen (head lr 5e-4), then --epochs end to end with the encoder at a 10x
lower learning rate (5e-5), AdamW (weight decay 0.05), 5% warm-up then cosine decay,
class-weighted cross-entropy, gradient clipping at 1. The epoch with the best validation
balanced accuracy is kept (runs/my_task/best.pt) and the test split is scored once.
The paper used 2 + 20 epochs (2 + 10 for TUEV), batch 32.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nomad_eeg import CORPUS_ID, Classifier, channel_coords, load_pretrained   # noqa: E402
from nomad_eeg.data import EEGSplit, load_meta                                 # noqa: E402
from nomad_eeg.metrics import bootstrap_ci, scores                             # noqa: E402


@torch.no_grad()
def predict(model, dl, coords, dev):
    model.eval()
    ps, ys = [], []
    for x, y in dl:
        x = x.to(dev, non_blocking=True)
        cid = torch.full((x.size(0),), CORPUS_ID, dtype=torch.long, device=dev)
        with torch.autocast("cuda", torch.bfloat16, enabled=dev == "cuda"):
            ps.append(model(x, coords, cid, train_encoder=False).float().softmax(1).cpu())
        ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(ps).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", default=None, help="local nomad_eeg_base.pt (default: download)")
    ap.add_argument("--frozen-epochs", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr-head", type=float, default=5e-4)
    ap.add_argument("--lr-enc", type=float, default=5e-5)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=0, help="cap steps per epoch (quick test)")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)

    meta = load_meta(a.data)
    classes = meta["classes"]
    coords = channel_coords(meta["channels"]).to(dev)
    model = Classifier(load_pretrained(a.weights), n_classes=len(classes)).to(dev)
    tr, va, te = (EEGSplit(a.data, sp) for sp in ("train", "val", "test"))
    print(f"{len(meta['channels'])} electrodes, classes {classes} | train {len(tr):,} "
          f"val {len(va):,} test {len(te):,} | device {dev}", flush=True)

    counts = np.bincount(tr.y, minlength=len(classes)).astype(np.float64)
    w = torch.tensor(counts.sum() / (len(classes) * np.maximum(counts, 1)), dtype=torch.float32, device=dev)
    lossf = nn.CrossEntropyLoss(weight=w)
    kw = dict(num_workers=a.workers, pin_memory=dev == "cuda", persistent_workers=a.workers > 0)
    dl_tr = DataLoader(tr, batch_size=a.batch, shuffle=True, drop_last=True, **kw)
    dl_va = DataLoader(va, batch_size=128, **kw)
    dl_te = DataLoader(te, batch_size=128, **kw)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    head = [p for n, p in model.named_parameters() if not n.startswith("enc.")]
    best, hist = {"val_balanced_accuracy": -1.0}, []

    def phase(name, n_epochs, train_encoder, groups):
        nonlocal best
        opt = torch.optim.AdamW(groups, weight_decay=a.wd, betas=(0.9, 0.99))
        per = min(len(dl_tr), a.max_steps) if a.max_steps else len(dl_tr)
        total, warm = n_epochs * per, max(1, int(0.05 * n_epochs * per))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * 0.5 *
                                                  (1 + math.cos(math.pi * min(1.0, s / total))))
        for ep in range(n_epochs):
            model.train(); model.enc.train(train_encoder)
            t0, run_loss, n = time.time(), 0.0, 0
            for step, (x, y) in enumerate(dl_tr):
                if a.max_steps and step >= a.max_steps:
                    break
                x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
                cid = torch.full((x.size(0),), CORPUS_ID, dtype=torch.long, device=dev)
                with torch.autocast("cuda", torch.bfloat16, enabled=dev == "cuda"):
                    logits = model(x, coords, cid, train_encoder=train_encoder)
                loss = lossf(logits.float(), y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step()
                run_loss += loss.item(); n += 1
            v = scores(*predict(model, dl_va, coords, dev))
            hist.append({"phase": name, "epoch": ep + 1, "train_loss": run_loss / max(1, n),
                         **{f"val_{k}": x for k, x in v.items()}})
            mark = ""
            if v["balanced_accuracy"] > best["val_balanced_accuracy"]:
                best = {"val_balanced_accuracy": v["balanced_accuracy"], "phase": name, "epoch": ep + 1}
                torch.save({"state_dict": model.state_dict(), "classes": classes,
                            "channels": meta["channels"]}, out / "best.pt")
                mark = "  <- best"
            print(f"[{name} ep {ep + 1}] loss {run_loss / max(1, n):.4f} | val "
                  + " ".join(f"{k} {x:.4f}" for k, x in v.items()) + f" ({time.time() - t0:.0f}s){mark}",
                  flush=True)
            (out / "history.json").write_text(json.dumps({"best": best, "history": hist}, indent=2))

    for p in model.enc.parameters():
        p.requires_grad_(False)
    phase("frozen", a.frozen_epochs, False, [{"params": head, "lr": a.lr_head}])
    for p in model.enc.parameters():
        p.requires_grad_(True)
    phase("finetune", a.epochs, True, [{"params": head, "lr": a.lr_head},
                                       {"params": model.enc.parameters(), "lr": a.lr_enc}])

    model.load_state_dict(torch.load(out / "best.pt", map_location=dev, weights_only=True)["state_dict"])
    yt, pt = predict(model, dl_te, coords, dev)
    res = {"selected": best, "args": vars(a), "test": scores(yt, pt), "ci95": bootstrap_ci(yt, pt)}
    (out / "test.json").write_text(json.dumps(res, indent=2))
    print(f"\nTEST (selected {best['phase']} epoch {best['epoch']}): " + "  ".join(
        f"{k} {v:.4f} [{res['ci95'][k][0]:.3f}, {res['ci95'][k][1]:.3f}]" for k, v in res["test"].items()))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
