"""Frozen NOMAD-EEG embeddings, as used for every linear probe in the paper."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from .hub import CORPUS_ID
from .model import SplitSSM, channel_coords


@torch.no_grad()
def extract_features(model: SplitSSM, x: np.ndarray, channels: Sequence[str],
                     batch_size: int = 64, n_windows: int = 5, corpus_id: int = CORPUS_ID,
                     device: str | torch.device | None = None) -> np.ndarray:
    """x: preprocessed EEG [N, E, T] at 200 Hz, channels: E electrode names.

    Returns [N, n_windows * 2048] float32 features (10,240 for the paper's 5 windows).
    On a GPU the forward pass runs in bfloat16 autocast, as in the paper.
    """
    device = torch.device(device or next(model.parameters()).device)
    model = model.to(device).eval()
    coords = channel_coords(list(channels)).to(device)
    out = []
    for i in range(0, len(x), batch_size):
        xb = torch.from_numpy(np.asarray(x[i:i + batch_size], np.float32)).to(device)
        cid = torch.full((xb.size(0),), corpus_id, dtype=torch.long, device=device)
        with torch.autocast("cuda", torch.bfloat16, enabled=device.type == "cuda"):
            f = model.features(xb, coords, cid, n_windows=n_windows)
        out.append(f.float().flatten(1).cpu().numpy())
    return np.concatenate(out)
