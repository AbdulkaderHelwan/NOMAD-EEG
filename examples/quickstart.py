"""Embed EEG recordings with three different montages using the same pre-trained weights.

    python examples/quickstart.py path/to/nomad_eeg_base.pt

Random signals stand in for real data here; for real recordings, preprocess them first
(nomad_eeg.preprocess + nomad_eeg.RobustScaler, or scripts/prepare_dataset.py).
"""
import sys

import numpy as np
import torch

import nomad_eeg

dev = "cuda" if torch.cuda.is_available() else "cpu"
model = nomad_eeg.load_pretrained(sys.argv[1] if len(sys.argv) > 1 else None, device=dev)
print(f"NOMAD-EEG: {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M parameters")

montages = {
    "19-electrode 10-20 cap": ["FP1", "FP2", "F7", "F3", "FZ", "F4", "F8", "T3", "C3", "CZ",
                               "C4", "T4", "T5", "P3", "PZ", "P4", "T6", "O1", "O2"],
    "8-electrode BCI cap": ["FZ", "C3", "CZ", "C4", "P3", "PZ", "P4", "OZ"],
    "2 bipolar sleep channels": ["FPZ-CZ", "PZ-OZ"],
}
rng = np.random.default_rng(0)
for name, chans in montages.items():
    x = rng.standard_normal((4, len(chans), 5 * nomad_eeg.FS)).astype(np.float32)   # 4 windows of 5 s
    f = nomad_eeg.extract_features(model, x, chans)
    print(f"{name:<26} input {x.shape} -> features {f.shape}")
