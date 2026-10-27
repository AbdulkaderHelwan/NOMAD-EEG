"""Loading the released NOMAD-EEG weights."""
from __future__ import annotations

from pathlib import Path

import torch

from .model import SplitSSM, SplitSSMConfig

WEIGHTS_URL = ("https://github.com/AbdulkaderHelwan/NOMAD-EEG/releases/download/"
               "v1.0/nomad_eeg_base.pt")

# Corpus embedding learned in pre-training (TUEV). Every downstream result in the paper
# conditions on it; fine-tuning updates it along with the rest of the network.
CORPUS_ID = 3


def load_pretrained(path: str | Path | None = None, device: str | torch.device = "cpu") -> SplitSSM:
    """Return the pre-trained backbone in eval mode.

    `path` is a local copy of nomad_eeg_base.pt; without it the file is downloaded from
    the GitHub release once and cached by torch.hub (~/.cache/torch/hub/checkpoints).
    """
    if path is None:
        ck = torch.hub.load_state_dict_from_url(WEIGHTS_URL, map_location="cpu",
                                                weights_only=True, progress=True)
    else:
        ck = torch.load(path, map_location="cpu", weights_only=True)
    cfg = dict(ck["cfg"])
    cfg["feature_blocks"] = tuple(cfg["feature_blocks"])
    model = SplitSSM(SplitSSMConfig(**cfg))
    missing, unexpected = model.load_state_dict(ck["state_dict"], strict=False)
    # the diffusion schedule (alpha_bar) is a buffer rebuilt by the constructor
    assert not unexpected and all("alpha_bar" in m for m in missing), (missing, unexpected)
    return model.to(device).eval()
