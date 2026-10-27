"""Loading the released NOMAD-EEG weights."""
from __future__ import annotations

from pathlib import Path

import torch

from .model import SplitSSM, SplitSSMConfig

# Direct-download link of the released weights (set when the file is published).
WEIGHTS_URL = "TODO: OneDrive direct-download link"
WEIGHTS_FILE = "nomad_eeg_base.pt"

# Corpus embedding learned in pre-training (TUEV). Every downstream result in the paper
# conditions on it; fine-tuning updates it along with the rest of the network.
CORPUS_ID = 3


def _read(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        # the full training checkpoint also stores optimiser state and run statistics
        return torch.load(path, map_location="cpu", weights_only=False)


def load_pretrained(path: str | Path | None = None, device: str | torch.device = "cpu",
                    url: str | None = None) -> SplitSSM:
    """Return the pre-trained backbone in eval mode.

    `path` is a local copy of the weights. Without it the file is downloaded once from
    `url` (default WEIGHTS_URL) and cached by torch.hub (~/.cache/torch/hub/checkpoints).
    Both the release file (keys cfg, state_dict) and the full training checkpoint
    (keys cfg, ema, model, ...) are accepted; from the latter the EMA weights are used,
    as for every result in the paper.
    """
    if path is None:
        url = url or WEIGHTS_URL
        if not url.startswith("http"):
            raise RuntimeError("No download link configured: download the weights manually "
                               "and pass load_pretrained(path=...)")
        dst = Path(torch.hub.get_dir()) / "checkpoints" / WEIGHTS_FILE
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            torch.hub.download_url_to_file(url, str(dst), progress=True)
        path = dst
    ck = _read(Path(path))
    sd = ck["state_dict"] if "state_dict" in ck else ck["ema"]
    cfg = dict(ck["cfg"])
    cfg["feature_blocks"] = tuple(cfg["feature_blocks"])
    model = SplitSSM(SplitSSMConfig(**cfg))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # the diffusion schedule (alpha_bar) is a buffer rebuilt by the constructor
    assert not unexpected and all("alpha_bar" in m for m in missing), (missing, unexpected)
    return model.to(device).eval()
