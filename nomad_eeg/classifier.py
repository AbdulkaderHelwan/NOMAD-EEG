"""Fine-tuning head used in the paper (TUEV, TUAB, TUSZ, Mumtaz, EEGMMIDB, Sleep-EDF, SEED-V).

The gated activations of the four feature blocks are pooled to `n_win` time windows per
electrode, projected to `d` dimensions, and read out by one learned attention query over
all (electrode, window) tokens, concatenated with their mean. Because the head pools over
electrodes, it works with any montage, like the backbone.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .model import SplitSSM, timestep_embedding


class Classifier(nn.Module):
    def __init__(self, enc: SplitSSM, n_classes=6, d=256, n_win=25, dropout=0.1):
        super().__init__()
        self.enc, self.n_win = enc, n_win
        self.fb = tuple(enc.cfg.feature_blocks)
        D = enc.cfg.d_model
        self.proj = nn.Sequential(nn.Linear(len(self.fb) * D, d), nn.GELU(), nn.LayerNorm(d))
        self.tpos = nn.Parameter(torch.zeros(n_win, d))
        self.query = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.attn = nn.MultiheadAttention(d, 4, batch_first=True)
        self.out = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * d, n_classes))

    def gates(self, x, coords, corpus, grad_ckpt):
        m = self.enc
        b, e, tt = x.shape
        h = m.stem(x.reshape(b * e, 1, tt)).transpose(1, 2)
        h = h.reshape(b, e, h.size(1), -1)
        # the clean end of the trajectory, as in features(): t=1 flow, t=0 diffusion
        t = torch.ones(b, device=x.device) if m.cfg.objective == "flow" \
            else torch.zeros(b, device=x.device)
        cond = m.time_mlp(timestep_embedding(t, m.cfg.d_model)) + m.corpus_emb(corpus)
        out = []
        for i, blk in enumerate(m.blocks):
            if i > max(self.fb):
                break
            if grad_ckpt and torch.is_grad_enabled():
                h, _, u = checkpoint(blk, h, cond, coords, True, use_reentrant=False)
            else:
                h, _, u = blk(h, cond, coords, return_gate=True)
            if i in self.fb:
                out.append(u)
        return torch.cat(out, dim=-1)                                  # [B, E, T', 4D]

    def forward(self, x, coords, corpus, train_encoder: bool, grad_ckpt=True):
        with torch.set_grad_enabled(train_encoder and torch.is_grad_enabled()):
            g = self.gates(x, coords, corpus, grad_ckpt)
        b, e, tt, c = g.shape
        g = F.adaptive_avg_pool1d(g.permute(0, 1, 3, 2).reshape(b * e, c, tt), self.n_win)
        g = g.reshape(b, e, c, self.n_win).permute(0, 1, 3, 2)          # [B, E, W, C]
        z = self.proj(g.float()) + self.tpos                            # [B, E, W, d]
        z = z.reshape(b, e * self.n_win, -1)
        pooled, _ = self.attn(self.query.expand(b, -1, -1), z, z)
        return self.out(torch.cat([pooled[:, 0], z.mean(1)], dim=-1))


