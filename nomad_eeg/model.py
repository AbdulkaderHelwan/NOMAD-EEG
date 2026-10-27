"""NOMAD-EEG backbone (SplitSSM): a montage-agnostic diffusion network for EEG.

Input: a signal x of shape [B, E, T] (E electrodes in any order, T samples at 200 Hz)
and the 3D position of every electrode, coords [E, 3] (see `channel_coords`). Each block
mixes time with a bidirectional S4D state-space layer (no patches) and electrodes with
attention whose only electrode identity is a Fourier embedding of the 3D position.

`SplitSSM.features()` gives the frozen embedding used in the paper: the gated activations
of blocks 3, 7, 11 and 15, averaged over electrodes, then mean and standard deviation over
`n_windows` equal time windows (5 windows -> 4 x 2 x 256 x 5 = 10,240 dimensions).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

# 21 channels: ROC is dropped. It has no standard 10-20 coordinate and is absent
# from 40% of TUEV samples, so it would teach the model that zeros are signal.
CHANNELS_21 = ["FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
               "F7", "F8", "T3", "T4", "T5", "T6", "A1", "A2", "FZ", "CZ", "PZ"]


def channel_coords(names=CHANNELS_21, ref=CHANNELS_21) -> torch.Tensor:
    """3D electrode positions, metres -> the unit scale used in pre-training.

    Names are looked up in MNE's standard 10-20 montage, then in the 10-05 montage for
    denser caps (e.g. FC5 on a 64-channel cap). A bipolar derivation written "A-B"
    (Sleep-EDF's Fpz-Cz) is placed at the midpoint of its two electrodes, as REVE does.

    The scale is fixed by the reference set (the 21 pre-training channels), not by the
    names passed in, so an electrode gets the same coordinate in a 2-, 19- or 64-channel
    recording. Channels without a standard position (ECG, EMG, ...) raise KeyError and
    should be dropped before calling the model.
    """
    import mne
    import numpy as np
    table = {}
    for m in ("standard_1005", "standard_1020"):          # 10-20 last: it wins on overlap
        pos = mne.channels.make_standard_montage(m).get_positions()["ch_pos"]
        table.update({k.upper(): np.asarray(v, dtype=np.float32) for k, v in pos.items()})

    def one(c):
        c = c.upper()
        if c in table:
            return table[c]
        if "-" in c:
            a, b = c.split("-", 1)
            if a in table and b in table:
                return (table[a] + table[b]) / 2
        raise KeyError(f"no 10-20/10-05 coordinate for {c}")

    xyz = torch.from_numpy(np.stack([one(c) for c in names]))
    ref_xyz = torch.from_numpy(np.stack([one(c) for c in ref]))
    return xyz / ref_xyz.norm(dim=-1).max()


# ----------------------------------------------------------------------------- S4D
class S4D(nn.Module):
    """Diagonal state-space layer, bidirectional, applied along time via FFT.

    Pure PyTorch (no pykeops or custom CUDA kernels).

    Kernel for one feature channel h with state dimension N:
        K[l] = 2 * Re( sum_n  C_n * (exp(dt*A_n) - 1)/A_n * exp(dt*A_n)^l )
    which is the standard S4D parameterisation (diagonal A, exponential kernel).
    """

    def __init__(self, d_model: int, d_state: int = 64, bidirectional: bool = True,
                 dt_min: float = 1e-3, dt_max: float = 1e-1):
        super().__init__()
        self.d_model, self.n, self.bi = d_model, d_state // 2, bidirectional
        k = 2 if bidirectional else 1

        log_dt = torch.rand(d_model) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)

        # S4D-Lin initialisation: A = -1/2 + i*pi*n
        log_A_real = torch.log(0.5 * torch.ones(k, d_model, self.n))
        A_imag = math.pi * torch.arange(self.n).repeat(k, d_model, 1)
        self.log_A_real = nn.Parameter(log_A_real)
        self.A_imag = nn.Parameter(A_imag.float())
        self.C = nn.Parameter(torch.randn(k, d_model, self.n, 2) * (0.5 ** 0.5))
        self.D = nn.Parameter(torch.randn(d_model))

    def kernel(self, L: int) -> torch.Tensor:
        dt = self.log_dt.exp()                                   # (H,)
        A = -self.log_A_real.exp() + 1j * self.A_imag            # (k,H,N)
        C = torch.view_as_complex(self.C)                        # (k,H,N)
        dtA = A * dt[None, :, None]                              # (k,H,N)
        # (exp(dtA)-1)/A  is the zero-order-hold discretisation of B (B folded into C)
        Cb = C * (torch.exp(dtA) - 1.0) / A
        vand = torch.exp(dtA[..., None] * torch.arange(L, device=dt.device))   # (k,H,N,L)
        return 2.0 * torch.einsum("khn,khnl->khl", Cb, vand).real              # (k,H,L)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """u: [B, H, L] -> [B, H, L]"""
        L = u.size(-1)
        k = self.kernel(L)
        if self.bi:
            kf, kb = k[0], k[1]
            k = F.pad(kf, (0, L)) + F.pad(kb.flip(-1), (L, 0))   # (H, 2L) non-causal
            n_fft = 2 * L
        else:
            k, n_fft = k[0], 2 * L
        # FFT in at least float32: bf16/fp16 autocast would otherwise lose the kernel
        # tail, but do not downcast float64 (the precision tests rely on that).
        fdt = torch.float32 if u.dtype in (torch.float16, torch.bfloat16) else u.dtype
        K = torch.fft.rfft(k.to(fdt), n=n_fft)
        U = torch.fft.rfft(u.to(fdt), n=n_fft)
        y = torch.fft.irfft(U * K, n=n_fft).to(u.dtype)
        y = y[..., L:2 * L] if self.bi else y[..., :L]
        return y + u * self.D[None, :, None]


# ------------------------------------------------------------------- electrode mixer
class ElectrodeAttention(nn.Module):
    """Multi-head attention across electrode tokens at each timepoint.

    Permutation-invariant by construction; electrode identity enters only through
    Fourier features of the 3D coordinate, so a corpus with a different montage (or
    a subset of electrodes) is simply a different set of tokens.
    """

    def __init__(self, d_model: int, n_heads: int = 4, n_freq: int = 8):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.n_freq = n_freq
        self.coord_proj = nn.Linear(3 * 2 * n_freq, d_model)

    def coord_features(self, coords: torch.Tensor) -> torch.Tensor:
        """coords: [E, 3] -> [E, d_model]; multi-frequency Fourier features (REVE)."""
        freqs = 2.0 ** torch.arange(self.n_freq, device=coords.device) * math.pi
        ang = coords[:, :, None] * freqs[None, None, :]          # [E,3,F]
        feats = torch.cat([ang.sin(), ang.cos()], dim=-1).flatten(1)
        return self.coord_proj(feats)

    def forward(self, h: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """h: [B, E, T, D] -> [B, E, T, D]"""
        b, e, t, d = h.shape
        x = h.permute(0, 2, 1, 3).reshape(b * t, e, d)           # electrodes as tokens
        x = x + self.coord_features(coords)[None]
        # Every (sample, timestep) is its own attention problem, so the kernel launches
        # b*t*heads blocks. CUDA caps a grid dimension at 65,535: Sleep-EDF (30 s, batch
        # 32 -> 48,000 sequences x 4 heads) crashed with "invalid configuration argument".
        # Chunking is exact -- the sequences are independent -- and a no-op below the cap.
        step = max(1, 65535 // self.attn.num_heads)
        if x.size(0) <= step:
            out, _ = self.attn(x, x, x, need_weights=False)
        else:
            out = torch.cat([self.attn(c, c, c, need_weights=False)[0]
                             for c in x.split(step, dim=0)])
        return out.view(b, t, e, d).permute(0, 2, 1, 3)


# -------------------------------------------------------------------------- the block
class SplitSSMBlock(nn.Module):
    def __init__(self, d_model: int, d_cond: int, d_state: int = 64, n_heads: int = 4,
                 dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.film = nn.Linear(d_cond, 2 * d_model)
        nn.init.zeros_(self.film.weight); nn.init.zeros_(self.film.bias)
        self.ssm = S4D(d_model, d_state)
        self.elec = ElectrodeAttention(d_model, n_heads)
        self.gate_proj = nn.Linear(d_model, 2 * d_model)
        self.res_proj = nn.Linear(d_model, d_model)
        self.skip_proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, h, cond, coords, return_gate: bool = False):
        b, e, t, d = h.shape
        z = self.norm(h)
        scale, shift = self.film(cond)[:, None, None, :].chunk(2, dim=-1)
        z = z * (1 + scale) + shift

        z = self.ssm(z.reshape(b * e, t, d).transpose(1, 2)).transpose(1, 2).view(b, e, t, d)
        z = z + self.elec(z, coords)

        g, f = self.gate_proj(self.drop(z)).chunk(2, dim=-1)
        u = torch.sigmoid(g) * torch.tanh(f)                     # features read here
        out = h + self.res_proj(u), self.skip_proj(u)
        return (*out, u) if return_gate else out


# -------------------------------------------------------------------------- the model
@dataclass
class SplitSSMConfig:
    d_model: int = 256
    n_blocks: int = 16
    d_state: int = 64
    n_heads: int = 4
    d_cond: int = 256
    stem_stride: int = 4          # the ONLY temporal compression in the model
    stem_kernel: int = 9
    n_corpora: int = 16
    dropout: float = 0.0
    objective: str = "flow"       # "flow" (rectified flow) or "diffusion" (eps-pred)
    n_diffusion_steps: int = 1000
    feature_blocks: tuple = field(default_factory=lambda: (3, 7, 11, 15))
    # Recompute each block's activations in the backward pass instead of storing them.
    # At d=256, 16 blocks, 1,500 steps x 21 electrodes, the S4D FFT intermediates are
    # ~0.5 GB per tensor per block at batch 8; storing all 16 blocks' worth does not
    # fit a 40 GB A100. Costs ~30% extra compute.
    grad_ckpt: bool = True


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    a = t.float()[:, None] * freqs[None] * 1000.0
    return torch.cat([a.sin(), a.cos()], dim=-1)


class SplitSSM(nn.Module):
    def __init__(self, cfg: SplitSSMConfig = SplitSSMConfig()):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.stem = nn.Conv1d(1, d, cfg.stem_kernel, stride=cfg.stem_stride,
                              padding=cfg.stem_kernel // 2)
        self.time_mlp = nn.Sequential(
            nn.Linear(d, cfg.d_cond), nn.SiLU(), nn.Linear(cfg.d_cond, cfg.d_cond))
        self.corpus_emb = nn.Embedding(cfg.n_corpora, cfg.d_cond)
        self.blocks = nn.ModuleList([
            SplitSSMBlock(d, cfg.d_cond, cfg.d_state, cfg.n_heads, cfg.dropout)
            for _ in range(cfg.n_blocks)])
        self.head = nn.Sequential(nn.LayerNorm(d), nn.SiLU(), nn.Linear(d, d))
        self.out = nn.ConvTranspose1d(d, 1, cfg.stem_stride * 2,
                                      stride=cfg.stem_stride, padding=cfg.stem_stride // 2)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

        if cfg.objective == "diffusion":                     # cosine schedule (Nichol & Dhariwal)
            s, T = 0.008, cfg.n_diffusion_steps
            ts = torch.arange(T + 1, dtype=torch.float64) / T
            ab = torch.cos((ts + s) / (1 + s) * math.pi / 2) ** 2
            self.register_buffer("alpha_bar", (ab / ab[0])[:-1].float())

    # ------------------------------------------------------------------ forward
    def forward(self, x, t, coords, corpus=None, return_gates=False):
        """x: [B, E, T] noisy/interpolated signal. t: [B] in [0,1]. coords: [E,3]."""
        b, e, tt = x.shape
        h = self.stem(x.reshape(b * e, 1, tt)).transpose(1, 2)          # [B*E, T', D]
        h = h.view(b, e, h.size(1), -1)

        cond = self.time_mlp(timestep_embedding(t, self.cfg.d_model))
        if corpus is not None:
            cond = cond + self.corpus_emb(corpus)

        skips, gates = 0.0, []
        for i, blk in enumerate(self.blocks):
            want = return_gates and i in self.cfg.feature_blocks
            if want:
                h, s, g = blk(h, cond, coords, return_gate=True)
                gates.append(g)
            elif self.cfg.grad_ckpt and self.training and torch.is_grad_enabled():
                h, s = torch.utils.checkpoint.checkpoint(blk, h, cond, coords,
                                                         use_reentrant=False)
            else:
                h, s = blk(h, cond, coords)
            skips = skips + s

        y = self.head(skips)
        bt = y.size(2)
        y = self.out(y.reshape(b * e, bt, -1).transpose(1, 2)).squeeze(1)
        y = F.interpolate(y[:, None], size=tt, mode="linear", align_corners=False)
        y = y.view(b, e, tt)
        return (y, gates) if return_gates else y

    # ---------------------------------------------------------------- objectives
    def loss(self, x0, coords, corpus=None):
        """One scalar loss. The objective is the only thing that differs between arms."""
        b = x0.size(0)
        noise = torch.randn_like(x0)
        if self.cfg.objective == "flow":
            # Rectified flow: straight path from noise (t=0) to data (t=1).
            t = torch.rand(b, device=x0.device)
            tt = t[:, None, None]
            xt = (1 - tt) * noise + tt * x0
            target = x0 - noise                                  # constant velocity
        elif self.cfg.objective == "diffusion":
            idx = torch.randint(0, self.cfg.n_diffusion_steps, (b,), device=x0.device)
            ab = self.alpha_bar[idx][:, None, None]
            xt = ab.sqrt() * x0 + (1 - ab).sqrt() * noise
            target = noise                                       # eps-prediction
            t = idx.float() / self.cfg.n_diffusion_steps
        else:
            raise ValueError(self.cfg.objective)
        return F.mse_loss(self(xt, t, coords, corpus), target)

    # ------------------------------------------------------------------ features
    @torch.no_grad()
    def features(self, x, coords, corpus=None, n_windows: int = 8):
        """Downstream readout: clean input, near-zero time, windowed mean+std of gates.

        Returns [B, n_windows, n_feature_blocks * 2 * D]. No noise is added and the
        generative head is not used -- see the module docstring for why the pooling
        is windowed rather than global.
        """
        b = x.size(0)
        # t=1 is "data" under rectified flow and near-clean under the cosine schedule;
        # either way this is the clean end of the trajectory.
        t = torch.ones(b, device=x.device) if self.cfg.objective == "flow" \
            else torch.zeros(b, device=x.device)
        _, gates = self(x, t, coords, corpus, return_gates=True)
        feats = []
        for g in gates:                                          # [B, E, T', D]
            gm = g.mean(dim=1)                                   # pool electrodes
            tt = gm.size(1)
            w = tt // n_windows
            gw = gm[:, :w * n_windows].view(b, n_windows, w, -1)
            feats.append(torch.cat([gw.mean(2), gw.std(2)], dim=-1))
        return torch.cat(feats, dim=-1)


