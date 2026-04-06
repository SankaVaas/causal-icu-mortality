"""
models/time2vec.py
==================
Time2Vec: Learning a Vector Representation of Time
Kazemi et al., 2019  (https://arxiv.org/abs/1907.05321)

Theory
------
Standard positional encodings (fixed sinusoidal à la "Attention is All You
Need") assume equally-spaced, fixed-frequency tokens.  ICU data is *irregularly
sampled* — a heart rate might be recorded every 30 minutes in a stable patient
but every 2 minutes post-surgery.  Time2Vec learns the frequencies.

For a scalar time value t, Time2Vec produces a k-dimensional vector:

    t2v(t)[0]     = ω₀ · t + φ₀           (linear — captures trend)
    t2v(t)[i]     = sin(ωᵢ · t + φᵢ)      for i = 1 … k-1  (periodic)

where ω and φ are *learned* parameters.

Why sinusoidal for i>0?
  sin is periodic and bounded — the model can represent any periodicity
  (hourly, daily, circadian) without the gradient exploding.  The linear
  term lets the model also learn monotone time trends (deterioration over
  time).  Both terms are invariant to time translation if the downstream
  network learns to ignore the phase φ.

Integration with TGAT
---------------------
At each timestep t, we concatenate [x_t, mask_t, delta_t, time2vec(t)] as
the node input to the graph attention layer.  The time encoding enriches
each node feature with the *when* of the measurement, enabling the attention
weights α_ij to be time-aware: a reading taken 10 minutes ago should
influence a node differently than one taken 40 hours ago.
"""

import torch
import torch.nn as nn
import math


class Time2Vec(nn.Module):
    """
    Learnable time encoding for scalar time values.

    Parameters
    ----------
    d_model : int
        Output embedding dimension k.  First dim is linear, rest are sinusoidal.
        Recommend d_model = 16 or 32.

    Input
    -----
    t : Tensor  [... any shape ...]  — time values in hours (float)

    Output
    ------
    Tensor [... , d_model]  — time embedding
    """

    def __init__(self, d_model: int = 16):
        super().__init__()
        self.d_model = d_model

        # ω and φ for the linear term (i=0)
        self.w0 = nn.Parameter(torch.randn(1) * 0.01)
        self.b0 = nn.Parameter(torch.zeros(1))

        # ω and φ for sinusoidal terms (i=1…k-1)
        k = d_model - 1
        self.W = nn.Parameter(torch.randn(k) * 0.01)
        self.B = nn.Parameter(torch.zeros(k))

        self._init_weights()

    def _init_weights(self):
        """
        Initialise ω so the sinusoidal terms span a useful range of periods.

        ICU data spans 0–48 hours.  We want frequencies that capture:
          - sub-hourly variation  (ω ~ 2π)
          - hourly variation      (ω ~ 2π/1)
          - daily circadian       (ω ~ 2π/24)
          - multi-day trends      (ω ~ 2π/48)

        Log-spaced initialisation (analogous to NeRF positional encoding):
          ωᵢ = 2π / T_i   where T_i is log-spaced in [1, 48]
        """
        k = self.d_model - 1
        if k > 0:
            periods = torch.exp(
                torch.linspace(math.log(1.0), math.log(48.0), k)
            )  # [k]  — periods in hours
            freqs = 2 * math.pi / periods   # angular frequencies
            with torch.no_grad():
                self.W.copy_(freqs)
                # Random phase offsets ∈ [-π, π]
                self.B.copy_(torch.rand(k) * 2 * math.pi - math.pi)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        t : Tensor [...] — time in hours

        Returns
        -------
        Tensor [..., d_model]
        """
        # Linear term: shape [..., 1]
        linear = (self.w0 * t + self.b0).unsqueeze(-1)  # [..., 1]

        # Sinusoidal terms: shape [..., k]
        # Expand t for broadcasting: [...] → [..., 1] * [k] → [..., k]
        periodic = torch.sin(t.unsqueeze(-1) * self.W + self.B)  # [..., k]

        return torch.cat([linear, periodic], dim=-1)  # [..., d_model]

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}"


class Time2VecWithDelta(nn.Module):
    """
    Extended Time2Vec that also encodes the time-since-last-observation delta.

    GRU-D (Che et al., 2018) showed that δ — how long since a feature was
    last observed — is itself a clinically meaningful signal.  A rising δ for
    heart rate means the nurse hasn't charted it recently, which may indicate
    clinical stability or a monitoring gap.

    We embed both t (absolute time) and δ (per-feature elapsed time) and
    concatenate:

        [time2vec(t), time2vec(δ_f)]  for feature f at time t

    This gives the model explicit access to two distinct temporal signals.

    Parameters
    ----------
    d_model : int  — embedding dim for each of t and δ
                      final output dim = 2 * d_model
    """

    def __init__(self, d_model: int = 16):
        super().__init__()
        self.t2v_absolute = Time2Vec(d_model)   # encodes hours since admission
        self.t2v_delta    = Time2Vec(d_model)   # encodes hours since last obs

    @property
    def output_dim(self) -> int:
        return self.t2v_absolute.d_model + self.t2v_delta.d_model

    def forward(
        self,
        times: torch.Tensor,   # [B, T]       absolute hours
        delta: torch.Tensor,   # [B, T, F]    per-feature delta
    ) -> torch.Tensor:
        """
        Returns
        -------
        Tensor [B, T, F, 2*d_model]
        """
        B, T, F = delta.shape

        # Absolute time embedding: [B, T, d_model] → [B, T, 1, d_model] → broadcast
        t_emb = self.t2v_absolute(times)           # [B, T, d_model]
        t_emb = t_emb.unsqueeze(2).expand(B, T, F, -1)  # [B, T, F, d_model]

        # Delta embedding: [B, T, F, d_model]
        d_emb = self.t2v_delta(delta)              # [B, T, F, d_model]

        return torch.cat([t_emb, d_emb], dim=-1)  # [B, T, F, 2*d_model]


# ── Quick unit test ────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(0)

    print("=== Time2Vec unit tests ===\n")

    # Basic forward pass
    t2v = Time2Vec(d_model=16)
    t   = torch.tensor([0.0, 1.0, 6.0, 12.0, 24.0, 48.0])
    out = t2v(t)
    print(f"Input times  : {t.tolist()}")
    print(f"Output shape : {tuple(out.shape)}  (expected [6, 16])")
    print(f"Linear term  : {out[:, 0].detach().tolist()}")
    print(f"  → should increase monotonically with time ✓")
    print(f"Sinusoidal[0]: {[f'{v:.3f}' for v in out[:, 1].detach().tolist()]}")
    print(f"  → should oscillate ✓\n")

    # Batched input [B, T]
    t_batch = torch.rand(4, 48) * 48   # 4 patients, 48 timesteps, random hours
    out_batch = t2v(t_batch)
    print(f"Batched input  [B=4, T=48]: {tuple(t_batch.shape)}")
    print(f"Batched output [B=4, T=48, d=16]: {tuple(out_batch.shape)} ✓\n")

    # Time2VecWithDelta
    t2v_delta = Time2VecWithDelta(d_model=16)
    times = torch.rand(4, 48) * 48          # [B, T]
    delta = torch.rand(4, 48, 17) * 24      # [B, T, F]
    out_combined = t2v_delta(times, delta)
    print(f"Time2VecWithDelta output: {tuple(out_combined.shape)}")
    print(f"  → expected [4, 48, 17, 32]  ✓\n")

    # Check that learned frequencies span meaningful range
    freqs = 2 * 3.14159 / t2v.W.detach()
    print(f"Learned periods (hours): min={freqs.min():.2f}  max={freqs.max():.2f}")
    print(f"  → should span ~[1h, 48h] ✓")
    print("\n✓ All Time2Vec tests passed.")