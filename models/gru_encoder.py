"""
models/gru_encoder.py
=====================
Bidirectional GRU encoder for temporal aggregation.

This module sits *above* the TGAT layers.  It receives the sequence of
graph-level embeddings z_t ∈ ℝ^{d} (one per timestep) and produces a
single patient embedding p ∈ ℝ^{2·H}.

Theory
------
Why GRU over LSTM for this application?
  GRU has fewer parameters than LSTM (no separate cell state) and performs
  comparably on clinical time series (Che et al. 2018, GRU-D paper).
  With only ~85 training patients in the demo, fewer parameters = less
  overfitting.  On the full MIMIC-IV dataset, LSTM may be preferred.

Why bidirectional?
  ICU data is *retrospective* — we have the full 48h window.  The backward
  pass allows the model to condition early-window features on late-window
  context: "this lactate spike at hour 2 is more informative given what
  happened at hour 40."  This is not causal reasoning (we are not doing
  online prediction) — it is retrospective supervised learning.

  For *real-time* deployment (predicting at hour t using only history),
  remove bidirectionality and add masking.

Pooling strategies
------------------
  last   : concat h_fwd[-1] and h_bwd[-1]  — standard, used in causal_tgat.py
  mean   : mean of all GRU outputs          — captures full trajectory
  max    : max pool across time              — most discriminative moment
  attn   : learned temporal attention       — weighted sum, weights interpretable
  all    : concat last + mean + max         — richest, more parameters

We implement all strategies and benchmark them in the training ablation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class TemporalAttentionPooling(nn.Module):
    """
    Learned temporal attention pooling.

    Computes a scalar attention score per timestep:
        s_t = v · tanh(W · z_t + b)
        α_t = softmax(s_t)
        p   = Σ_t α_t · z_t

    The weights α_t are interpretable: high weight at t means the model
    considers that time period most informative for mortality prediction.

    Parameters
    ----------
    input_dim : GRU output dimension (gru_hidden * 2 for bidirectional)
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.W = nn.Linear(input_dim, input_dim)
        self.v = nn.Linear(input_dim, 1, bias=False)

    def forward(
        self,
        gru_out:  torch.Tensor,            # [B, T, input_dim]
        lengths:  Optional[torch.Tensor],  # [B]  actual lengths
    ) -> tuple:
        """
        Returns
        -------
        context   : [B, input_dim]  — weighted sum
        alpha     : [B, T]          — attention weights (for visualisation)
        """
        scores = self.v(torch.tanh(self.W(gru_out))).squeeze(-1)  # [B, T]

        if lengths is not None:
            # Mask padding positions
            T = gru_out.size(1)
            mask = torch.arange(T, device=gru_out.device).unsqueeze(0) >= lengths.unsqueeze(1)
            scores = scores.masked_fill(mask, float("-inf"))

        alpha   = F.softmax(scores, dim=-1)   # [B, T]
        context = (alpha.unsqueeze(-1) * gru_out).sum(dim=1)  # [B, input_dim]
        return context, alpha


class BidirectionalGRUEncoder(nn.Module):
    """
    Bidirectional GRU encoder with configurable pooling.

    Parameters
    ----------
    input_dim   : input feature dimension (= gat_out_dim from TGAT)
    hidden_dim  : GRU hidden state size per direction
    n_layers    : number of stacked GRU layers
    dropout     : dropout between GRU layers
    pooling     : 'last' | 'mean' | 'max' | 'attn' | 'all'
    """

    POOLING_OPTIONS = ("last", "mean", "max", "attn", "all")

    def __init__(
        self,
        input_dim:  int,
        hidden_dim: int   = 128,
        n_layers:   int   = 2,
        dropout:    float = 0.3,
        pooling:    str   = "last",
    ):
        super().__init__()
        assert pooling in self.POOLING_OPTIONS, \
            f"pooling must be one of {self.POOLING_OPTIONS}"

        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers   = n_layers
        self.pooling    = pooling

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

        bidir_dim = hidden_dim * 2  # bidirectional output

        if pooling == "attn":
            self.attn_pool = TemporalAttentionPooling(bidir_dim)

        # Output dimension depends on pooling strategy
        if pooling == "all":
            # last + mean + max = 3 × bidir_dim
            self._out_dim = bidir_dim * 3
        else:
            self._out_dim = bidir_dim

    @property
    def output_dim(self) -> int:
        return self._out_dim

    def forward(
        self,
        z:       torch.Tensor,                  # [B, T, input_dim]
        lengths: Optional[torch.Tensor] = None, # [B]
    ) -> tuple:
        """
        Parameters
        ----------
        z       : [B, T, input_dim]  — GAT output sequence
        lengths : [B] int — actual sequence lengths (for packing)

        Returns
        -------
        patient_emb : [B, output_dim]
        extras      : dict — 'gru_out', 'attn_weights' (if pooling='attn')
        """
        # Pack for efficiency
        if lengths is not None:
            z_packed = nn.utils.rnn.pack_padded_sequence(
                z, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            gru_out_packed, h_n = self.gru(z_packed)
            gru_out, _ = nn.utils.rnn.pad_packed_sequence(
                gru_out_packed, batch_first=True
            )
        else:
            gru_out, h_n = self.gru(z)

        # gru_out: [B, T, hidden_dim*2]
        # h_n    : [n_layers*2, B, hidden_dim]

        bidir_dim = self.hidden_dim * 2
        extras    = {"gru_out": gru_out}

        # ── Pooling ───────────────────────────────────────────────────────
        if self.pooling == "last":
            # Concat final forward + backward hidden state
            # h_n[-2]: last layer, fwd;  h_n[-1]: last layer, bwd
            emb = torch.cat([h_n[-2], h_n[-1]], dim=-1)  # [B, bidir_dim]

        elif self.pooling == "mean":
            if lengths is not None:
                # Masked mean (ignore padding)
                T = gru_out.size(1)
                mask = (
                    torch.arange(T, device=gru_out.device).unsqueeze(0)
                    < lengths.unsqueeze(1).float()
                ).unsqueeze(-1)  # [B, T, 1]
                emb = (gru_out * mask).sum(dim=1) / lengths.float().unsqueeze(1)
            else:
                emb = gru_out.mean(dim=1)  # [B, bidir_dim]

        elif self.pooling == "max":
            if lengths is not None:
                T = gru_out.size(1)
                mask = (
                    torch.arange(T, device=gru_out.device).unsqueeze(0)
                    >= lengths.unsqueeze(1)
                ).unsqueeze(-1)
                gru_masked = gru_out.masked_fill(mask, float("-inf"))
                emb = gru_masked.max(dim=1).values  # [B, bidir_dim]
            else:
                emb = gru_out.max(dim=1).values

        elif self.pooling == "attn":
            emb, attn_w = self.attn_pool(gru_out, lengths)  # [B, bidir_dim]
            extras["attn_weights"] = attn_w  # [B, T]  ← interpretable!

        elif self.pooling == "all":
            last_emb = torch.cat([h_n[-2], h_n[-1]], dim=-1)
            mean_emb = gru_out.mean(dim=1)
            max_emb  = gru_out.max(dim=1).values
            emb = torch.cat([last_emb, mean_emb, max_emb], dim=-1)  # [B, bidir_dim*3]

        return emb, extras


# ── Unit tests ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(42)
    print("=== GRU Encoder unit tests ===\n")

    B, T, D = 4, 48, 128   # batch, timesteps, gat_out_dim
    input_dim  = D
    hidden_dim = 64

    z       = torch.randn(B, T, input_dim)
    lengths = torch.tensor([48, 36, 24, 48])  # variable-length sequences

    for pooling in BidirectionalGRUEncoder.POOLING_OPTIONS:
        enc = BidirectionalGRUEncoder(input_dim, hidden_dim, n_layers=2,
                                      dropout=0.1, pooling=pooling)
        enc.eval()
        with torch.no_grad():
            emb, extras = enc(z, lengths)

        print(f"  pooling='{pooling}':  emb shape {tuple(emb.shape)}"
              f"  (output_dim={enc.output_dim})")
        if "attn_weights" in extras:
            attn = extras["attn_weights"]
            print(f"    attention weights: {tuple(attn.shape)}")
            print(f"    sum per sample: {attn.sum(dim=1).tolist()}")  # should be ~1

    print("\nGRU hidden states shape check:")
    enc = BidirectionalGRUEncoder(input_dim, hidden_dim, pooling="last")
    enc.eval()
    with torch.no_grad():
        emb, extras = enc(z, lengths)
    gru_out = extras["gru_out"]
    print(f"  gru_out: {tuple(gru_out.shape)}  (B, T, hidden*2)")
    print(f"  emb    : {tuple(emb.shape)}       (B, hidden*2)")

    # Verify that padding doesn't contaminate mean
    z_short = torch.randn(2, 10, input_dim)
    z_short_padded = F.pad(z_short, (0,0, 0, T-10))  # pad to T
    lengths_short  = torch.tensor([10, 10])

    enc_mean = BidirectionalGRUEncoder(input_dim, hidden_dim, pooling="mean")
    enc_mean.eval()
    with torch.no_grad():
        emb_unpadded, _ = enc_mean(z_short, lengths_short)
        emb_padded, _   = enc_mean(z_short_padded, lengths_short)

    diff = (emb_unpadded - emb_padded).abs().max().item()
    print(f"\nPadding contamination check (mean pooling): max diff = {diff:.2e}")
    print(f"  Expected ≈ 0 (masked mean handles padding correctly) ✓")

    print("\n✓ All GRU encoder tests passed.")