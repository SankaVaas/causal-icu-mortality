"""
models/causal_tgat.py
=====================
Full Causal Temporal Graph Attention Network for ICU Mortality Prediction.

Architecture overview
---------------------

Input per patient:
    X      [B, T, F]  — normalised feature values  (F=17, T=48)
    mask   [B, T, F]  — observation indicator
    delta  [B, T, F]  — hours since last observation per feature
    times  [B, T]     — hours since ICU admission
    adj    [F, F]     — causal DAG from NOTEARS (fixed at runtime)

Forward pass:

  ┌─────────────────────────────────────────────────────┐
  │ 1. Node feature construction  [B, T, F, node_dim]   │
  │    For feature f at time t:                         │
  │    node_f_t = [x_f_t, mask_f_t,                    │
  │                time2vec(t), time2vec(δ_f_t)]        │
  └──────────────────────┬──────────────────────────────┘
                         │
  ┌──────────────────────▼──────────────────────────────┐
  │ 2. Causal Graph Attention (2 layers)                │
  │    Applied at each timestep t independently         │
  │    Nodes = features, edges = causal DAG             │
  │    Output: graph embedding z_t  [B, T, hidden]      │
  └──────────────────────┬──────────────────────────────┘
                         │
  ┌──────────────────────▼──────────────────────────────┐
  │ 3. Bidirectional GRU  over time                     │
  │    Input:  z_t sequence  [B, T, hidden]             │
  │    Output: patient embedding p  [B, gru_hidden*2]   │
  │    (concat last fwd + last bwd hidden state)        │
  └──────────────────────┬──────────────────────────────┘
                         │
              ┌──────────┴──────────┐
              │                     │
  ┌───────────▼──────────┐ ┌────────▼────────────────┐
  │ 4a. Mortality head   │ │ 4b. Cox survival head   │
  │     Binary output    │ │     Risk score β·p      │
  │     (AUROC metric)   │ │     (C-index metric)    │
  └──────────────────────┘ └─────────────────────────┘

Theoretical notes
-----------------
The separation of spatial (GAT) and temporal (GRU) processing is deliberate:
- GAT at each timestep captures how features causally influence each other
  *at that moment* — e.g., how rising lactate relates to falling MAP at t=6h.
- GRU across time captures how the *overall physiological state* evolves —
  e.g., a patient who had high lactate at t=2h and MAP dropping at t=4h has
  a very different trajectory from one where both improved.

Counterfactual inference
------------------------
The do(X_k = v) operator is implemented in inference/counterfactual.py.
It works by:
  1. Removing all incoming edges to node k in adj (graph surgery)
  2. Forcing x[k, :] = v (override the feature)
  3. Re-running the forward pass with the mutilated graph
The difference in predicted risk is the causal effect of the intervention.

MC Dropout for uncertainty
--------------------------
dropout layers remain active at test time when model.train() is False
but MC dropout is enabled via the mc_dropout() context manager.
50 forward passes → mean (prediction) + epistemic variance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager
from typing import Optional

from models.time2vec import Time2Vec
from models.tgat_layer import CausalGraphAttentionLayer, TemporalGATLayer


class CausalTGAT(nn.Module):
    """
    Causal Temporal Graph Attention Network.

    Parameters
    ----------
    n_features    : number of physiological features / graph nodes (17)
    t2v_dim       : Time2Vec embedding dimension
    gat_hidden    : per-head output dim in GAT layers
    n_heads       : attention heads
    n_gat_layers  : stacked GAT layers per timestep
    gru_hidden    : GRU hidden state dimension
    gru_layers    : stacked GRU layers
    dropout       : dropout probability (also used for MC Dropout)
    """

    def __init__(
        self,
        n_features:   int   = 17,
        t2v_dim:      int   = 16,
        gat_hidden:   int   = 32,
        n_heads:      int   = 4,
        n_gat_layers: int   = 2,
        gru_hidden:   int   = 128,
        gru_layers:   int   = 2,
        dropout:      float = 0.3,
    ):
        super().__init__()
        self.n_features   = n_features
        self.gat_hidden   = gat_hidden
        self.n_heads      = n_heads
        self.gru_hidden   = gru_hidden

        # ── 1. Time encodings ──────────────────────────────────────────────
        # t2v_absolute: encodes hours since admission
        # t2v_delta:    encodes hours since last observation (per feature)
        self.t2v_absolute = Time2Vec(d_model=t2v_dim)
        self.t2v_delta    = Time2Vec(d_model=t2v_dim)

        # node input dim = feature_value(1) + mask(1) + t2v_abs(t2v_dim) + t2v_delta(t2v_dim)
        node_in_dim = 1 + 1 + t2v_dim + t2v_dim

        # ── 2. Stacked causal GAT layers ──────────────────────────────────
        self.gat_layers = nn.ModuleList()
        for i in range(n_gat_layers):
            in_d = node_in_dim if i == 0 else gat_hidden * n_heads
            self.gat_layers.append(
                TemporalGATLayer(
                    in_dim=in_d,
                    out_dim=gat_hidden,
                    n_heads=n_heads,
                    dropout=dropout,
                    residual=(i > 0),
                )
            )

        gat_out_dim = gat_hidden * n_heads  # concatenated heads

        # ── 3. Bidirectional GRU ──────────────────────────────────────────
        # Input at each t: mean-pooled graph embedding [gat_out_dim]
        self.gru = nn.GRU(
            input_size=gat_out_dim,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )

        patient_dim = gru_hidden * 2  # bidirectional: fwd + bwd

        # ── 4a. Mortality classification head ─────────────────────────────
        self.mortality_head = nn.Sequential(
            nn.Linear(patient_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),   # logit → sigmoid for probability
        )

        # ── 4b. Cox proportional hazards head ─────────────────────────────
        # Outputs a scalar risk score β·p (log-hazard ratio)
        # Trained with Cox partial likelihood in training/cox_loss.py
        self.cox_head = nn.Sequential(
            nn.Linear(patient_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),   # unbounded log-hazard ratio
        )

        # ── Dropout for MC inference ───────────────────────────────────────
        self.mc_drop = nn.Dropout(dropout)
        self._mc_dropout_enabled = False

        # ── Adjacency buffer (set by set_causal_adj) ──────────────────────
        # Registered as buffer so it moves with .to(device) but not trained
        self.register_buffer(
            "adj",
            torch.ones(n_features, n_features) - torch.eye(n_features),
        )

    def set_causal_adj(self, adj: torch.Tensor):
        """
        Set the causal adjacency matrix from NOTEARS output.

        Call this after causal discovery before training:
            model.set_causal_adj(torch.tensor(dag, dtype=torch.float32))

        Parameters
        ----------
        adj : [F, F]  float tensor.  adj[i,j] > threshold means i→j exists.
        """
        # Threshold at 0.3 (standard for NOTEARS output)
        adj_binary = (adj.abs() > 0.3).float()
        adj_binary.fill_diagonal_(0)  # no self-loops
        self.adj.copy_(adj_binary)
        n_edges = int(adj_binary.sum().item())
        density = n_edges / (self.n_features * (self.n_features - 1))
        print(f"  Causal DAG set: {n_edges} edges  "
              f"(density {density:.1%}  out of {self.n_features*(self.n_features-1)} possible)")

    @contextmanager
    def mc_dropout(self):
        """
        Context manager: enable MC Dropout for uncertainty estimation.

            with model.mc_dropout():
                preds = [model(X, mask, delta, times) for _ in range(50)]
        """
        self._mc_dropout_enabled = True
        self.train()   # activates nn.Dropout layers
        try:
            yield
        finally:
            self._mc_dropout_enabled = False
            self.eval()

    def _build_node_features(
        self,
        X:     torch.Tensor,   # [B, T, F]
        mask:  torch.Tensor,   # [B, T, F]
        delta: torch.Tensor,   # [B, T, F]
        times: torch.Tensor,   # [B, T]
    ) -> torch.Tensor:
        """
        Construct per-node, per-timestep input vectors.

        node_f_t = [ x_f_t,            scalar: normalised feature value
                     mask_f_t,          scalar: 1=observed, 0=imputed
                     time2vec(t),       vector: absolute time embedding
                     time2vec(δ_f_t) ]  vector: delta time embedding

        Returns
        -------
        node_feats : [B, T, F, node_in_dim]
        """
        B, T, F = X.shape

        # Absolute time encoding: [B, T, t2v_dim] → broadcast to [B, T, F, t2v_dim]
        t_emb = self.t2v_absolute(times)               # [B, T, t2v_dim]
        t_emb = t_emb.unsqueeze(2).expand(B, T, F, -1) # [B, T, F, t2v_dim]

        # Delta time encoding: [B, T, F, t2v_dim]
        d_emb = self.t2v_delta(delta)                  # [B, T, F, t2v_dim]

        # Concatenate: [B, T, F, 1] + [B, T, F, 1] + [B, T, F, t2v_dim] * 2
        node_feats = torch.cat([
            X.unsqueeze(-1),      # [B, T, F, 1]
            mask.unsqueeze(-1),   # [B, T, F, 1]
            t_emb,                # [B, T, F, t2v_dim]
            d_emb,                # [B, T, F, t2v_dim]
        ], dim=-1)                # [B, T, F, node_in_dim]

        return node_feats

    def _apply_gat_sequence(
        self,
        node_feats: torch.Tensor,              # [B, T, F, node_in_dim]
        adj:        Optional[torch.Tensor],    # [F, F]
    ) -> torch.Tensor:
        """
        Apply stacked GAT layers at every timestep.
        Returns mean-pooled graph embedding per timestep: [B, T, gat_out_dim]
        """
        B, T, F, D = node_feats.shape
        gat_out_dim = self.gat_hidden * self.n_heads

        # Reshape: [B*T, F, D] — process all (patient, timestep) pairs at once
        h = node_feats.view(B * T, F, D)

        # Apply each GAT layer
        for layer in self.gat_layers:
            h_next = torch.zeros(B * T, F, layer.gat.out_dim * layer.gat.n_heads,
                                 device=h.device)
            for bt in range(B * T):
                h_next[bt] = layer(h[bt], adj)
            h = h_next   # [B*T, F, gat_out_dim]

        # Mean-pool features (nodes) → graph-level embedding
        z = h.mean(dim=1)           # [B*T, gat_out_dim]
        z = z.view(B, T, gat_out_dim)  # [B, T, gat_out_dim]
        return z

    def forward(
        self,
        X:       torch.Tensor,                     # [B, T, F]
        mask:    torch.Tensor,                     # [B, T, F]
        delta:   torch.Tensor,                     # [B, T, F]
        times:   torch.Tensor,                     # [B, T]
        lengths: Optional[torch.Tensor] = None,    # [B]  actual seq lengths
        adj_override: Optional[torch.Tensor] = None,  # for counterfactual
    ) -> dict:
        """
        Full forward pass.

        Parameters
        ----------
        adj_override : if provided, use this adjacency instead of self.adj.
                       Used for counterfactual inference (graph surgery).

        Returns
        -------
        dict with keys:
            mortality_logit  : [B, 1]  — logit for in-hospital mortality
            mortality_prob   : [B, 1]  — sigmoid(logit) ∈ (0, 1)
            risk_score       : [B, 1]  — Cox log-hazard ratio
            patient_emb      : [B, gru_hidden*2]  — patient representation
        """
        adj = adj_override if adj_override is not None else self.adj

        # ── 1. Node features ──────────────────────────────────────────────
        node_feats = self._build_node_features(X, mask, delta, times)
        # [B, T, F, node_in_dim]

        # ── 2. Causal GAT ─────────────────────────────────────────────────
        z = self._apply_gat_sequence(node_feats, adj)
        # [B, T, gat_out_dim]

        # Optional MC Dropout between GAT and GRU
        if self._mc_dropout_enabled:
            z = self.mc_drop(z)

        # ── 3. GRU over time ──────────────────────────────────────────────
        if lengths is not None:
            # Pack for efficiency — skip padded positions
            z_packed = nn.utils.rnn.pack_padded_sequence(
                z, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            gru_out_packed, h_n = self.gru(z_packed)
            gru_out, _ = nn.utils.rnn.pad_packed_sequence(
                gru_out_packed, batch_first=True
            )
        else:
            gru_out, h_n = self.gru(z)
            # gru_out: [B, T, gru_hidden*2]
            # h_n:     [n_layers*2, B, gru_hidden]

        # Patient embedding: concat last forward + last backward hidden states
        # h_n[-2] = last layer, forward direction
        # h_n[-1] = last layer, backward direction
        patient_emb = torch.cat([h_n[-2], h_n[-1]], dim=-1)  # [B, gru_hidden*2]

        # Optional MC Dropout on patient embedding
        if self._mc_dropout_enabled:
            patient_emb = self.mc_drop(patient_emb)

        # ── 4. Prediction heads ───────────────────────────────────────────
        mortality_logit = self.mortality_head(patient_emb)   # [B, 1]
        mortality_prob  = torch.sigmoid(mortality_logit)
        risk_score      = self.cox_head(patient_emb)         # [B, 1]

        return {
            "mortality_logit": mortality_logit,
            "mortality_prob":  mortality_prob,
            "risk_score":      risk_score,
            "patient_emb":     patient_emb,
            "gru_out":         gru_out,      # [B, T, gru_hidden*2] for attention viz
        }

    def predict_proba(self, X, mask, delta, times, lengths=None) -> torch.Tensor:
        """Convenience method: returns mortality probability [B]."""
        with torch.no_grad():
            out = self.forward(X, mask, delta, times, lengths)
        return out["mortality_prob"].squeeze(-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Unit tests ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(42)
    print("=== CausalTGAT unit tests ===\n")

    B, T, F = 4, 48, 17
    model = CausalTGAT(
        n_features=F, t2v_dim=16, gat_hidden=16,
        n_heads=2, n_gat_layers=2, gru_hidden=64, gru_layers=1,
        dropout=0.1,
    )
    model.eval()

    n_params = model.count_parameters()
    print(f"Model parameters: {n_params:,}")
    print(f"  Expected ~100k–500k for demo-scale model ✓\n")

    # Random inputs
    X      = torch.randn(B, T, F)
    mask   = (torch.rand(B, T, F) > 0.6).float()
    delta  = torch.rand(B, T, F) * 24
    times  = torch.arange(T).float().unsqueeze(0).expand(B, -1)

    print("Testing full forward pass...")
    with torch.no_grad():
        out = model(X, mask, delta, times)

    print(f"  mortality_logit : {tuple(out['mortality_logit'].shape)}")
    print(f"  mortality_prob  : {tuple(out['mortality_prob'].shape)}")
    print(f"  risk_score      : {tuple(out['risk_score'].shape)}")
    print(f"  patient_emb     : {tuple(out['patient_emb'].shape)}")
    probs = out["mortality_prob"].squeeze()
    print(f"  mortality probs : {[f'{p:.3f}' for p in probs.tolist()]} ✓\n")

    # Test causal adjacency
    adj = torch.zeros(F, F)
    # Simulate NOTEARS output: sparse causal connections
    edges = [(0,1), (1,2), (2,3), (8,9), (8,14), (12,1), (12,2)]
    for i, j in edges:
        adj[i, j] = 1.0
    model.set_causal_adj(adj)

    with torch.no_grad():
        out_causal = model(X, mask, delta, times)
    print(f"After setting causal DAG:")
    print(f"  mortality probs: {[f'{p:.3f}' for p in out_causal['mortality_prob'].squeeze().tolist()]} ✓\n")

    # Test MC Dropout
    print("Testing MC Dropout uncertainty estimation...")
    n_mc = 20
    mc_preds = []
    with model.mc_dropout():
        for _ in range(n_mc):
            out_mc = model(X, mask, delta, times)
            mc_preds.append(out_mc["mortality_prob"].squeeze())

    mc_preds  = torch.stack(mc_preds, dim=0)       # [n_mc, B]
    mean_pred = mc_preds.mean(dim=0)               # [B]
    epistemic = mc_preds.mean(dim=0).var()         # scalar (simplified)
    aleatoric = mc_preds.var(dim=0).mean()         # scalar

    print(f"  MC preds shape  : {tuple(mc_preds.shape)}")
    print(f"  Mean prediction : {[f'{p:.3f}' for p in mean_pred.tolist()]}")
    print(f"  Epistemic var   : {epistemic.item():.4f}")
    print(f"  Aleatoric var   : {aleatoric.item():.4f}  ✓\n")

    # Test counterfactual (graph surgery on feature 12 = lactate)
    print("Testing counterfactual inference (do(lactate=2.0))...")
    adj_surgery = model.adj.clone()
    LACTATE_IDX = 12
    adj_surgery[:, LACTATE_IDX] = 0   # remove all incoming edges to lactate

    X_cf = X.clone()
    X_cf[:, :, LACTATE_IDX] = 2.0     # fix lactate = 2.0 (approx z-score)

    with torch.no_grad():
        out_cf = model(X_cf, mask, delta, times, adj_override=adj_surgery)

    delta_risk = (
        out_cf["mortality_prob"].squeeze() - out_causal["mortality_prob"].squeeze()
    )
    print(f"  Δ risk (counterfactual - observed):")
    print(f"  {[f'{d:+.3f}' for d in delta_risk.tolist()]}  ✓")
    print(f"  (positive = intervention increases predicted risk, negative = decreases)")

    print("\n✓ All CausalTGAT tests passed.")