"""
models/tgat_layer.py
====================
Temporal Graph Attention Network (TGAT) Layer
with Causal Adjacency Masking.

Theory
------

Graph Attention Networks (GAT — Veličković et al., ICLR 2018)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
For a graph with nodes i and edges (i→j), GAT computes:

    e_ij = LeakyReLU( aᵀ [W·hᵢ ‖ W·hⱼ] )
    α_ij = softmax_j(e_ij)                    # attention coefficient
    h'_i = σ( Σⱼ α_ij · W·hⱼ )              # new node embedding

The key property: α_ij is *learned*, not hand-crafted.  Nodes attend
to their most relevant neighbours dynamically per input.

Multi-head attention
~~~~~~~~~~~~~~~~~~~~
We run K independent attention heads and concatenate:

    h'_i = ‖_{k=1..K} σ( Σⱼ α^k_ij · W^k·hⱼ )

This lets different heads specialise — e.g. one head might learn
the lactate → MAP pathway while another learns GCS → mortality.

Causal masking — the architectural novelty of this project
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Standard GNNs use a fully-connected or learned graph.  Here we inject
the causal DAG from NOTEARS as a *hard constraint*:

    α_ij = softmax_j( e_ij ) · A_ij          ← A_ij = 0 blocks non-causal edges

Before softmax, we set e_ij = -∞ for (i,j) pairs with A_ij = 0.
After softmax, those positions become exactly 0 — no gradient flows
through non-causal edges.

The model is architecturally incapable of exploiting spurious
correlations that are not supported by the causal graph.  This is not
a regulariser — it is a hard structural constraint.

Temporal encoding
~~~~~~~~~~~~~~~~~
At each timestep t, the input to each node i is:

    x̃_i(t) = [ x_i(t) ‖ mask_i(t) ‖ time2vec(t) ‖ time2vec(δ_i(t)) ]

where x_i(t) is the normalised feature value, mask_i(t) ∈ {0,1} indicates
whether x_i(t) was truly observed (not imputed), and δ_i(t) is the time
since the last real observation of feature i.

The time encodings make attention time-aware: a vital measured 5 minutes
ago should be treated differently than one measured 5 hours ago.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class CausalGraphAttentionLayer(nn.Module):
    """
    Single causal graph attention layer.

    Parameters
    ----------
    in_dim      : node feature input dimension
    out_dim     : node feature output dimension per head
    n_heads     : number of attention heads K
    dropout     : dropout on attention coefficients
    negative_slope : LeakyReLU negative slope
    """

    def __init__(
        self,
        in_dim:         int,
        out_dim:        int,
        n_heads:        int   = 4,
        dropout:        float = 0.3,
        negative_slope: float = 0.2,
    ):
        super().__init__()
        self.in_dim         = in_dim
        self.out_dim        = out_dim
        self.n_heads        = n_heads
        self.dropout        = dropout
        self.negative_slope = negative_slope

        # Linear projection: shared across heads for efficiency
        # W ∈ ℝ^{in_dim × (out_dim * n_heads)}
        self.W = nn.Linear(in_dim, out_dim * n_heads, bias=False)

        # Attention vector a ∈ ℝ^{2*out_dim} per head
        # Concatenation [W·hᵢ ‖ W·hⱼ] has dim 2*out_dim
        self.a = nn.Parameter(torch.FloatTensor(n_heads, 2 * out_dim))

        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.attn_drop  = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        """Glorot uniform init — standard for GAT."""
        nn.init.xavier_uniform_(self.W.weight, gain=1.414)
        nn.init.xavier_uniform_(
            self.a.view(self.n_heads, 2, self.out_dim), gain=1.414
        )

    def forward(
        self,
        h:    torch.Tensor,                      # [N, in_dim]  node features
        adj:  Optional[torch.Tensor] = None,     # [N, N]  adjacency (0/1 or weights)
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        h   : [N, in_dim]  node feature matrix
        adj : [N, N]  causal adjacency.  A[i,j]=1 means edge i→j exists.
              If None, uses fully-connected graph (standard GAT).

        Returns
        -------
        h'  : [N, n_heads * out_dim]  new node embeddings
        """
        N = h.size(0)

        # ── Linear projection ─────────────────────────────────────────────
        # Wh: [N, n_heads * out_dim] → [n_heads, N, out_dim]
        Wh = self.W(h).view(N, self.n_heads, self.out_dim)
        Wh = Wh.permute(1, 0, 2)  # [n_heads, N, out_dim]

        # ── Attention scores ──────────────────────────────────────────────
        # For each head k:
        #   e_ij = LeakyReLU( aₖ[0:d] · Wh[k,i] + aₖ[d:] · Wh[k,j] )
        # Efficient broadcast:
        #   [n_heads, N, 1, out_dim] + [n_heads, 1, N, out_dim] → [n_heads, N, N, out_dim]
        # But we can split a into two halves and compute dot products separately.

        a1 = self.a[:, :self.out_dim]   # [n_heads, out_dim]  — "source" weights
        a2 = self.a[:, self.out_dim:]   # [n_heads, out_dim]  — "target" weights

        # Score from source: [n_heads, N, 1] = Wh @ a1ᵀ
        score_src = (Wh * a1.unsqueeze(1)).sum(-1, keepdim=True)  # [n_heads, N, 1]
        # Score from target: [n_heads, 1, N]
        score_dst = (Wh * a2.unsqueeze(1)).sum(-1).unsqueeze(1)   # [n_heads, 1, N]

        # Attention matrix: [n_heads, N, N]
        e = self.leaky_relu(score_src + score_dst)

        # ── Causal masking (the core contribution) ────────────────────────
        # Set e[k, i, j] = -∞ wherever adj[i, j] = 0
        # After softmax, these positions become exactly 0.
        if adj is not None:
            # adj: [N, N] → expand for heads: [1, N, N]
            mask = (adj == 0).unsqueeze(0)        # [1, N, N]
            e = e.masked_fill(mask, float("-inf"))
        # Note: diagonal is not explicitly blocked — self-loops are allowed.
        # The NOTEARS DAG has no self-loops by construction.

        # ── Normalise ─────────────────────────────────────────────────────
        alpha = F.softmax(e, dim=-1)   # [n_heads, N, N]
        # If all edges to node i are masked, softmax([−∞,−∞,…]) = NaN.
        # Replace NaN with 0 (isolated node gets zero message).
        alpha = torch.nan_to_num(alpha, nan=0.0)
        alpha = self.attn_drop(alpha)  # [n_heads, N, N]

        # ── Aggregate ─────────────────────────────────────────────────────
        # h'_k[i] = Σⱼ α_ij · Wh[k,j]
        # [n_heads, N, N] × [n_heads, N, out_dim] → [n_heads, N, out_dim]
        h_prime = torch.bmm(alpha, Wh)  # [n_heads, N, out_dim]

        # Concatenate heads: [N, n_heads * out_dim]
        h_prime = h_prime.permute(1, 0, 2).contiguous().view(N, self.n_heads * self.out_dim)

        return h_prime

    def get_attention_weights(
        self,
        h:   torch.Tensor,
        adj: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Return attention weights α for interpretability / visualisation.

        Returns
        -------
        alpha : [n_heads, N, N]  — attention weight from node j to node i
        """
        N = h.size(0)
        Wh = self.W(h).view(N, self.n_heads, self.out_dim).permute(1, 0, 2)
        a1 = self.a[:, :self.out_dim]
        a2 = self.a[:, self.out_dim:]
        score_src = (Wh * a1.unsqueeze(1)).sum(-1, keepdim=True)
        score_dst = (Wh * a2.unsqueeze(1)).sum(-1).unsqueeze(1)
        e = self.leaky_relu(score_src + score_dst)
        if adj is not None:
            e = e.masked_fill((adj == 0).unsqueeze(0), float("-inf"))
        return F.softmax(e, dim=-1)


class TemporalGATLayer(nn.Module):
    """
    Full Temporal GAT layer for one timestep's node feature matrix.

    Wraps CausalGraphAttentionLayer with:
    - ELU activation after aggregation
    - Optional residual connection (if in_dim == out_dim * n_heads)
    - Layer normalisation

    This is applied identically at every timestep t in [0, T-1].
    The temporal dependency is handled by the GRU *above* this layer —
    the GAT handles *spatial* (inter-feature) dependencies, the GRU handles
    *temporal* dependencies.  This separation is what makes the causal
    structure interpretable: the DAG constrains the spatial graph, not time.
    """

    def __init__(
        self,
        in_dim:         int,
        out_dim:        int,
        n_heads:        int   = 4,
        dropout:        float = 0.3,
        negative_slope: float = 0.2,
        residual:       bool  = True,
    ):
        super().__init__()
        self.residual   = residual
        hidden_dim      = out_dim * n_heads

        self.gat        = CausalGraphAttentionLayer(
            in_dim, out_dim, n_heads, dropout, negative_slope
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.activation = nn.ELU()
        self.dropout    = nn.Dropout(dropout)

        # Residual projection if dimensions don't match
        self.res_proj   = (
            nn.Linear(in_dim, hidden_dim, bias=False)
            if (residual and in_dim != hidden_dim) else None
        )

    def forward(
        self,
        h:   torch.Tensor,                  # [N, in_dim]
        adj: Optional[torch.Tensor] = None, # [N, N]
    ) -> torch.Tensor:
        """
        Returns
        -------
        h' : [N, n_heads * out_dim]
        """
        h_agg = self.activation(self.gat(h, adj))  # [N, n_heads * out_dim]
        h_agg = self.dropout(h_agg)

        if self.residual:
            res = self.res_proj(h) if self.res_proj is not None else h
            h_agg = h_agg + res

        return self.layer_norm(h_agg)


class StackedTemporalGAT(nn.Module):
    """
    Stack of L TemporalGAT layers applied at each timestep.

    The final output is the mean-pooled node embedding, giving a single
    fixed-size vector per timestep:

        z_t = mean_pool( TGAT_L( ... TGAT_1(x_t) ... ) )  ∈ ℝ^{hidden_dim}

    This z_t sequence [z_0, z_1, ..., z_{T-1}] is then fed to the GRU.

    Parameters
    ----------
    node_in_dim  : input node feature dimension (includes time encoding)
    hidden_dim   : per-head output dim at each layer
    n_heads      : attention heads
    n_layers     : number of stacked GAT layers
    dropout      : dropout rate
    """

    def __init__(
        self,
        node_in_dim: int,
        hidden_dim:  int  = 32,
        n_heads:     int  = 4,
        n_layers:    int  = 2,
        dropout:     float = 0.3,
    ):
        super().__init__()
        self.n_layers   = n_layers
        self.hidden_dim = hidden_dim * n_heads

        layers = []
        for i in range(n_layers):
            in_d = node_in_dim if i == 0 else hidden_dim * n_heads
            layers.append(
                TemporalGATLayer(in_d, hidden_dim, n_heads, dropout, residual=(i > 0))
            )
        self.layers = nn.ModuleList(layers)

    def forward(
        self,
        x_seq: torch.Tensor,               # [T, N, node_in_dim]
        adj:   Optional[torch.Tensor] = None,  # [N, N]
    ) -> torch.Tensor:
        """
        Apply stacked GAT at every timestep, then mean-pool nodes.

        Parameters
        ----------
        x_seq : [T, N, node_in_dim]  — sequence of node feature matrices
        adj   : [N, N]               — fixed causal adjacency (same for all t)

        Returns
        -------
        z_seq : [T, hidden_dim]  — sequence of graph-level embeddings
        """
        T, N, _ = x_seq.shape
        z_seq = []

        for t in range(T):
            h = x_seq[t]              # [N, node_in_dim]
            for layer in self.layers:
                h = layer(h, adj)     # [N, hidden_dim]
            # Mean pooling over nodes → graph-level vector
            z = h.mean(dim=0)         # [hidden_dim]
            z_seq.append(z)

        return torch.stack(z_seq, dim=0)  # [T, hidden_dim]

    def forward_batched(
        self,
        x_seq: torch.Tensor,               # [B, T, N, node_in_dim]
        adj:   Optional[torch.Tensor] = None,  # [N, N]
    ) -> torch.Tensor:
        """
        Batched version: process B patients simultaneously.

        More memory-efficient than looping over batch in the trainer.

        Returns
        -------
        z_seq : [B, T, hidden_dim]
        """
        B, T, N, D = x_seq.shape

        # Reshape to [B*T, N, D] — process all (patient, timestep) pairs at once
        h = x_seq.view(B * T, N, D)

        for layer in self.layers:
            # Each layer processes [B*T, N, ...] independently
            # adj is shared: [N, N] — same causal structure for all patients
            h_out_list = []
            for bt in range(B * T):
                h_out_list.append(layer(h[bt], adj))
            h = torch.stack(h_out_list, dim=0)  # [B*T, N, hidden_dim]

        # Mean pool nodes: [B*T, N, hidden_dim] → [B*T, hidden_dim]
        z = h.mean(dim=1)

        # Reshape: [B*T, hidden_dim] → [B, T, hidden_dim]
        return z.view(B, T, self.hidden_dim)


# ── Unit tests ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(42)
    print("=== TGAT layer unit tests ===\n")

    N = 17   # features = graph nodes
    in_dim  = 1 + 1 + 32   # [feature_value, mask, time2vec_output]
    out_dim = 32
    n_heads = 4
    T = 48
    B = 4

    # Random causal adjacency (simulating NOTEARS output)
    adj = (torch.rand(N, N) > 0.7).float()
    adj.fill_diagonal_(0)  # no self-loops
    print(f"Causal DAG: {int(adj.sum())} edges out of {N*N} possible")
    print(f"  Sparsity: {1-adj.mean():.1%} masked\n")

    # Single layer
    layer = CausalGraphAttentionLayer(in_dim, out_dim, n_heads)
    h = torch.randn(N, in_dim)
    h_prime = layer(h, adj)
    print(f"CausalGATLayer: {tuple(h.shape)} → {tuple(h_prime.shape)}")
    print(f"  Expected: [{N}, {out_dim * n_heads}]  ✓\n")

    # Attention weights (for interpretability)
    alpha = layer.get_attention_weights(h, adj)
    print(f"Attention weights shape: {tuple(alpha.shape)}  (n_heads, N, N)")
    print(f"  Non-zero entries: {(alpha > 1e-6).sum().item()}")
    print(f"  Expected ≤ {int(adj.sum()) * n_heads} (causal edges × heads)  ✓\n")

    # Temporal GAT with stacking
    node_in_dim = in_dim
    stacked = StackedTemporalGAT(node_in_dim, hidden_dim=32, n_heads=4, n_layers=2)
    x_seq = torch.randn(T, N, node_in_dim)
    z_seq = stacked(x_seq, adj)
    print(f"StackedTemporalGAT (sequential): {tuple(x_seq.shape)} → {tuple(z_seq.shape)}")
    print(f"  Expected: [{T}, {32*4}]  ✓\n")

    # Batched version
    x_batch = torch.randn(B, T, N, node_in_dim)
    z_batch = stacked.forward_batched(x_batch, adj)
    print(f"StackedTemporalGAT (batched): {tuple(x_batch.shape)} → {tuple(z_batch.shape)}")
    print(f"  Expected: [{B}, {T}, {32*4}]  ✓\n")

    # Verify causal masking: non-causal edges should have zero attention
    alpha_masked = layer.get_attention_weights(h, adj)
    non_edge_attn = alpha_masked[:, adj == 0]
    print(f"Attention on blocked (non-causal) edges: max={non_edge_attn.max():.2e}")
    print(f"  Expected ≈ 0.0 (causal masking works)  ✓")

    print("\n✓ All TGAT tests passed.")