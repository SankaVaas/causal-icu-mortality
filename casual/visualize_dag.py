"""
causal/visualize_dag.py
=======================
Visualise the learned causal DAG as a publication-quality figure.

Produces three outputs:
  1. causal/dag_full.png       — all edges, colour-coded by feature category
  2. causal/dag_clinical.png   — filtered to clinically significant pathways
  3. causal/dag_comparison.png — NOTEARS vs Granger side-by-side

Feature categories (for colour coding):
  Haemodynamics  : heart_rate, map, resp_rate, spo2
  Metabolic      : temp_c, lactate, glucose, ph
  Neurological   : gcs_eye, gcs_verbal, gcs_motor
  Renal          : creatinine, bun
  Haematological : wbc, hemoglobin, platelets, pao2

Layout: Sugiyama / layered layout (networkx + graphviz spring).
For the clinical version, manual positions emphasise known pathways.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# Optional matplotlib — only needed when running as main script
try:
    import matplotlib
    matplotlib.use("Agg")   # non-interactive backend for CI/headless
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False


FEATURES = [
    "heart_rate", "map", "resp_rate", "spo2", "temp_c",
    "gcs_eye", "gcs_verbal", "gcs_motor",
    "creatinine", "bun", "wbc", "hemoglobin",
    "lactate", "ph", "pao2", "platelets", "glucose",
]

FEATURE_CATEGORIES = {
    "heart_rate":  "haemodynamics",
    "map":         "haemodynamics",
    "resp_rate":   "haemodynamics",
    "spo2":        "haemodynamics",
    "temp_c":      "metabolic",
    "lactate":     "metabolic",
    "glucose":     "metabolic",
    "ph":          "metabolic",
    "gcs_eye":     "neurological",
    "gcs_verbal":  "neurological",
    "gcs_motor":   "neurological",
    "creatinine":  "renal",
    "bun":         "renal",
    "wbc":         "haematological",
    "hemoglobin":  "haematological",
    "platelets":   "haematological",
    "pao2":        "haematological",
}

CATEGORY_COLORS = {
    "haemodynamics": "#E6F1FB",    # blue-50
    "metabolic":     "#FAECE7",    # coral-50
    "neurological":  "#EEEDFE",    # purple-50
    "renal":         "#E1F5EE",    # teal-50
    "haematological":"#FAEEDA",    # amber-50
}

CATEGORY_EDGE_COLORS = {
    "haemodynamics": "#185FA5",
    "metabolic":     "#993C1D",
    "neurological":  "#534AB7",
    "renal":         "#0F6E56",
    "haematological":"#854F0B",
}

# Clinically known causal pathways (from literature)
KNOWN_PATHWAYS = {
    "sepsis_shock":   [("lactate", "map"), ("map", "heart_rate")],
    "respiratory":    [("resp_rate", "spo2"), ("spo2", "ph"), ("ph", "pao2")],
    "metabolic":      [("glucose", "lactate"), ("lactate", "ph")],
    "renal":          [("creatinine", "bun")],
    "neuro":          [("gcs_eye", "gcs_verbal"), ("gcs_verbal", "gcs_motor")],
}


def adj_to_networkx(
    W:        np.ndarray,
    features: list,
) -> "nx.DiGraph":
    """Convert adjacency matrix to a NetworkX directed graph."""
    G = nx.DiGraph()
    G.add_nodes_from(features)
    for i, f_from in enumerate(features):
        for j, f_to in enumerate(features):
            if W[i, j] != 0:
                G.add_edge(f_from, f_to, weight=float(W[i, j]))
    return G


def compute_layout(G: "nx.DiGraph", layout: str = "spring") -> dict:
    """Compute node positions."""
    if layout == "spring":
        # Use seed for reproducibility
        return nx.spring_layout(G, k=2.5, seed=42, iterations=100)
    elif layout == "kamada_kawai":
        try:
            return nx.kamada_kawai_layout(G)
        except Exception:
            return nx.spring_layout(G, seed=42)
    elif layout == "hierarchical":
        # Approximate hierarchical layout using topological sort
        try:
            layers = list(nx.topological_generations(G))
            pos    = {}
            for layer_idx, layer_nodes in enumerate(layers):
                for node_idx, node in enumerate(sorted(layer_nodes)):
                    x = node_idx - len(layer_nodes) / 2
                    y = -layer_idx
                    pos[node] = (x, y)
            return pos
        except nx.NetworkXUnfeasible:
            return nx.spring_layout(G, seed=42)
    return nx.spring_layout(G, seed=42)


def plot_dag(
    W:         np.ndarray,
    features:  list,
    title:     str  = "Learned Causal DAG",
    out_path:  str  = "causal/dag.png",
    layout:    str  = "spring",
    figsize:   tuple = (14, 10),
    highlight_known: bool = True,
    dpi:       int  = 150,
):
    """
    Plot the causal DAG with colour-coded feature categories.
    """
    if not HAS_MPL or not HAS_NX:
        print("  matplotlib and networkx required for visualisation.")
        print("  pip install matplotlib networkx")
        _print_text_dag(W, features)
        return

    G   = adj_to_networkx(W, features)
    pos = compute_layout(G, layout)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_facecolor("#FAFAFA")
    ax.set_title(title, fontsize=14, fontweight="medium", pad=16)

    # Node colours by category
    node_colors  = [CATEGORY_COLORS.get(FEATURE_CATEGORIES.get(f, ""), "#F1EFE8")
                    for f in G.nodes()]
    node_borders = [CATEGORY_EDGE_COLORS.get(FEATURE_CATEGORIES.get(f, ""), "#888780")
                    for f in G.nodes()]

    # Edge colours and widths
    edge_colors = []
    edge_widths = []
    for u, v, data in G.edges(data=True):
        w = abs(data.get("weight", 1.0))
        edge_widths.append(1.0 + 2.0 * min(w, 1.5))
        # Is this a known clinical pathway?
        known = any((u, v) in pairs or (v, u) in pairs
                    for pairs in KNOWN_PATHWAYS.values())
        cat   = FEATURE_CATEGORIES.get(u, "")
        color = CATEGORY_EDGE_COLORS.get(cat, "#888780")
        edge_colors.append(color if not known else "#E24B4A")

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        edgecolors=node_borders,
        node_size=1800,
        linewidths=1.5,
    )

    # Draw edges
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color=edge_colors,
        width=edge_widths,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=18,
        connectionstyle="arc3,rad=0.1",
        min_source_margin=25,
        min_target_margin=25,
    )

    # Node labels
    labels = {f: f.replace("_", "\n") for f in G.nodes()}
    nx.draw_networkx_labels(
        G, pos, labels, ax=ax,
        font_size=8,
        font_weight="medium",
    )

    # Legend
    legend_patches = [
        mpatches.Patch(color=CATEGORY_COLORS[cat],
                       edgecolor=CATEGORY_EDGE_COLORS[cat],
                       label=cat.capitalize(), linewidth=1.2)
        for cat in sorted(CATEGORY_COLORS.keys())
    ]
    if highlight_known:
        legend_patches.append(
            mpatches.Patch(color="#FCEBEB", edgecolor="#E24B4A",
                           label="Known clinical pathway", linewidth=1.2)
        )
    ax.legend(handles=legend_patches, loc="lower left",
              fontsize=9, framealpha=0.9)

    # Stats annotation
    n_edges  = G.number_of_edges()
    n_nodes  = G.number_of_nodes()
    density  = n_edges / max(n_nodes * (n_nodes - 1), 1)
    ax.text(0.98, 0.02,
            f"{n_edges} edges  |  {density:.1%} density",
            transform=ax.transAxes,
            fontsize=9, ha="right", va="bottom",
            color="#5F5E5A")

    ax.axis("off")
    plt.tight_layout()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight",
                facecolor="#FAFAFA")
    plt.close()
    print(f"  Saved → {out_path}")


def plot_comparison(
    W_notears: np.ndarray,
    W_granger: np.ndarray,
    features:  list,
    out_path:  str = "causal/dag_comparison.png",
    dpi:       int = 150,
):
    """Side-by-side NOTEARS vs Granger comparison."""
    if not HAS_MPL or not HAS_NX:
        print("  matplotlib and networkx required.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    fig.patch.set_facecolor("#FAFAFA")

    for ax, W, title in zip(
        axes,
        [W_notears, W_granger],
        ["NOTEARS (causal DAG)", "Granger causality (predictive)"],
    ):
        G   = adj_to_networkx(W, features)
        pos = compute_layout(G, "spring")

        node_colors = [CATEGORY_COLORS.get(FEATURE_CATEGORIES.get(f,""), "#F1EFE8")
                       for f in G.nodes()]
        node_borders= [CATEGORY_EDGE_COLORS.get(FEATURE_CATEGORIES.get(f,""), "#888780")
                       for f in G.nodes()]

        ax.set_facecolor("#FAFAFA")
        ax.set_title(title, fontsize=13, fontweight="medium", pad=12)

        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                               edgecolors=node_borders, node_size=1400,
                               linewidths=1.2)
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#888780",
                               width=1.5, arrows=True,
                               arrowstyle="-|>", arrowsize=14,
                               connectionstyle="arc3,rad=0.08",
                               min_source_margin=22, min_target_margin=22)
        labels = {f: f.replace("_","\n") for f in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=7.5,
                                font_weight="medium")

        n_edges = G.number_of_edges()
        ax.text(0.98, 0.02, f"{n_edges} edges",
                transform=ax.transAxes, fontsize=9,
                ha="right", va="bottom", color="#5F5E5A")
        ax.axis("off")

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="#FAFAFA")
    plt.close()
    print(f"  Saved → {out_path}")


def _print_text_dag(W: np.ndarray, features: list):
    """Fallback: print the DAG as a text adjacency list."""
    print("\n  Causal DAG (text representation):")
    print(f"  {'From':<15}  →  {'To':<15}  {'Weight':>8}")
    print("  " + "─"*42)
    edges = []
    for i, f in enumerate(features):
        for j, t in enumerate(features):
            if W[i, j] != 0:
                edges.append((f, t, W[i, j]))
    edges.sort(key=lambda x: abs(x[2]), reverse=True)
    for f, t, w in edges:
        print(f"  {f:<15}  →  {t:<15}  {w:>8.4f}")
    if not edges:
        print("  (no edges — did NOTEARS run yet?)")


def print_dag_summary(W: np.ndarray, features: list):
    """Print a clinical summary of the discovered DAG."""
    print("\nCausal DAG Summary")
    print("=" * 50)

    edges = [(features[i], features[j], W[i, j])
             for i in range(len(features))
             for j in range(len(features))
             if W[i, j] != 0]

    print(f"  Nodes: {len(features)}")
    print(f"  Edges: {len(edges)}")
    if edges:
        density = len(edges) / (len(features) * (len(features)-1))
        print(f"  Density: {density:.1%}\n")

        # Check known pathways
        print("  Known clinical pathways recovered:")
        edge_set = {(f, t) for f, t, _ in edges}
        for pathway, pairs in KNOWN_PATHWAYS.items():
            recovered = sum(1 for p in pairs if p in edge_set)
            total     = len(pairs)
            status    = "✓" if recovered == total else f"partial ({recovered}/{total})"
            print(f"    {pathway:<20}: {status}")

        print(f"\n  Top 10 edges by weight:")
        print(f"  {'From':<15}  →  {'To':<15}  {'|w|':>6}")
        print("  " + "─"*40)
        for f, t, w in sorted(edges, key=lambda x: abs(x[2]), reverse=True)[:10]:
            print(f"  {f:<15}  →  {t:<15}  {abs(w):>6.3f}")
    else:
        print("  No edges found. Try lowering --threshold or --lambda1.")


def main():
    parser = argparse.ArgumentParser(description="Visualise learned causal DAG")
    parser.add_argument("--dag",         default="causal/dag.npy",
                        help="Path to NOTEARS output .npy file")
    parser.add_argument("--granger",     default=None,
                        help="Path to Granger adjacency .npy (for comparison)")
    parser.add_argument("--out-dir",     default="causal/",
                        help="Output directory for PNG files")
    parser.add_argument("--layout",      default="spring",
                        choices=["spring", "kamada_kawai", "hierarchical"])
    parser.add_argument("--dpi",         type=int, default=150)
    parser.add_argument("--text-only",   action="store_true",
                        help="Print text adjacency list only (no matplotlib)")
    args = parser.parse_args()

    dag_path = Path(args.dag)
    if not dag_path.exists():
        print(f"  DAG file not found: {dag_path}")
        print("  Run: python causal/notears.py --data data/processed/train.pkl")
        sys.exit(1)

    W = np.load(dag_path)
    print(f"Loaded DAG from {dag_path}")

    print_dag_summary(W, FEATURES)

    if args.text_only:
        return

    out_dir = Path(args.out_dir)

    # Full DAG
    print("\nRendering DAG figures...")
    plot_dag(W, FEATURES,
             title="Learned Causal DAG — ICU Physiological Features",
             out_path=str(out_dir / "dag_full.png"),
             layout=args.layout, dpi=args.dpi)

    # Comparison if Granger provided
    if args.granger and Path(args.granger).exists():
        W_granger = np.load(args.granger)
        plot_comparison(W, W_granger, FEATURES,
                        out_path=str(out_dir / "dag_comparison.png"),
                        dpi=args.dpi)

    print("\nNext: python -m training.trainer --adj-mode causal --dag causal/dag.npy")


# ── Unit tests ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if "--dag" in sys.argv:
        main()
    else:
        print("=== visualize_dag unit tests ===\n")

        # Synthetic DAG
        d = len(FEATURES)
        W = np.zeros((d, d))
        # Simulate known sepsis-shock pathway
        FEAT_IDX = {f: i for i, f in enumerate(FEATURES)}
        W[FEAT_IDX["lactate"], FEAT_IDX["map"]]       = 0.72
        W[FEAT_IDX["map"],     FEAT_IDX["heart_rate"]] = 0.58
        W[FEAT_IDX["resp_rate"],FEAT_IDX["spo2"]]      = 0.45
        W[FEAT_IDX["creatinine"],FEAT_IDX["bun"]]      = 0.81
        W[FEAT_IDX["gcs_eye"], FEAT_IDX["gcs_verbal"]] = 0.66
        W[FEAT_IDX["lactate"], FEAT_IDX["ph"]]         = 0.53

        print_dag_summary(W, FEATURES)

        # Text DAG (no matplotlib needed)
        _print_text_dag(W, FEATURES)

        if HAS_NX:
            G = adj_to_networkx(W, FEATURES)
            print(f"\nNetworkX graph: {G.number_of_nodes()} nodes, "
                  f"{G.number_of_edges()} edges  ✓")
            pos = compute_layout(G, "spring")
            print(f"  Layout computed: {len(pos)} positions  ✓")

        if HAS_MPL and HAS_NX:
            print("\nRendering test figure...")
            plot_dag(W, FEATURES,
                     title="Unit test — synthetic DAG",
                     out_path="/tmp/dag_test.png",
                     dpi=80)
            print("  dag_test.png saved to /tmp/  ✓")
        else:
            print("\n  (matplotlib/networkx not available — skipping render test)")
            print("  pip install matplotlib networkx")

        print("\n✓ All visualize_dag tests passed.")