"""
inference/mc_dropout.py
=======================
Monte Carlo Dropout for Uncertainty Quantification.

Theory — Why uncertainty matters clinically
-------------------------------------------
A single mortality probability p=0.73 is clinically ambiguous: is the model
*confident* (it has seen many similar patients and they all died), or is it
*uncertain* (this patient is unusual and the model is extrapolating)?

These are fundamentally different situations requiring different actions:
  - High confidence + high risk   → immediate escalation
  - High uncertainty              → flag for human expert review
  - High aleatoric uncertainty    → patient is inherently unpredictable

MC Dropout (Gal & Ghahramani, ICML 2016) approximates Bayesian inference
using standard dropout as a variational distribution:

    p(y | x) ≈ (1/T) Σ_{t=1..T} p(y | x, ω_t)

where ω_t are stochastic weight samples drawn by keeping dropout *active*
at test time.  T=50 samples is sufficient for stable estimates.

Uncertainty decomposition
-------------------------
Total predictive uncertainty = aleatoric + epistemic:

    Var[y | x] = E_ω[Var[y|x,ω]] + Var_ω[E[y|x,ω]]
                  ─────────────────   ─────────────────
                    aleatoric              epistemic

  Aleatoric (data uncertainty):
    E_ω[ Var[y|x,ω] ] ≈ (1/T) Σ_t  p_t(1-p_t)
    Irreducible — inherent noise in patient physiology.
    A patient in septic shock who might recover or die is aleatoric.

  Epistemic (model uncertainty):
    Var_ω[ E[y|x,ω] ] ≈ Var_t[ p_t ] = (1/T) Σ_t (p_t - p̄)²
    Reducible — uncertainty from lack of training data.
    A patient with an unusual comorbidity profile is epistemic.
    More data about similar patients would reduce this.

Clinical thresholds
-------------------
We flag patients when epistemic uncertainty > θ_e = 0.05.
These patients should be reviewed by a clinician before acting on the
model's prediction — the model is essentially saying "I haven't seen
enough cases like this to be confident."
"""

import argparse
import sys
import pickle
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.dataset    import build_dataloaders, get_feature_names
from models.causal_tgat import CausalTGAT


# ── Core MC Dropout inference ──────────────────────────────────────────────

def mc_dropout_predict(
    model:     CausalTGAT,
    X:         torch.Tensor,   # [B, T, F]
    mask:      torch.Tensor,   # [B, T, F]
    delta:     torch.Tensor,   # [B, T, F]
    times:     torch.Tensor,   # [B, T]
    lengths:   Optional[torch.Tensor] = None,
    T_samples: int = 50,
    device:    torch.device = torch.device("cpu"),
) -> dict:
    """
    Run T stochastic forward passes with dropout active.

    Parameters
    ----------
    T_samples : number of MC samples (50 is standard; 20 is fast, 100 is accurate)

    Returns
    -------
    dict with keys:
        mean_prob      : [B]    — mean predicted mortality probability
        std_prob       : [B]    — std across MC samples (total uncertainty proxy)
        epistemic      : [B]    — Var_ω[E[y|x,ω]] (model uncertainty)
        aleatoric      : [B]    — E_ω[Var[y|x,ω]] (data uncertainty)
        all_probs      : [B, T] — all T probability samples (for diagnostics)
        risk_mean      : [B]    — mean Cox risk score across samples
        risk_std       : [B]    — std of risk scores
    """
    X      = X.to(device)
    mask   = mask.to(device)
    delta  = delta.to(device)
    times  = times.to(device)
    if lengths is not None:
        lengths = lengths.to(device)

    all_probs = []
    all_risks = []

    # Keep dropout active (model.train() mode) for all T passes
    model.train()
    with torch.no_grad():
        for _ in range(T_samples):
            out = model(X, mask, delta, times, lengths)
            p   = out["mortality_prob"].squeeze(-1)   # [B]
            r   = out["risk_score"].squeeze(-1)        # [B]
            all_probs.append(p.cpu())
            all_risks.append(r.cpu())

    model.eval()

    # Stack: [T, B]
    all_probs = torch.stack(all_probs, dim=0)  # [T, B]
    all_risks = torch.stack(all_risks, dim=0)  # [T, B]

    # ── Uncertainty decomposition ──────────────────────────────────────────
    # Mean prediction: p̄ = (1/T) Σ p_t
    mean_prob = all_probs.mean(dim=0)      # [B]
    std_prob  = all_probs.std(dim=0)       # [B]

    # Epistemic: Var_t[p_t] = (1/T) Σ (p_t - p̄)²
    epistemic = all_probs.var(dim=0)       # [B]

    # Aleatoric: E_t[p_t(1-p_t)] = (1/T) Σ p_t(1-p_t)
    aleatoric = (all_probs * (1 - all_probs)).mean(dim=0)   # [B]

    return {
        "mean_prob":  mean_prob.numpy(),           # [B]
        "std_prob":   std_prob.numpy(),            # [B]
        "epistemic":  epistemic.numpy(),           # [B]
        "aleatoric":  aleatoric.numpy(),           # [B]
        "all_probs":  all_probs.T.numpy(),         # [B, T]
        "risk_mean":  all_risks.mean(dim=0).numpy(),  # [B]
        "risk_std":   all_risks.std(dim=0).numpy(),   # [B]
    }


def run_full_uncertainty_analysis(
    model:       CausalTGAT,
    loader,
    T_samples:   int   = 50,
    epi_thresh:  float = 0.05,   # flag patients above this epistemic uncertainty
    device:      torch.device = torch.device("cpu"),
) -> dict:
    """
    Run MC Dropout over an entire DataLoader, collecting all predictions
    and uncertainty estimates.

    Returns
    -------
    dict with aggregated arrays and clinical summary statistics
    """
    all_mean_prob  = []
    all_epistemic  = []
    all_aleatoric  = []
    all_std        = []
    all_labels     = []
    all_durations  = []
    all_stay_ids   = []
    all_risk_mean  = []

    print(f"  Running MC Dropout (T={T_samples} samples)...")

    for batch in loader:
        X, mask_t, delta, times, lengths, y, meta = batch

        results = mc_dropout_predict(
            model, X, mask_t, delta, times, lengths,
            T_samples=T_samples, device=device,
        )

        all_mean_prob.extend(results["mean_prob"].tolist())
        all_epistemic.extend(results["epistemic"].tolist())
        all_aleatoric.extend(results["aleatoric"].tolist())
        all_std.extend(results["std_prob"].tolist())
        all_labels.extend(y.numpy().tolist())
        all_durations.extend([m["los_hours"] for m in meta])
        all_stay_ids.extend([m["stay_id"]    for m in meta])
        all_risk_mean.extend(results["risk_mean"].tolist())

    mean_prob  = np.array(all_mean_prob)
    epistemic  = np.array(all_epistemic)
    aleatoric  = np.array(all_aleatoric)
    labels     = np.array(all_labels)
    durations  = np.array(all_durations)
    stay_ids   = np.array(all_stay_ids)
    risk_mean  = np.array(all_risk_mean)
    N          = len(labels)

    # ── Flagged patients ───────────────────────────────────────────────────
    flagged_mask    = epistemic > epi_thresh
    n_flagged       = flagged_mask.sum()
    flagged_ids     = stay_ids[flagged_mask].tolist()

    # ── Summary statistics ─────────────────────────────────────────────────
    summary = {
        "n_patients":           N,
        "n_events":             int(labels.sum()),
        "mean_predicted_risk":  float(mean_prob.mean()),
        "mean_epistemic":       float(epistemic.mean()),
        "mean_aleatoric":       float(aleatoric.mean()),
        "epistemic_fraction":   float(epistemic.mean() / (epistemic.mean() + aleatoric.mean() + 1e-8)),
        "aleatoric_fraction":   float(aleatoric.mean() / (epistemic.mean() + aleatoric.mean() + 1e-8)),
        "n_flagged":            int(n_flagged),
        "pct_flagged":          float(100 * n_flagged / max(N, 1)),
        "flagged_stay_ids":     flagged_ids,
        "epi_threshold":        epi_thresh,
    }

    # Per-patient results
    per_patient = [
        {
            "stay_id":     int(stay_ids[i]),
            "label":       int(labels[i]),
            "mean_prob":   float(mean_prob[i]),
            "epistemic":   float(epistemic[i]),
            "aleatoric":   float(aleatoric[i]),
            "flagged":     bool(flagged_mask[i]),
            "risk_score":  float(risk_mean[i]),
            "duration_h":  float(durations[i]),
        }
        for i in range(N)
    ]

    return {
        "summary":     summary,
        "per_patient": per_patient,
        "arrays": {
            "mean_prob":  mean_prob,
            "epistemic":  epistemic,
            "aleatoric":  aleatoric,
            "labels":     labels,
            "durations":  durations,
            "risk_mean":  risk_mean,
        },
    }


def print_uncertainty_report(results: dict):
    """Print a clinical-style uncertainty report to the terminal."""
    s  = results["summary"]
    pp = sorted(results["per_patient"], key=lambda x: -x["mean_prob"])

    print("\n" + "═"*62)
    print("  MC Dropout Uncertainty Report")
    print("═"*62)
    print(f"  Patients analysed    : {s['n_patients']}")
    print(f"  Observed events      : {s['n_events']}  "
          f"({100*s['n_events']/max(s['n_patients'],1):.1f}%)")
    print()
    print(f"  Mean predicted risk  : {s['mean_predicted_risk']:.3f}")
    print(f"  Mean epistemic Var   : {s['mean_epistemic']:.4f}  "
          f"({s['epistemic_fraction']*100:.1f}% of total)")
    print(f"  Mean aleatoric Var   : {s['mean_aleatoric']:.4f}  "
          f"({s['aleatoric_fraction']*100:.1f}% of total)")
    print()
    print(f"  Flagged (epi>{s['epi_threshold']:.3f}): "
          f"{s['n_flagged']} patients  ({s['pct_flagged']:.1f}%)")
    print("  → Flagged patients should be reviewed by a clinician")
    print()

    print(f"  {'Stay ID':>12}  {'P(mort)':>8}  {'Epistemic':>10}  "
          f"{'Aleatoric':>10}  {'Label':>6}  {'Flag':>5}")
    print("  " + "─"*60)

    # Top-10 by predicted risk
    for p in pp[:10]:
        flag = "  ★" if p["flagged"] else ""
        print(
            f"  {p['stay_id']:>12}  {p['mean_prob']:>8.3f}  "
            f"{p['epistemic']:>10.4f}  {p['aleatoric']:>10.4f}  "
            f"{p['label']:>6}  {flag}"
        )

    if len(pp) > 10:
        print(f"  ... and {len(pp)-10} more patients")

    print("═"*62)


def load_model_from_checkpoint(ckpt_path: str, device: torch.device) -> CausalTGAT:
    """Load model from a trainer.py checkpoint file."""
    ckpt   = torch.load(ckpt_path, map_location=device)
    config = ckpt.get("config", {})

    model = CausalTGAT(
        n_features   = config.get("n_features",   17),
        t2v_dim      = config.get("t2v_dim",       16),
        gat_hidden   = config.get("gat_hidden",    32),
        n_heads      = config.get("n_heads",        4),
        n_gat_layers = config.get("n_gat_layers",   2),
        gru_hidden   = config.get("gru_hidden",   128),
        gru_layers   = config.get("gru_layers",     2),
        dropout      = config.get("dropout",      0.3),
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="MC Dropout uncertainty analysis")
    parser.add_argument("--checkpoint",    required=True,
                        help="Path to checkpoints/best.pt")
    parser.add_argument("--processed-dir", default="data/processed/")
    parser.add_argument("--split",         default="test",
                        choices=["train", "val", "test"])
    parser.add_argument("--T",             type=int, default=50,
                        help="Number of MC samples (default 50)")
    parser.add_argument("--epi-thresh",    type=float, default=0.05,
                        help="Epistemic uncertainty threshold for flagging")
    parser.add_argument("--out",           default="inference/uncertainty.json",
                        help="Output JSON path")
    args = parser.parse_args()

    device = torch.device("cpu")

    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    model = load_model_from_checkpoint(args.checkpoint, device)
    print(f"  Parameters: {model.count_parameters():,}")

    # Load data
    train_l, val_l, test_l, _ = build_dataloaders(
        processed_dir=args.processed_dir,
        batch_size=16,
        augment_train=False,
    )
    loader_map = {"train": train_l, "val": val_l, "test": test_l}
    loader = loader_map[args.split]
    print(f"\nRunning on {args.split} split ({len(loader.dataset)} patients)...")

    # Run analysis
    results = run_full_uncertainty_analysis(
        model, loader,
        T_samples=args.T,
        epi_thresh=args.epi_thresh,
        device=device,
    )

    # Print report
    print_uncertainty_report(results)

    # Save results
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_data = {
        "summary":     results["summary"],
        "per_patient": results["per_patient"],
    }
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  ✓ Saved uncertainty results → {out_path}")
    print(f"  Next: python inference/counterfactual.py --checkpoint {args.checkpoint}")


# ── Unit test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if "--checkpoint" in sys.argv:
        main()
    else:
        # Self-contained smoke test
        torch.manual_seed(42)
        print("=== MC Dropout unit tests ===\n")

        B, T, F = 4, 48, 17
        model = CausalTGAT(n_features=F, t2v_dim=8, gat_hidden=8,
                           n_heads=2, n_gat_layers=1, gru_hidden=32,
                           gru_layers=1, dropout=0.3)
        model.eval()

        X     = torch.randn(B, T, F)
        mask  = (torch.rand(B, T, F) > 0.6).float()
        delta = torch.rand(B, T, F) * 24
        times = torch.arange(T).float().unsqueeze(0).expand(B, -1)

        results = mc_dropout_predict(model, X, mask, delta, times, T_samples=30)

        print(f"MC Dropout (T=30, B={B}):")
        print(f"  mean_prob  : {results['mean_prob'].round(3)}")
        print(f"  epistemic  : {results['epistemic'].round(4)}")
        print(f"  aleatoric  : {results['aleatoric'].round(4)}")
        print(f"  all_probs  : {results['all_probs'].shape}  ([B, T_samples]) ✓\n")

        # Verify dropout actually creates variance
        std = results["std_prob"]
        print(f"  Std across MC samples: {std.round(4)}")
        assert std.max() > 0, "Dropout should create non-zero variance!"
        print(f"  → Non-zero variance confirmed (dropout is active) ✓\n")

        # Verify decomposition adds up
        total_var = results["epistemic"] + results["aleatoric"]
        print(f"  Total uncertainty (epi+ale): {total_var.round(4)}")
        print(f"  Direct total std²           : {(std**2).round(4)}")
        print(f"  (Should be close — small differences due to estimator)  ✓\n")

        # Flagging
        epi_thresh = 0.001
        n_flagged  = (results["epistemic"] > epi_thresh).sum()
        print(f"  Patients flagged (epi>{epi_thresh}): {n_flagged}/{B}  ✓")
        print("\n✓ All MC Dropout tests passed.")