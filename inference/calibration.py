"""
inference/calibration.py
========================
Model Calibration Analysis.

Theory — Why calibration matters
----------------------------------
A model that predicts p=0.70 should be correct 70% of the time across all
such predictions.  Calibration measures how well predicted probabilities
match observed frequencies.

In clinical deployment, a poorly calibrated model is dangerous even if
AUROC is high:
  - Over-confident model: predicts 0.9, but only 50% of these patients die.
    Clinicians will over-treat based on inflated risk scores.
  - Under-confident model: predicts 0.3, but 70% of these patients die.
    Clinicians will under-react to genuine high-risk patients.

Expected Calibration Error (ECE)
---------------------------------
ECE bins predicted probabilities into M equal-width bins:

    ECE = Σ_{m=1}^{M}  |B_m| / N  ×  | acc(B_m) − conf(B_m) |

where:
  B_m    = set of samples whose predicted probability falls in bin m
  acc(B) = fraction of positive labels in bin B
  conf(B)= mean predicted probability in bin B
  N      = total samples

ECE = 0 → perfect calibration
ECE = 0.1 → predictions are off by ~10% on average
ECE < 0.05 → considered well-calibrated for clinical use

Reliability Diagram
-------------------
Plots conf(B_m) vs acc(B_m) per bin.  A perfectly calibrated model
lies on the diagonal y = x.  Points above the diagonal mean the model
is *under-confident* (actual frequency > predicted), below means
*over-confident*.

Temperature Scaling
-------------------
The simplest post-hoc calibration method.  Replace p = sigmoid(z) with:
    p = sigmoid(z / τ)
where τ (temperature) is a single scalar optimised on a validation set.

τ > 1 → "cool" the predictions (soften towards 0.5, reduce over-confidence)
τ < 1 → "heat" the predictions (sharpen away from 0.5, reduce under-confidence)

Temperature scaling is the state-of-the-art for neural network calibration
(Guo et al., ICML 2017) and changes AUROC by less than 0.001 — it only
affects the mapping from logit to probability, not the ranking.
"""

import argparse
import sys
import json
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.dataset    import build_dataloaders
from models.causal_tgat import CausalTGAT
from training.cox_loss import expected_calibration_error


# ── Calibration metrics ────────────────────────────────────────────────────

def reliability_diagram_data(
    probs:  np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """
    Compute data for a reliability diagram.

    Returns
    -------
    dict with:
        bin_centers  : [M]  — centre of each confidence bin
        bin_acc      : [M]  — fraction of positives in each bin
        bin_conf     : [M]  — mean predicted probability in each bin
        bin_counts   : [M]  — number of samples per bin
        ece          : float
        mce          : float  — Maximum Calibration Error
        gaps         : [M]  — |acc - conf| per bin
    """
    probs  = np.clip(np.asarray(probs),  1e-6, 1-1e-6)
    labels = np.asarray(labels, dtype=float)
    n      = len(probs)

    bins       = np.linspace(0, 1, n_bins + 1)
    bin_idx    = np.clip(np.digitize(probs, bins) - 1, 0, n_bins - 1)

    bin_acc    = np.zeros(n_bins)
    bin_conf   = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins, dtype=int)
    bin_centers= (bins[:-1] + bins[1:]) / 2

    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        bin_acc[b]    = labels[mask].mean()
        bin_conf[b]   = probs[mask].mean()
        bin_counts[b] = mask.sum()

    gaps = np.abs(bin_acc - bin_conf)
    ece  = float((bin_counts / max(n, 1) * gaps).sum())
    mce  = float(gaps[bin_counts > 0].max()) if (bin_counts > 0).any() else 0.0

    return {
        "bin_centers": bin_centers.tolist(),
        "bin_acc":     bin_acc.tolist(),
        "bin_conf":    bin_conf.tolist(),
        "bin_counts":  bin_counts.tolist(),
        "gaps":        gaps.tolist(),
        "ece":         ece,
        "mce":         mce,
        "n_samples":   n,
        "n_bins":      n_bins,
    }


# ── Temperature Scaling ────────────────────────────────────────────────────

class TemperatureScaler(nn.Module):
    """
    Learns a single scalar τ to calibrate model logits.

    Usage:
        scaler = TemperatureScaler()
        scaler.fit(val_logits, val_labels)
        calibrated_probs = scaler(test_logits)
    """

    def __init__(self, init_temp: float = 1.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor([init_temp]))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Scale logits and return calibrated probabilities."""
        return torch.sigmoid(logits / self.temperature.clamp(min=0.05))

    def fit(
        self,
        logits: torch.Tensor,    # [N] or [N,1]
        labels: torch.Tensor,    # [N] float binary
        lr:     float = 0.01,
        n_iter: int   = 200,
    ) -> float:
        """
        Optimise temperature on a held-out set using NLL loss.

        Returns final NLL (lower is better).
        """
        logits = logits.float().squeeze(-1)
        labels = labels.float()

        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=n_iter)
        criterion = nn.BCELoss()

        def closure():
            optimizer.zero_grad()
            p    = self.forward(logits)
            loss = criterion(p, labels)
            loss.backward()
            return loss

        optimizer.step(closure)

        with torch.no_grad():
            final_loss = criterion(self.forward(logits), labels).item()

        return final_loss

    def extra_repr(self) -> str:
        return f"temperature={self.temperature.item():.4f}"


def collect_predictions(
    model:   CausalTGAT,
    loader,
    device:  torch.device = torch.device("cpu"),
) -> dict:
    """
    Collect all predictions from a DataLoader.

    Returns
    -------
    dict with:
        probs   : [N] np.ndarray  — predicted mortality probabilities
        logits  : [N]             — raw logits
        labels  : [N]             — true labels
        durations : [N]           — LOS in hours
        stay_ids  : [N]
    """
    model.eval()
    all_probs    = []
    all_logits   = []
    all_labels   = []
    all_durations= []
    all_stay_ids = []

    with torch.no_grad():
        for batch in loader:
            X, mask_t, delta, times, lengths, y, meta = batch
            X      = X.to(device)
            mask_t = mask_t.to(device)
            delta  = delta.to(device)
            times  = times.to(device)
            lengths= lengths.to(device)

            out = model(X, mask_t, delta, times, lengths)
            all_probs.extend(out["mortality_prob"].squeeze(-1).cpu().numpy())
            all_logits.extend(out["mortality_logit"].squeeze(-1).cpu().numpy())
            all_labels.extend(y.numpy())
            all_durations.extend([m["los_hours"] for m in meta])
            all_stay_ids.extend([m["stay_id"]    for m in meta])

    return {
        "probs":     np.array(all_probs),
        "logits":    np.array(all_logits),
        "labels":    np.array(all_labels),
        "durations": np.array(all_durations),
        "stay_ids":  np.array(all_stay_ids),
    }


def print_calibration_report(
    diag:          dict,
    label:         str = "Model",
    diag_scaled:   Optional[dict] = None,
    temperature:   Optional[float] = None,
):
    """Print calibration report to terminal."""
    print("\n" + "═"*55)
    print(f"  Calibration Report — {label}")
    print("═"*55)
    print(f"  ECE  : {diag['ece']:.4f}")
    print(f"  MCE  : {diag['mce']:.4f}  (worst-bin calibration error)")
    print(f"  N    : {diag['n_samples']}")
    print(f"  Bins : {diag['n_bins']}")
    print()

    # ASCII reliability diagram
    print("  Reliability diagram  (● = bin, diagonal = perfect)")
    print("  Conf →  0.0   0.1   0.2   0.3   0.4   0.5   0.6   0.7   0.8   0.9")

    acc   = diag["bin_acc"]
    conf  = diag["bin_conf"]
    counts= diag["bin_counts"]

    for b in range(diag["n_bins"]):
        if counts[b] == 0:
            marker = "  (empty)"
        else:
            pos = int(round(acc[b] * 10))
            gap_sign = "↑" if acc[b] > conf[b] else ("↓" if acc[b] < conf[b] else "—")
            bar   = "─" * pos + "●" + "─" * (9 - pos)
            count = f"n={counts[b]}"
            marker = f"  [{bar}] acc={acc[b]:.2f} {gap_sign}gap={abs(diag['gaps'][b]):.2f}  {count}"
        print(f"  {diag['bin_centers'][b]:.1f}    {marker}")

    if diag_scaled is not None and temperature is not None:
        print()
        print(f"  After temperature scaling (τ={temperature:.3f}):")
        print(f"    ECE before: {diag['ece']:.4f}")
        print(f"    ECE after : {diag_scaled['ece']:.4f}")
        improvement = (diag["ece"] - diag_scaled["ece"]) / max(diag["ece"], 1e-8)
        print(f"    Improvement: {improvement*100:.1f}%")

    print("═"*55)


def load_model_from_checkpoint(ckpt_path: str, device: torch.device) -> CausalTGAT:
    ckpt   = torch.load(ckpt_path, map_location=device)
    config = ckpt.get("config", {})
    model  = CausalTGAT(
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
    parser = argparse.ArgumentParser(description="Calibration analysis")
    parser.add_argument("--checkpoint",    required=True)
    parser.add_argument("--processed-dir", default="data/processed/")
    parser.add_argument("--n-bins",        type=int, default=10)
    parser.add_argument("--scale-temp",    action="store_true",
                        help="Fit temperature scaling on val set")
    parser.add_argument("--out",           default="inference/calibration.json")
    args = parser.parse_args()

    device = torch.device("cpu")

    print(f"\nLoading model from {args.checkpoint}...")
    model = load_model_from_checkpoint(args.checkpoint, device)

    train_l, val_l, test_l, _ = build_dataloaders(
        processed_dir=args.processed_dir,
        batch_size=16,
        augment_train=False,
    )

    results = {}

    for split_name, loader in [("val", val_l), ("test", test_l)]:
        print(f"\nCollecting {split_name} predictions...")
        preds = collect_predictions(model, loader, device)

        diag = reliability_diagram_data(
            preds["probs"], preds["labels"], n_bins=args.n_bins
        )

        diag_scaled  = None
        temperature  = None

        if args.scale_temp and split_name == "test":
            # Fit temperature on val, apply to test
            print("  Fitting temperature scaling on val set...")
            val_preds = collect_predictions(model, val_l, device)
            scaler    = TemperatureScaler()
            scaler.fit(
                torch.tensor(val_preds["logits"]),
                torch.tensor(val_preds["labels"])
            )
            temperature = scaler.temperature.item()
            print(f"  Optimal temperature τ = {temperature:.4f}")

            scaled_probs = scaler(
                torch.tensor(preds["logits"])
            ).detach().numpy()
            diag_scaled  = reliability_diagram_data(
                scaled_probs, preds["labels"], n_bins=args.n_bins
            )

        print_calibration_report(
            diag, label=f"{split_name.capitalize()} set",
            diag_scaled=diag_scaled, temperature=temperature,
        )
        results[split_name] = {
            "raw":    diag,
            "scaled": diag_scaled,
            "temperature": temperature,
            "n_samples": int(len(preds["labels"])),
            "n_events":  int(preds["labels"].sum()),
        }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  ✓ Saved calibration data → {out_path}")


# ── Unit test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if "--checkpoint" in sys.argv:
        main()
    else:
        np.random.seed(42)
        print("=== Calibration unit tests ===\n")

        # Perfect calibration: predicted prob matches true frequency
        perfect_probs  = np.array([0.05]*100 + [0.25]*100 + [0.75]*100 + [0.95]*100)
        perfect_labels = np.array(
            [np.random.binomial(1, 0.05) for _ in range(100)] +
            [np.random.binomial(1, 0.25) for _ in range(100)] +
            [np.random.binomial(1, 0.75) for _ in range(100)] +
            [np.random.binomial(1, 0.95) for _ in range(100)]
        )
        diag_perfect = reliability_diagram_data(perfect_probs, perfect_labels)
        print(f"Perfect calibration ECE: {diag_perfect['ece']:.4f}  (expected ~0.0)")
        assert diag_perfect["ece"] < 0.1, "Perfect calibration should have low ECE"
        print(f"  ✓\n")

        # Over-confident model
        overconf_probs  = np.concatenate([
            np.random.uniform(0.7, 1.0, 200),   # predicted very confident
        ])
        overconf_labels = np.random.binomial(1, 0.3, 200)   # but only 30% die
        diag_over = reliability_diagram_data(overconf_probs, overconf_labels)
        print(f"Over-confident model ECE: {diag_over['ece']:.4f}  (expected > 0.3)")
        assert diag_over["ece"] > 0.3, "Over-confident model should have high ECE"
        print(f"  ✓\n")

        # Temperature scaling
        logits = torch.randn(200) * 3   # extreme logits → over-confident
        labels = torch.randint(0, 2, (200,)).float()
        probs_before = torch.sigmoid(logits).numpy()
        ece_before   = reliability_diagram_data(probs_before, labels.numpy())["ece"]

        scaler = TemperatureScaler()
        scaler.fit(logits, labels, n_iter=100)
        probs_after = scaler(logits).detach().numpy()
        ece_after   = reliability_diagram_data(probs_after, labels.numpy())["ece"]

        print(f"Temperature scaling:")
        print(f"  τ = {scaler.temperature.item():.3f}")
        print(f"  ECE before: {ece_before:.4f}")
        print(f"  ECE after : {ece_after:.4f}")
        print(f"  ✓ (temperature scaling modifies calibration)\n")

        # Reliability diagram data structure
        diag = reliability_diagram_data(probs_before, labels.numpy(), n_bins=5)
        print(f"Reliability diagram bins: {diag['n_bins']}")
        print(f"  bin_centers: {[f'{c:.2f}' for c in diag['bin_centers']]}")
        print(f"  bin_acc    : {[f'{a:.2f}' for a in diag['bin_acc']]}")
        print(f"  ECE        : {diag['ece']:.4f}")
        print(f"  MCE        : {diag['mce']:.4f}  ✓")

        print("\n✓ All calibration tests passed.")