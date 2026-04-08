"""
training/trainer.py
===================
Full training loop for the Causal TGAT model.

Features
--------
  - Combined Cox + Focal loss (configurable α)
  - Early stopping on validation AUROC
  - Gradient clipping (prevents exploding gradients in deep GRU)
  - Learning rate warmup + cosine decay scheduler
  - Per-epoch logging of AUROC, C-index, ECE, Cox loss, BCE loss
  - Model checkpointing (best val AUROC)
  - Ablation mode: swap causal DAG for fully-connected / Granger graph
  - Rich terminal output with epoch progress

Usage
-----
    # Standard training
    python training/trainer.py

    # With causal DAG from NOTEARS
    python training/trainer.py --dag causal/dag.npy

    # Ablation: fully-connected graph
    python training/trainer.py --adj-mode full

    # Ablation: Granger causality graph
    python training/trainer.py --adj-mode granger --dag causal/granger_adj.npy
"""

import argparse
import json
import time
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# ── Project imports ────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))
from data.dataset    import build_dataloaders, get_feature_names, get_class_weights
from models.causal_tgat import CausalTGAT
from training.cox_loss   import CombinedLoss, compute_metrics
from training.focal_loss import FocalLoss, get_focal_alpha_from_dataset


# ── Default hyperparameters ────────────────────────────────────────────────
DEFAULT_CONFIG = {
    # Data
    "processed_dir":  "data/processed/",
    "batch_size":     8,         # small batch for demo; scale to 32 for full MIMIC-IV
    # Model
    "n_features":     17,
    "t2v_dim":        16,
    "gat_hidden":     32,
    "n_heads":        4,
    "n_gat_layers":   2,
    "gru_hidden":     128,
    "gru_layers":     2,
    "dropout":        0.3,
    # Training
    "epochs":         80,
    "lr":             3e-4,
    "weight_decay":   1e-4,
    "grad_clip":      1.0,       # max gradient norm
    "cox_alpha":      0.5,       # weight of Cox loss (1-alpha = BCE weight)
    "focal_gamma":    2.0,
    "warmup_epochs":  5,
    # Early stopping
    "patience":       15,        # epochs without val AUROC improvement
    "min_delta":      1e-4,
    # Output
    "checkpoint_dir": "checkpoints/",
    "adj_mode":       "causal",  # "causal" | "full" | "none"
    "dag_path":       None,      # path to .npy causal adjacency matrix
}


class EarlyStopping:
    """
    Stop training when validation metric doesn't improve for `patience` epochs.
    Saves the best model state dict.
    """

    def __init__(self, patience: int = 15, min_delta: float = 1e-4, mode: str = "max"):
        self.patience   = patience
        self.min_delta  = min_delta
        self.mode       = mode
        self.best_score = None
        self.counter    = 0
        self.best_state = None

    def __call__(self, score: float, model: nn.Module) -> bool:
        """Returns True if training should stop."""
        if self.best_score is None:
            self.best_score = score
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
            return False

        improved = (
            score > self.best_score + self.min_delta
            if self.mode == "max"
            else score < self.best_score - self.min_delta
        )

        if improved:
            self.best_score = score
            self.counter    = 0
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1

        return self.counter >= self.patience

    def restore_best(self, model: nn.Module):
        """Load the best model weights back into the model."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


class WarmupCosineScheduler:
    """
    Linear warmup for `warmup_epochs`, then cosine annealing.
    Modifies the optimizer's learning rate in-place.
    """

    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int, base_lr: float):
        self.optimizer      = optimizer
        self.warmup_epochs  = warmup_epochs
        self.total_epochs   = total_epochs
        self.base_lr        = base_lr

    def step(self, epoch: int):
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / max(
                1, self.total_epochs - self.warmup_epochs
            )
            lr = self.base_lr * 0.5 * (1 + np.cos(np.pi * progress))
            lr = max(lr, self.base_lr * 0.01)  # floor at 1% of base LR

        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr


def run_epoch(
    model:      CausalTGAT,
    loader,
    criterion:  CombinedLoss,
    optimizer:  Optional[torch.optim.Optimizer],
    device:     torch.device,
    grad_clip:  float = 1.0,
    is_train:   bool  = True,
) -> dict:
    """
    Run one epoch (train or eval).

    Returns
    -------
    dict with keys: loss_total, loss_cox, loss_bce,
                    auroc, auprc, c_index, ece, n_events
    """
    model.train() if is_train else model.eval()

    all_probs     = []
    all_risks     = []
    all_labels    = []
    all_durations = []
    total_loss    = 0.0
    total_cox     = 0.0
    total_bce     = 0.0
    n_batches     = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()

    with ctx:
        for batch in loader:
            X, mask, delta, times, lengths, y, meta = batch
            X      = X.to(device)
            mask   = mask.to(device)
            delta  = delta.to(device)
            times  = times.to(device)
            lengths= lengths.to(device)
            y      = y.to(device)

            # Duration = LOS (censored patients contribute to risk set)
            durations = torch.tensor(
                [m["los_hours"] for m in meta], dtype=torch.float32
            ).to(device)

            # Forward pass
            out = model(X, mask, delta, times, lengths)

            # Loss
            loss, components = criterion(
                out["mortality_logit"],
                out["risk_score"],
                y,
                durations,
            )

            if is_train and optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            # Collect predictions
            all_probs.append(out["mortality_prob"].squeeze(-1).detach().cpu())
            all_risks.append(out["risk_score"].squeeze(-1).detach().cpu())
            all_labels.append(y.detach().cpu())
            all_durations.append(durations.detach().cpu())

            total_loss += components["total"]
            total_cox  += components["cox"]
            total_bce  += components["bce"]
            n_batches  += 1

    all_probs     = torch.cat(all_probs).numpy()
    all_risks     = torch.cat(all_risks).numpy()
    all_labels    = torch.cat(all_labels).numpy()
    all_durations = torch.cat(all_durations).numpy()

    metrics = compute_metrics(all_probs, all_risks, all_labels, all_durations)
    metrics["loss_total"] = total_loss / max(n_batches, 1)
    metrics["loss_cox"]   = total_cox  / max(n_batches, 1)
    metrics["loss_bce"]   = total_bce  / max(n_batches, 1)

    return metrics


def format_metrics(metrics: dict, prefix: str = "") -> str:
    """Format metrics dict into a compact one-line string."""
    def fmt(v):
        if isinstance(v, float) and not np.isnan(v):
            return f"{v:.4f}"
        return "  nan"

    parts = [
        f"loss={fmt(metrics.get('loss_total', float('nan')))}",
        f"AUROC={fmt(metrics.get('auroc', float('nan')))}",
        f"C-idx={fmt(metrics.get('c_index', float('nan')))}",
        f"ECE={fmt(metrics.get('ece', float('nan')))}",
        f"evt={metrics.get('n_events', 0)}",
    ]
    return f"{prefix}  " + "  ".join(parts)


def train(config: dict):
    """Full training run."""

    # ── Setup ──────────────────────────────────────────────────────────────
    device = torch.device("cpu")   # CPU only as specified in project constraints
    Path(config["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)

    print("\n" + "═"*65)
    print("  Causal TGAT — ICU Mortality Prediction")
    print("  Device:", device)
    print("═"*65 + "\n")

    # ── Data ──────────────────────────────────────────────────────────────
    print("Loading data...")
    train_loader, val_loader, test_loader, stats = build_dataloaders(
        processed_dir=config["processed_dir"],
        batch_size=config["batch_size"],
        augment_train=True,
    )

    features = get_feature_names(config["processed_dir"])
    print(f"  Features: {features}\n")

    # Class weights for focal loss
    n_train_samples = len(train_loader.dataset)
    n_pos = sum(s["y"] for s in train_loader.dataset.samples)
    n_neg = n_train_samples - n_pos
    focal_alpha = get_focal_alpha_from_dataset(n_pos, n_neg)
    pos_weight  = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)
    print(f"  Class balance: {n_pos} positive / {n_neg} negative")
    print(f"  Focal α={focal_alpha:.3f}  pos_weight={pos_weight.item():.2f}\n")

    # ── Model ──────────────────────────────────────────────────────────────
    print("Building model...")
    model = CausalTGAT(
        n_features   = config["n_features"],
        t2v_dim      = config["t2v_dim"],
        gat_hidden   = config["gat_hidden"],
        n_heads      = config["n_heads"],
        n_gat_layers = config["n_gat_layers"],
        gru_hidden   = config["gru_hidden"],
        gru_layers   = config["gru_layers"],
        dropout      = config["dropout"],
    ).to(device)

    print(f"  Parameters: {model.count_parameters():,}")

    # ── Causal adjacency ───────────────────────────────────────────────────
    adj_mode = config.get("adj_mode", "causal")
    dag_path = config.get("dag_path")

    if adj_mode == "causal" and dag_path and Path(dag_path).exists():
        print(f"\nLoading causal DAG from {dag_path}...")
        adj = torch.tensor(np.load(dag_path), dtype=torch.float32)
        model.set_causal_adj(adj)
    elif adj_mode == "full":
        print("\nAblation: fully-connected graph (no causal constraint)")
        adj_full = torch.ones(config["n_features"], config["n_features"])
        adj_full.fill_diagonal_(0)
        model.set_causal_adj(adj_full)
    elif adj_mode == "none":
        print("\nAblation: no graph (identity adjacency = self-loops only)")
        model.set_causal_adj(torch.eye(config["n_features"]))
    else:
        print("\nNo causal DAG provided — using default fully-connected graph.")
        print("  → Run causal/notears.py first to get a causal DAG.\n")

    # ── Loss and optimiser ─────────────────────────────────────────────────
    criterion = CombinedLoss(
        alpha=config["cox_alpha"],
        pos_weight=pos_weight,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )

    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=config["warmup_epochs"],
        total_epochs=config["epochs"],
        base_lr=config["lr"],
    )

    early_stop = EarlyStopping(
        patience=config["patience"],
        min_delta=config["min_delta"],
        mode="max",   # maximise AUROC
    )

    # ── Training loop ──────────────────────────────────────────────────────
    history = {"train": [], "val": []}
    best_val_auroc = 0.0
    t_start = time.time()

    print("\n" + "─"*65)
    print(f"  Training for up to {config['epochs']} epochs  "
          f"(patience={config['patience']})")
    print("─"*65)
    print(f"  {'Epoch':>5}  {'LR':>8}  "
          f"{'Train AUROC':>11}  {'Val AUROC':>10}  {'Val C-idx':>9}  "
          f"{'Val ECE':>8}  {'Events':>7}")
    print("  " + "─"*63)

    for epoch in range(config["epochs"]):
        lr = scheduler.step(epoch)

        # Train
        train_metrics = run_epoch(
            model, train_loader, criterion, optimizer, device,
            grad_clip=config["grad_clip"], is_train=True,
        )

        # Validate
        val_metrics = run_epoch(
            model, val_loader, criterion, optimizer=None, device=device,
            is_train=False,
        )

        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        # Log
        t_auroc  = train_metrics.get("auroc",   float("nan"))
        v_auroc  = val_metrics.get("auroc",     float("nan"))
        v_cidx   = val_metrics.get("c_index",   float("nan"))
        v_ece    = val_metrics.get("ece",        float("nan"))
        v_events = val_metrics.get("n_events",   0)

        best_marker = " ★" if (not np.isnan(v_auroc) and v_auroc > best_val_auroc) else ""
        if not np.isnan(v_auroc) and v_auroc > best_val_auroc:
            best_val_auroc = v_auroc

        def f(x): return f"{x:.4f}" if not np.isnan(x) else "  nan"

        print(
            f"  {epoch+1:>5}  {lr:>8.2e}  "
            f"{f(t_auroc):>11}  {f(v_auroc):>10}  {f(v_cidx):>9}  "
            f"{f(v_ece):>8}  {v_events:>7}{best_marker}"
        )

        # Checkpoint
        ckpt_dir = Path(config["checkpoint_dir"])
        if not np.isnan(v_auroc) and v_auroc == best_val_auroc:
            ckpt_path = ckpt_dir / "best.pt"
            torch.save({
                "epoch":       epoch + 1,
                "model_state": model.state_dict(),
                "val_auroc":   v_auroc,
                "val_c_index": v_cidx,
                "val_ece":     v_ece,
                "config":      config,
            }, ckpt_path)

        # Early stopping
        stop_score = v_auroc if not np.isnan(v_auroc) else 0.0
        if early_stop(stop_score, model):
            print(f"\n  Early stopping at epoch {epoch+1} "
                  f"(no improvement for {config['patience']} epochs)")
            break

    # ── Restore best weights ───────────────────────────────────────────────
    early_stop.restore_best(model)
    elapsed = time.time() - t_start

    print("\n" + "─"*65)
    print(f"  Training complete in {elapsed/60:.1f} min")
    print(f"  Best val AUROC: {best_val_auroc:.4f}")
    print("─"*65 + "\n")

    # ── Final test evaluation ──────────────────────────────────────────────
    print("Evaluating on test set...")
    test_metrics = run_epoch(
        model, test_loader, criterion, optimizer=None, device=device,
        is_train=False,
    )

    print("\n  ┌─────────────────────────────────────┐")
    print("  │         Final Test Results          │")
    print("  ├─────────────────────────────────────┤")
    for key, val in [
        ("AUROC",   test_metrics.get("auroc",   float("nan"))),
        ("AUPRC",   test_metrics.get("auprc",   float("nan"))),
        ("C-index", test_metrics.get("c_index", float("nan"))),
        ("ECE",     test_metrics.get("ece",     float("nan"))),
    ]:
        v = f"{val:.4f}" if not np.isnan(val) else "  nan"
        print(f"  │  {key:<12} {v:>10}                  │")
    print(f"  │  Events      {test_metrics.get('n_events',0):>10}                  │")
    print("  └─────────────────────────────────────┘\n")

    # ── Save history and final checkpoint ─────────────────────────────────
    history_path = Path(config["checkpoint_dir"]) / "history.json"
    with open(history_path, "w") as f:
        # Convert nan to null for JSON
        def clean(x):
            if isinstance(x, float) and np.isnan(x):
                return None
            return x
        clean_history = {
            split: [{k: clean(v) for k, v in m.items()} for m in epochs]
            for split, epochs in history.items()
        }
        json.dump({"history": clean_history, "test": {k: clean(v) for k, v in test_metrics.items()}}, f, indent=2)

    final_ckpt = Path(config["checkpoint_dir"]) / "final.pt"
    torch.save({
        "model_state": model.state_dict(),
        "test_metrics": test_metrics,
        "config": config,
    }, final_ckpt)

    print(f"  Saved: checkpoints/best.pt  (best val AUROC)")
    print(f"  Saved: checkpoints/final.pt (last epoch)")
    print(f"  Saved: checkpoints/history.json\n")
    print("  Next steps:")
    print("    python inference/mc_dropout.py --checkpoint checkpoints/best.pt")
    print("    python inference/counterfactual.py --checkpoint checkpoints/best.pt")

    return model, test_metrics, history


def main():
    parser = argparse.ArgumentParser(description="Train Causal TGAT")
    parser.add_argument("--processed-dir",  default=DEFAULT_CONFIG["processed_dir"])
    parser.add_argument("--dag",            default=None, dest="dag_path",
                        help="Path to NOTEARS DAG .npy file")
    parser.add_argument("--adj-mode",       default="full",
                        choices=["causal", "full", "none"],
                        help="Graph type: causal|full|none")
    parser.add_argument("--epochs",         type=int,   default=DEFAULT_CONFIG["epochs"])
    parser.add_argument("--batch-size",     type=int,   default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--lr",             type=float, default=DEFAULT_CONFIG["lr"])
    parser.add_argument("--dropout",        type=float, default=DEFAULT_CONFIG["dropout"])
    parser.add_argument("--gru-hidden",     type=int,   default=DEFAULT_CONFIG["gru_hidden"])
    parser.add_argument("--gat-hidden",     type=int,   default=DEFAULT_CONFIG["gat_hidden"])
    parser.add_argument("--n-heads",        type=int,   default=DEFAULT_CONFIG["n_heads"])
    parser.add_argument("--cox-alpha",      type=float, default=DEFAULT_CONFIG["cox_alpha"])
    parser.add_argument("--patience",       type=int,   default=DEFAULT_CONFIG["patience"])
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CONFIG["checkpoint_dir"])
    parser.add_argument("--seed",           type=int,   default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    config = {**DEFAULT_CONFIG}
    config.update({
        "processed_dir":  args.processed_dir,
        "dag_path":       args.dag_path,
        "adj_mode":       args.adj_mode,
        "epochs":         args.epochs,
        "batch_size":     args.batch_size,
        "lr":             args.lr,
        "dropout":        args.dropout,
        "gru_hidden":     args.gru_hidden,
        "gat_hidden":     args.gat_hidden,
        "n_heads":        args.n_heads,
        "cox_alpha":      args.cox_alpha,
        "patience":       args.patience,
        "checkpoint_dir": args.checkpoint_dir,
    })

    train(config)


if __name__ == "__main__":
    main()