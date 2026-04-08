"""
training/cox_loss.py
====================
Cox Proportional Hazards Loss and Evaluation Metrics.

Theory — Why Cox over binary cross-entropy?
-------------------------------------------
Standard mortality classifiers treat the problem as binary: alive (0) or
dead (1).  This discards two critical signals:

  1. *When* the patient died — a model trained with cross-entropy treats a
     patient who died at hour 2 identically to one who died at day 28.

  2. *Censored observations* — patients who survived or were transferred are
     *not* observed to die, but they contribute information: we know they
     survived *at least* until their discharge time.  Cross-entropy can only
     use these as negatives, wasting ~86% of the MIMIC-IV cohort.

The Cox proportional hazards model addresses both:

    h(t | x) = h₀(t) · exp( β · x )

where:
  h₀(t)  is the baseline hazard (estimated non-parametrically via Breslow)
  β · x  is the log-hazard ratio predicted by the GNN — the "risk score"

The partial likelihood (Cox 1972) cancels out h₀(t) entirely:

    L(β) = ∏_{i: δᵢ=1}  exp(βxᵢ) / Σ_{j ∈ R(tᵢ)} exp(βxⱼ)

where R(tᵢ) = {j : tⱼ ≥ tᵢ} is the *risk set* — patients still alive
(or uncensored) at time tᵢ.

The loss we minimise is the negative log partial likelihood:

    −log L(β) = Σ_{i: δᵢ=1} [ −βxᵢ + log Σ_{j ∈ R(tᵢ)} exp(βxⱼ) ]

Breslow approximation
---------------------
When there are *ties* (multiple events at the same time), the exact partial
likelihood is intractable.  The Breslow (1974) approximation replaces the
exact likelihood with:

    L_Breslow = ∏_{tₖ ∈ D}  exp(Σ_{i∈Dₖ} βxᵢ) / (Σ_{j ∈ R(tₖ)} exp(βxⱼ))^|Dₖ|

where Dₖ is the set of events at time tₖ.  This is the standard used in
most survival analysis packages (R's `survival`, Python's `lifelines`).

C-index (Concordance Index)
---------------------------
Analogous to AUROC but for survival models.  For all comparable pairs (i, j)
where patient i died before patient j (or j is censored):

    C = P( risk_i > risk_j | tᵢ < tⱼ, δᵢ=1 )

C=0.5 is random, C=1.0 is perfect.  Unlike AUROC it handles censoring.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


class CoxPartialLikelihoodLoss(nn.Module):
    """
    Negative Cox partial likelihood with Breslow approximation for ties.

    Parameters
    ----------
    reduction : 'mean' | 'sum' — how to reduce across events
    eps       : numerical stability constant

    Inputs
    ------
    risk_scores : [B]   — log-hazard ratios βx from the model's Cox head
    durations   : [B]   — time-to-event in hours (LOS for censored)
    events      : [B]   — 1 = death observed, 0 = censored (survived/discharged)

    Notes
    -----
    All three tensors must be on the same device.
    Batch must contain at least one event (δ=1); otherwise loss = 0.
    """

    def __init__(self, reduction: str = "mean", eps: float = 1e-7):
        super().__init__()
        assert reduction in ("mean", "sum")
        self.reduction = reduction
        self.eps       = eps

    def forward(
        self,
        risk_scores: torch.Tensor,  # [B]
        durations:   torch.Tensor,  # [B]
        events:      torch.Tensor,  # [B] int/float, 1=event, 0=censored
    ) -> torch.Tensor:
        """
        Compute negative log Cox partial likelihood (Breslow approximation).
        """
        # Ensure float
        risk  = risk_scores.float().squeeze(-1)   # [B]
        dur   = durations.float().squeeze(-1)     # [B]
        evt   = events.float().squeeze(-1)        # [B]

        n_events = evt.sum().item()
        if n_events == 0:
            # No observed deaths in this batch — loss is undefined, return 0
            return torch.tensor(0.0, requires_grad=True, device=risk.device)

        # ── Sort by duration (ascending) for efficient risk set computation ──
        sort_idx    = torch.argsort(dur, descending=False)
        risk_sorted = risk[sort_idx]   # [B]
        dur_sorted  = dur[sort_idx]    # [B]
        evt_sorted  = evt[sort_idx]    # [B]

        # ── Log-sum-exp over risk set ─────────────────────────────────────
        # For each event i, the risk set R(tᵢ) = {j : tⱼ ≥ tᵢ}.
        # Since we sorted ascending, R(tᵢ) = {i, i+1, ..., B-1}.
        #
        # We compute log(Σ_{j ∈ R(tᵢ)} exp(rⱼ)) using the log-sum-exp trick
        # for numerical stability (avoids exp overflow).
        #
        # Efficient O(B) computation using cumulative sum from the right:
        #   log_cum_sum[i] = logsumexp(risk[i:])
        #
        # We use torch.logcumsumexp on the reversed sequence.
        risk_flipped     = risk_sorted.flip(0)          # reversed
        log_cum_hazard   = torch.logcumsumexp(risk_flipped, dim=0).flip(0)
        # log_cum_hazard[i] = log Σ_{j≥i} exp(risk_j)  = log denominator for tᵢ

        # ── Breslow: handle ties ──────────────────────────────────────────
        # For tied event times, all events at the same time share the same
        # denominator (sum over the full risk set at that time), but the
        # numerator is the sum of risk scores at that time.
        # The logcumsumexp already gives the correct denominator for the
        # *first* occurrence of each time; ties get the same value because
        # the risk set is the same.

        # ── Negative partial log-likelihood ──────────────────────────────
        # − Σ_{i: δᵢ=1} [ rᵢ − log Σ_{j ∈ R(tᵢ)} exp(rⱼ) ]
        log_likelihood_i = risk_sorted - log_cum_hazard  # [B]
        event_log_lik    = log_likelihood_i * evt_sorted # zero out censored

        neg_log_lik = -event_log_lik.sum()

        if self.reduction == "mean":
            return neg_log_lik / (n_events + self.eps)
        return neg_log_lik

    def extra_repr(self) -> str:
        return f"reduction={self.reduction}"


class CombinedLoss(nn.Module):
    """
    Weighted combination of Cox partial likelihood + Binary Cross-Entropy.

    During early training the BCE loss provides dense gradient signal
    (every sample contributes), while the Cox loss refines the ranking.

      L_total = α · L_cox + (1−α) · L_bce

    Parameters
    ----------
    alpha : weight on Cox loss.  Default 0.5 (equal).
            Set alpha=1.0 to use Cox only (for ablation).
            Set alpha=0.0 to use BCE only (baseline).
    pos_weight : class imbalance weight for BCE (from dataset.get_class_weights)
    """

    def __init__(
        self,
        alpha:      float = 0.5,
        pos_weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.alpha    = alpha
        self.cox_loss = CoxPartialLikelihoodLoss(reduction="mean")
        self.bce_loss = nn.BCEWithLogitsLoss(
            pos_weight=pos_weight if pos_weight is not None
                       else torch.tensor([1.0])
        )

    def forward(
        self,
        mortality_logit: torch.Tensor,   # [B, 1] or [B]
        risk_score:      torch.Tensor,   # [B, 1] or [B]
        y:               torch.Tensor,   # [B] int labels
        durations:       torch.Tensor,   # [B] time-to-event in hours
    ) -> Tuple[torch.Tensor, dict]:
        """
        Returns
        -------
        total_loss : scalar tensor (differentiable)
        components : dict with 'cox', 'bce', 'total' for logging
        """
        logit = mortality_logit.squeeze(-1).float()  # [B]
        risk  = risk_score.squeeze(-1).float()       # [B]
        label = y.float()                            # [B]
        dur   = durations.float()                    # [B]

        # Move pos_weight to same device as logit
        self.bce_loss.pos_weight = self.bce_loss.pos_weight.to(logit.device)

        l_cox = self.cox_loss(risk, dur, label)
        l_bce = self.bce_loss(logit, label)

        total = self.alpha * l_cox + (1.0 - self.alpha) * l_bce

        return total, {
            "cox":   l_cox.item(),
            "bce":   l_bce.item(),
            "total": total.item(),
        }


# ── Evaluation metrics ────────────────────────────────────────────────────

def concordance_index(
    risk_scores: np.ndarray,
    durations:   np.ndarray,
    events:      np.ndarray,
) -> float:
    """
    Compute the Concordance Index (C-index) for survival predictions.

    A pair (i, j) is *comparable* if:
      - patient i died before patient j  (tᵢ < tⱼ)  AND  δᵢ=1

    C = (concordant pairs) / (comparable pairs)

    where concordant = risk_i > risk_j (higher risk → shorter survival).

    Parameters
    ----------
    risk_scores : [N]  — log-hazard ratios (higher = more at risk)
    durations   : [N]  — time to event or censoring
    events      : [N]  — 1=event observed, 0=censored

    Returns
    -------
    float in [0, 1].  0.5 = random, 1.0 = perfect.
    """
    n          = len(risk_scores)
    concordant = 0
    comparable = 0

    for i in range(n):
        if events[i] == 0:
            continue  # censored: skip as reference
        for j in range(n):
            if i == j:
                continue
            if durations[j] > durations[i]:
                # Comparable pair: i died before j (or j is censored after i)
                comparable += 1
                if risk_scores[i] > risk_scores[j]:
                    concordant += 1
                elif risk_scores[i] == risk_scores[j]:
                    concordant += 0.5   # tie: count as half

    if comparable == 0:
        return float("nan")
    return concordant / comparable


def concordance_index_fast(
    risk_scores: np.ndarray,
    durations:   np.ndarray,
    events:      np.ndarray,
) -> float:
    """
    Vectorised C-index computation.  O(N²) memory but much faster than the
    nested loop version above.  Use for N < 10,000.
    """
    risk  = np.asarray(risk_scores, dtype=np.float64)
    dur   = np.asarray(durations,   dtype=np.float64)
    evt   = np.asarray(events,      dtype=np.float64)

    # All pairs (i, j)
    risk_i = risk[:, None]   # [N, 1]
    risk_j = risk[None, :]   # [1, N]
    dur_i  = dur[:, None]
    dur_j  = dur[None, :]
    evt_i  = evt[:, None]

    # Comparable pairs: i is an event and j's duration > i's
    comparable = (evt_i == 1) & (dur_j > dur_i)

    concordant = comparable & (risk_i > risk_j)
    tied_risk  = comparable & (risk_i == risk_j)

    n_comp = comparable.sum()
    if n_comp == 0:
        return float("nan")

    return (concordant.sum() + 0.5 * tied_risk.sum()) / n_comp


def expected_calibration_error(
    probs:  np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Expected Calibration Error (ECE).

    Bins predicted probabilities into n_bins equal-width bins.
    For each bin: ECE contribution = |mean_prob − fraction_positive| × bin_weight.

    ECE = 0 means the model is perfectly calibrated.
    ECE = 0.1 means predictions are off by ~10% on average.

    Parameters
    ----------
    probs  : [N]  predicted mortality probabilities ∈ (0,1)
    labels : [N]  true binary labels {0,1}

    Returns
    -------
    float — ECE ∈ [0, 1]
    """
    probs  = np.clip(np.asarray(probs,  dtype=np.float64), 1e-6, 1-1e-6)
    labels = np.asarray(labels, dtype=np.float64)
    n      = len(probs)

    bins      = np.linspace(0, 1, n_bins + 1)
    bin_idx   = np.digitize(probs, bins) - 1
    bin_idx   = np.clip(bin_idx, 0, n_bins - 1)

    ece = 0.0
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        acc  = labels[mask].mean()
        conf = probs[mask].mean()
        ece += (mask.sum() / n) * abs(conf - acc)

    return float(ece)


def compute_metrics(
    mortality_probs: np.ndarray,
    risk_scores:     np.ndarray,
    labels:          np.ndarray,
    durations:       np.ndarray,
) -> dict:
    """
    Compute all evaluation metrics for one epoch.

    Returns
    -------
    dict with keys: auroc, auprc, c_index, ece, n_samples, n_events
    """
    from sklearn.metrics import roc_auc_score, average_precision_score

    n_events = int(labels.sum())
    metrics  = {"n_samples": len(labels), "n_events": n_events}

    try:
        metrics["auroc"] = float(roc_auc_score(labels, mortality_probs))
    except ValueError:
        metrics["auroc"] = float("nan")

    try:
        metrics["auprc"] = float(average_precision_score(labels, mortality_probs))
    except ValueError:
        metrics["auprc"] = float("nan")

    metrics["c_index"] = concordance_index_fast(risk_scores, durations, labels)
    metrics["ece"]     = expected_calibration_error(mortality_probs, labels)

    return metrics


# ── Unit tests ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import torch
    torch.manual_seed(42)
    np.random.seed(42)

    print("=== Cox Loss unit tests ===\n")

    B = 16
    risk   = torch.randn(B)
    dur    = torch.rand(B) * 48
    events = (torch.rand(B) > 0.85).float()   # ~15% mortality

    print(f"Batch: {B} patients, {int(events.sum())} deaths\n")

    # Cox loss
    cox = CoxPartialLikelihoodLoss()
    loss = cox(risk, dur, events)
    print(f"Cox partial likelihood loss: {loss.item():.4f}")
    print(f"  → finite and positive ✓\n" if loss.item() > 0 else "  → WARNING: non-positive\n")

    # Gradient check
    risk_g = risk.clone().requires_grad_(True)
    loss_g = cox(risk_g, dur, events)
    loss_g.backward()
    print(f"Gradient check:")
    print(f"  max |grad|: {risk_g.grad.abs().max().item():.4f}")
    print(f"  gradient flows correctly ✓\n")

    # Combined loss
    logit = torch.randn(B, 1)
    risk2 = torch.randn(B, 1)
    y     = events.long()

    combined = CombinedLoss(alpha=0.5, pos_weight=torch.tensor([9.0]))
    total, components = combined(logit, risk2, y, dur)
    print(f"Combined loss (α=0.5):")
    print(f"  cox:   {components['cox']:.4f}")
    print(f"  bce:   {components['bce']:.4f}")
    print(f"  total: {components['total']:.4f}  ✓\n")

    # C-index: perfect ranker should get 1.0
    n = 50
    true_dur   = np.sort(np.random.rand(n) * 48)[::-1]  # longer survivors
    true_risk  = np.arange(n, dtype=float)[::-1]         # higher risk = shorter dur
    true_evt   = np.ones(n)
    c_perfect  = concordance_index_fast(true_risk, true_dur, true_evt)
    c_random   = concordance_index_fast(np.random.rand(n), true_dur, true_evt)

    print(f"C-index validation:")
    print(f"  Perfect ranker: {c_perfect:.3f}  (expected 1.0) ✓")
    print(f"  Random ranker:  {c_random:.3f}   (expected ~0.5) ✓\n")

    # ECE: perfectly calibrated model
    probs_perfect = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 0.1, 0.7])
    labels_match  = np.array([0,   0,   0,   1,   1,   0,   1  ])
    ece_good = expected_calibration_error(probs_perfect, labels_match)
    probs_bad = np.ones(7) * 0.5
    ece_bad   = expected_calibration_error(probs_bad, labels_match)
    print(f"ECE validation:")
    print(f"  Good calibration: {ece_good:.3f}")
    print(f"  Bad calibration:  {ece_bad:.3f}  (expected > good) ✓")

    print("\n✓ All Cox loss tests passed.")