"""Sample-efficiency-first metrics for the active-learning simulation.

Every metric is computed from the cumulative MEASURED set (the variants the
loop has spent budget on so far) against the full landscape oracle. The
primary metrics answer "how good is the best thing found, and how few
measurements did it take" -- not static correlation. Correlation (Spearman)
is provided as a secondary diagnostic and only when a fitted model is supplied.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from .oracle import Landscape


def max_activity_found(landscape: Landscape, measured_ids: Sequence[str]) -> float:
    """Best true activity among the measured variants so far (cumulative best)."""
    if len(measured_ids) == 0:
        return float("nan")
    return float(max(landscape.measure(v) for v in measured_ids))


def simple_regret(landscape: Landscape, measured_ids: Sequence[str]) -> float:
    """``global_max - best_found`` -- distance from the optimum (lower better)."""
    best = max_activity_found(landscape, measured_ids)
    if np.isnan(best):
        return float("nan")
    return float(landscape.global_max() - best)


def rounds_to_target(
    best_found_per_round: Sequence[float], threshold: float
) -> float:
    """First round index whose cumulative best reaches ``threshold``.

    Args:
        best_found_per_round: cumulative best activity at the end of each round,
            in round order (round 0, 1, ...).
        threshold: target activity.

    Returns:
        The 0-based round index of the first round that reaches the target, or
        ``float('nan')`` if the target is never reached within the trajectory.
        Sample efficiency = fewer rounds is better.
    """
    for r, best in enumerate(best_found_per_round):
        if not np.isnan(best) and best >= threshold:
            return float(r)
    return float("nan")


def top_k_recall(landscape: Landscape, measured_ids: Sequence[str], k: int) -> float:
    """Fraction of the true top-``k`` variants discovered so far (0..1)."""
    if k <= 0:
        return float("nan")
    true_top = set(landscape.top_k_ids(k))
    if not true_top:
        return float("nan")
    found = true_top.intersection(set(measured_ids))
    return float(len(found) / len(true_top))


def fraction_top_bin(
    landscape: Landscape, measured_ids: Sequence[str], quantile: float = 0.9
) -> float:
    """Fraction of MEASURED variants that fall in the true top bin (default decile).

    A binary "success" rate: of everything the loop chose to measure, how much
    of it landed in the landscape's top ``(1 - quantile)`` fraction. High values
    mean the loop is spending budget on genuinely good variants.
    """
    if len(measured_ids) == 0:
        return float("nan")
    cutoff = float(np.quantile(landscape.true_values(), quantile))
    hits = sum(1 for v in measured_ids if landscape.measure(v) >= cutoff)
    return float(hits / len(measured_ids))


def spearman(y_pred: Sequence[float], y_true: Sequence[float]) -> float:
    """Spearman rank correlation between predictions and truth (secondary).

    Returns ``nan`` if fewer than two points or no variance. Implemented with
    numpy (rank -> Pearson on ranks) to avoid a scipy dependency.
    """
    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    if y_pred.shape != y_true.shape:
        raise ValueError("spearman: y_pred and y_true must have the same shape.")
    if y_pred.size < 2:
        return float("nan")
    rp = _rankdata(y_pred)
    rt = _rankdata(y_true)
    if np.std(rp) == 0 or np.std(rt) == 0:
        return float("nan")
    return float(np.corrcoef(rp, rt)[0, 1])


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average-rank of each element (ties share the mean rank)."""
    order = np.argsort(a, kind="stable")
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(1, len(a) + 1, dtype=float)
    # Average ties.
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts), dtype=float)
    np.add.at(sums, inv, ranks)
    means = sums / counts
    return means[inv]


def compute_round_metrics(
    landscape: Landscape,
    measured_ids: Sequence[str],
    *,
    top_k: int = 10,
    target: Optional[float] = None,
    model_pred: Optional[Sequence[float]] = None,
    model_true: Optional[Sequence[float]] = None,
) -> Dict[str, float]:
    """Bundle the per-round metrics over the cumulative measured set.

    Args:
        landscape: the oracle.
        measured_ids: cumulative measured variant ids.
        top_k: k for :func:`top_k_recall`.
        target: optional target activity recorded as ``reached_target`` (0/1).
        model_pred / model_true: optional aligned arrays for :func:`spearman`.

    Note: ``rounds_to_target`` is trajectory-level (needs the whole sequence of
    per-round bests), so it is computed by the simulator, not here.
    """
    best = max_activity_found(landscape, measured_ids)
    metrics: Dict[str, float] = {
        "max_activity_found": best,
        "simple_regret": simple_regret(landscape, measured_ids),
        "top_k_recall": top_k_recall(landscape, measured_ids, top_k),
        "fraction_top_bin": fraction_top_bin(landscape, measured_ids),
    }
    if target is not None:
        metrics["reached_target"] = (
            float(0.0) if np.isnan(best) else float(1.0 if best >= target else 0.0)
        )
    if model_pred is not None and model_true is not None:
        metrics["spearman"] = spearman(model_pred, model_true)
    return metrics
