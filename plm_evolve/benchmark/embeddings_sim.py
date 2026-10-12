"""Honest, locality-swept synthetic embedding tables for the GAP-2 ablation.

GAP 2 (the defensible novelty): EVOLVEpro and its successor FolDE both
MEAN-POOL the whole-protein embedding. For a single substitution in a length-L
protein, mean-pooling dilutes the changed-position signal by ~1/L. The proposed
alternative carries the MUTATED-POSITION token embedding into the
active-learning scan loop. Whether that helps depends on WHERE the signal lives:
the zero-shot variant-effect literature argues it lives at the mutated-position
marginal, but that is an EMPIRICAL property of real PLMs to be measured
separately -- this module never assumes it.

What this module builds (and what it deliberately does NOT do)
--------------------------------------------------------------
It builds per-variant embedding tables for THREE poolings from ONE shared
latent, parameterised by a locality fraction ``alpha in [0, 1]`` and protein
length ``L``. The point is to expose the MECHANICS of pooling-driven dilution
and its SENSITIVITY to locality -- NOT to prove that mutated-token wins on real
proteins.

Honest construction (analytic gains, not a literal token field)
---------------------------------------------------------------
A literal L x D per-residue token field would (a) be huge for a ~600-residue protein
and (b) cannot reproduce both pooling limits under one plain arithmetic mean
(local-on-mutated + uniform-smear gives ``mean = s/L`` for every alpha). Instead
the two pooled signals are computed directly from pooling-dependent GAINS:

    g_mean(alpha, L) = (1 - alpha) + alpha / L
    g_mut (alpha, L) = alpha + (1 - alpha) / L

These are exact DUALS (swap ``alpha <-> 1 - alpha``) with a symmetric crossover
at ``alpha = 0.5``. They encode the real dilution physics:

    * the LOCAL part (fraction ``alpha`` of the signal) sits on the mutated
      token; mean-pooling over L positions divides it by L -> ``alpha / L`` in
      ``mean`` but ``alpha`` (full) in ``mutated_token``;
    * the SMEAR part (fraction ``1 - alpha``) is spread across all L positions;
      mean-pooling recovers it fully (``1 - alpha`` in ``mean``) while a single
      mutated token sees only its 1/L share (``(1 - alpha) / L``).

Limits (a SWEPT property, never hardcoded):

    * alpha -> 1 (fully local): ``g_mean -> 1/L`` (mean-pool signal vanishes as L
      grows) while ``g_mut -> 1`` (mutated-token holds);
    * alpha -> 0 (fully smeared): ``g_mean -> 1`` (mean keeps the full signal)
      while ``g_mut -> 1/L`` (mutated-token vanishes);
    * intermediate alpha: crossover.

Per-variant row (D dims, one fixed random unit direction ``u`` from the seed):

    s(v)        = true_activity(v) + N(0, signal_noise)     # activity-coupled
    table_p[v]  = g_p(alpha, L) * s(v) * u + distractors[v] (+ obs noise)
    delta_table = mean_table - mean_row(WT)                 # constant offset

The SAME ``s(v)`` and the SAME distractor draw feed all three tables, so the
ONLY thing that differs between poolings is the scalar gain ``g_p`` applied to
the shared signal direction -- the "only pooling differs" isolation the ablation
requires. ``s(WT) = 0`` by construction (the landscape sets WT activity to 0
before noise), so the WT mean row is pure background and ``delta`` removes it as
a constant offset.

RF offset-invariance and the delta CANARY
-----------------------------------------
RandomForest splits are thresholds on individual features. Subtracting a
CONSTANT vector (the WT mean row) from every variant row shifts all candidate
thresholds equally -> identical tree structure -> IDENTICAL predictions and
picks. Therefore ``delta`` is provably equivalent to ``mean`` under the RF
top-layer. ``delta`` is implemented as a CANARY: greedy-on-delta must equal
greedy-on-mean. A large delta-vs-mean gap is a BUG (distractors redrawn, WT row
not constant, etc.), not a result. The mode that actually changes information
content is ``mutated_token``.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from ..variants import parse_variant
from .oracle import Landscape


#: The three poolings this module emits. ``mean`` is the EVOLVEpro/FolDE
#: baseline; ``mutated_token`` is the GAP-2 mode; ``delta`` is the RF-invariance
#: canary (equivalent to ``mean`` under the RF top-layer).
POOLINGS = ("mean", "mutated_token", "delta")


def pooling_gain(pooling: str, locality_alpha: float, length: int) -> float:
    """The scalar signal gain ``g_p(alpha, L)`` for one pooling.

    Args:
        pooling: ``"mean"`` or ``"mutated_token"`` (``"delta"`` reuses the mean
            gain because it is the mean table minus a constant WT offset).
        locality_alpha: locality fraction in ``[0, 1]``. 1 = fully local
            (signal on the mutated token), 0 = fully smeared across positions.
        length: protein length ``L`` (number of residue positions).

    Returns:
        ``g_mean = (1 - alpha) + alpha / L`` or
        ``g_mut  = alpha + (1 - alpha) / L``.

    These are exact duals; the crossover is at ``alpha = 0.5``. This is the
    only place the dilution physics is encoded, and ``alpha`` / ``length`` are
    always passed in (never hardcoded).
    """
    if length < 1:
        raise ValueError(f"length (L) must be >= 1, got {length}.")
    if not (0.0 <= locality_alpha <= 1.0):
        raise ValueError(
            f"locality_alpha must be in [0, 1], got {locality_alpha}."
        )
    a = float(locality_alpha)
    inv_l = 1.0 / float(length)
    if pooling in ("mean", "delta"):
        return (1.0 - a) + a * inv_l
    if pooling == "mutated_token":
        return a + (1.0 - a) * inv_l
    raise ValueError(
        f"Unknown pooling {pooling!r}; expected one of {POOLINGS}."
    )


def _single_mutant_ids(landscape: Landscape) -> list[str]:
    """Single-mutant variant ids only (skip WT and ``:``-joined multi-mutants).

    The GAP-2 ablation is defined over single substitutions; multi-mutant
    oracle entries exist only to exercise the harness and have no mutated
    position to carry, so they are excluded here (fail-fast convention: they
    are simply not embedded, mirroring the core single-substitution scope).
    """
    return [
        v
        for v in landscape.variant_ids()
        if v != "WT" and ":" not in v
    ]


def synthetic_embedding_tables(
    landscape: Landscape,
    *,
    wt_len: Optional[int] = None,
    n_dims: int = 64,
    locality_alpha: float = 0.5,
    signal_noise: float = 0.05,
    distractor_scale: float = 0.5,
    obs_noise: float = 0.0,
    seed: int = 0,
    poolings: Sequence[str] = POOLINGS,
) -> Dict[str, pd.DataFrame]:
    """Build matched ``mean`` / ``mutated_token`` / ``delta`` embedding tables.

    ONE shared latent feeds all three poolings: a single activity-coupled scalar
    signal ``s(v)`` per variant and a single fixed random signal direction
    ``u``. The poolings differ ONLY by the scalar gain ``g_p(alpha, L)`` applied
    to ``s(v) * u`` (see :func:`pooling_gain`). The distractor draw is SHARED
    across poolings so the isolated variable is the pooling gain, not the noise.

    This proves MECHANICS (pooling-driven dilution) and SENSITIVITY (to the
    swept ``locality_alpha``). It does NOT prove mutated-token is better on real
    proteins -- that depends on the empirical alpha of real PLMs, measured
    separately.

    Args:
        landscape: the oracle landscape (its WT activity is 0, so ``s(WT) = 0``).
        wt_len: protein length ``L`` used in the gains. Defaults to the number of
            positions inferred from ``landscape.wt_sequence`` if present, else
            the max 1-based mutated position over the single mutants.
        n_dims: embedding dimensionality ``D`` (table width).
        locality_alpha: swept locality fraction in ``[0, 1]`` (1 = fully local).
        signal_noise: sd of Gaussian noise added to the activity coupling in
            ``s(v) = true_activity(v) + N(0, signal_noise)``.
        distractor_scale: sd of the per-dimension distractor noise shared across
            poolings (the "everything that is not the activity signal" floor).
        obs_noise: optional extra per-table observation noise (sd). Default 0 so
            the delta canary stays exact; if > 0 it is drawn INDEPENDENTLY per
            pooling and will (intentionally) perturb the delta == mean canary.
        seed: RNG seed -- the same seed yields identical tables.
        poolings: which pooling tables to return (subset of :data:`POOLINGS`).

    Returns:
        ``{pooling: DataFrame}`` for each requested pooling. Each DataFrame is
        indexed by single-mutant variant id (WT and multi-mutants excluded;
        matches the existing single-mutant embedding convention) with columns
        ``e0..e{D-1}``.

    Raises:
        ValueError: empty single-mutant set, bad alpha/length, unknown pooling.
    """
    for p in poolings:
        if p not in POOLINGS:
            raise ValueError(
                f"Unknown pooling {p!r}; expected a subset of {POOLINGS}."
            )
    if n_dims < 1:
        raise ValueError(f"n_dims must be >= 1, got {n_dims}.")

    single = _single_mutant_ids(landscape)
    if not single:
        raise ValueError(
            "Landscape has no single-mutant variants to embed; the GAP-2 "
            "ablation needs single substitutions."
        )

    # Infer L (protein length). Prefer the stashed WT sequence; else derive from
    # the maximum 1-based mutated position. Fail-fast on a degenerate L.
    if wt_len is None:
        wt_seq = getattr(landscape, "wt_sequence", None)
        if isinstance(wt_seq, str) and wt_seq:
            length = len(wt_seq)
        else:
            length = max(parse_variant(v).position for v in single)
    else:
        length = int(wt_len)
    if length < 1:
        raise ValueError(f"Inferred protein length L={length} is invalid.")

    rng = np.random.default_rng(seed)

    # One fixed random UNIT signal direction shared by all poolings. Normalising
    # keeps the signal magnitude comparable to the distractor scale.
    u = rng.normal(0.0, 1.0, size=n_dims)
    norm = np.linalg.norm(u)
    if norm == 0.0:  # vanishingly unlikely; fail-fast rather than divide by 0.
        raise ValueError("Signal direction drew the zero vector; change seed.")
    u = u / norm

    # Shared activity-coupled scalar signal s(v) = true_activity + signal noise.
    ids = single
    true_act = landscape.measure_many(ids)
    s = true_act + rng.normal(0.0, signal_noise, size=len(ids))

    # Shared distractor draw (the SAME across poolings -> only the gain differs).
    distractors = rng.normal(0.0, distractor_scale, size=(len(ids), n_dims))

    # s(WT) = 0 by construction (landscape WT activity is 0 before noise), so the
    # WT mean row is pure background: gain * 0 * u + WT distractors = WT
    # distractors. Draw WT background from the SAME generator so it is part of
    # the one shared latent; subtracting it from the mean table yields delta.
    wt_distractor = rng.normal(0.0, distractor_scale, size=n_dims)

    columns = [f"e{j}" for j in range(n_dims)]

    def _table(gain: float, with_obs: bool) -> np.ndarray:
        mat = np.outer(s, u) * gain + distractors
        if with_obs and obs_noise:
            mat = mat + rng.normal(0.0, obs_noise, size=mat.shape)
        return mat

    out: Dict[str, pd.DataFrame] = {}

    # Build the mean table once (delta is derived from it by a constant offset).
    g_mean = pooling_gain("mean", locality_alpha, length)
    mean_mat = _table(g_mean, with_obs="mean" in poolings or "delta" in poolings)

    if "mean" in poolings:
        out["mean"] = pd.DataFrame(mean_mat, index=ids, columns=columns)

    if "mutated_token" in poolings:
        g_mut = pooling_gain("mutated_token", locality_alpha, length)
        mut_mat = _table(g_mut, with_obs=True)
        out["mutated_token"] = pd.DataFrame(mut_mat, index=ids, columns=columns)

    if "delta" in poolings:
        # delta = mean table minus the (constant) WT mean row. WT s=0, so its
        # mean row is gain*0*u + wt_distractor = wt_distractor: a CONSTANT vector
        # subtracted from every variant row. This is a pure constant offset of
        # the mean table -> RF predictions/picks are identical (the canary).
        delta_mat = mean_mat - wt_distractor[None, :]
        out["delta"] = pd.DataFrame(delta_mat, index=ids, columns=columns)

    return out
