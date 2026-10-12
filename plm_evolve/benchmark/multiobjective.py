"""Multi-objective selection as a SECONDARY, opt-in HOOK (GAP 5).

v1 of plm-evolve is SINGLE-objective: the loop measures ONE scalar activity per
variant and ranks on it. Real campaigns sometimes care about a SECOND property
(stability, expression, solubility) alongside activity. This module adds that as
an OPT-IN COMPOSITION, not a novelty and not a fork of the loop:

    compose two or more objectives into ONE scalar oracle, then hand the
    composed :class:`~plm_evolve.benchmark.oracle.Landscape` to the EXISTING
    :func:`~plm_evolve.benchmark.simulate.run_simulation` / strategies. Nothing
    in the loop changes -- it still sees one scalar per variant.

The maximization convention of the loop is preserved everywhere here: a HIGHER
scalar is always BETTER on the combined objectives.

* :func:`scalarize` -- weighted ``linear`` sum or weighted ``chebyshev`` of
  per-objective MIN-MAX-normalised values; both return higher == better.
* :func:`pareto_front` -- the non-dominated set for MAXIMISATION (reporting).
* :func:`compose_objectives` (aliased :data:`ScalarizedLandscape`) -- THE HOOK:
  given 2+ Landscapes over the SAME variant ids, build ONE Landscape whose
  ``measure(v)`` returns the scalarized value.

Normalisation note (and why it is honest). Min-max normalisation in
:func:`compose_objectives` is computed ONCE over the FULL variant set at compose
time. That is legitimate: the oracle is ground truth over the whole landscape
(exactly like the single-objective oracle already is), and the *learner* still
only ever sees the scalar values of the variants it has MEASURED -- it never gets
the per-objective components or the global min/max. The composition just defines
what "true combined activity" means for each variant before any measurement.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Sequence, Union

import numpy as np

from .oracle import Landscape

__all__ = [
    "scalarize",
    "pareto_front",
    "compose_objectives",
    "ScalarizedLandscape",
]

ObjectivesInput = Union[Mapping[str, Sequence[float]], np.ndarray, Sequence[Sequence[float]]]


def _as_matrix(objectives: ObjectivesInput) -> tuple[np.ndarray, List[str]]:
    """Coerce ``objectives`` to an ``(n_variants, n_objectives)`` matrix.

    Accepts either a ``dict{name: array}`` (one column per objective, ALL of the
    SAME length aligned to a shared variant order) or an already-shaped
    ``(n_variants, n_objectives)`` matrix / nested sequence. Returns the matrix
    and the list of objective names (synthesised ``obj0, obj1, ...`` for the
    matrix form). Fail-fast on ragged columns or an empty input -- no silent
    broadcast.
    """
    if isinstance(objectives, Mapping):
        names = list(objectives.keys())
        if len(names) < 2:
            raise ValueError(
                f"scalarize needs >= 2 objectives, got {len(names)}."
            )
        cols = [np.asarray(objectives[n], dtype=float).ravel() for n in names]
        lengths = {c.shape[0] for c in cols}
        if len(lengths) != 1:
            raise ValueError(
                "All objective arrays must have the same length; got lengths "
                f"{ {n: c.shape[0] for n, c in zip(names, cols)} }."
            )
        matrix = np.column_stack(cols)
        return matrix, names

    matrix = np.asarray(objectives, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(
            f"objectives matrix must be 2-D (n_variants x n_objectives), got "
            f"shape {matrix.shape}."
        )
    if matrix.shape[1] < 2:
        raise ValueError(
            f"scalarize needs >= 2 objectives, got {matrix.shape[1]} columns."
        )
    names = [f"obj{j}" for j in range(matrix.shape[1])]
    return matrix, names


def _minmax_normalise(matrix: np.ndarray) -> np.ndarray:
    """Per-column min-max normalisation to ``[0, 1]`` (higher stays better).

    A constant column (zero range) maps to all-zeros -- it carries no
    discriminating information, so it contributes nothing to the scalar rather
    than producing NaN from a divide-by-zero.
    """
    lo = matrix.min(axis=0)
    hi = matrix.max(axis=0)
    rng = hi - lo
    out = np.zeros_like(matrix, dtype=float)
    nonconst = rng > 0
    out[:, nonconst] = (matrix[:, nonconst] - lo[nonconst]) / rng[nonconst]
    return out


def scalarize(
    objectives: ObjectivesInput,
    weights: Sequence[float],
    method: str = "linear",
    normalize: bool = True,
) -> np.ndarray:
    """Compose per-objective values into ONE scalar per variant (higher better).

    Args:
        objectives: either a ``dict{name: array}`` (columns aligned to a shared
            variant order) or an ``(n_variants x n_objectives)`` matrix. For the
            dict form, ``weights`` aligns to the dict's key/insertion order.
        weights: one non-negative weight per objective, in objective order. They
            need not sum to 1 (they are renormalised internally for ``linear``;
            their ratios set the trade-off for ``chebyshev``). Length MUST match
            the number of objectives -- fail-fast, no broadcast.
        method: ``"linear"`` -> weighted sum of min-max-normalised objectives;
            ``"chebyshev"`` -> weighted Chebyshev to the ideal point, returned
            NEGATED so closer-to-ideal == higher (Pareto-aware).
        normalize: min-max-normalise each objective before combining (default
            ``True``). With objectives on different scales this is what makes the
            weighting fair. ``False`` combines the raw values (the ideal point
            for chebyshev is then the per-objective raw max).

    Returns:
        A length-``n_variants`` array; a LARGER value is better on the combined
        objectives under both methods.

    Normalisation: each objective is mapped to ``[0, 1]`` by
    ``(x - min) / (max - min)`` over the supplied variant set; a constant
    objective maps to all-zeros (contributes nothing) instead of dividing by
    zero.
    """
    matrix, names = _as_matrix(objectives)
    n_obj = matrix.shape[1]

    w = np.asarray(weights, dtype=float).ravel()
    if w.shape[0] != n_obj:
        raise ValueError(
            f"weights length {w.shape[0]} != number of objectives {n_obj} "
            f"(objectives: {names}); pass exactly one weight per objective."
        )
    if np.any(w < 0):
        raise ValueError("weights must be non-negative.")
    if not np.any(w > 0):
        raise ValueError("at least one weight must be positive.")

    values = _minmax_normalise(matrix) if normalize else matrix.astype(float)

    if method == "linear":
        # Weighted sum of (normalised) objectives. Renormalise the weights so
        # the scalar is a convex combination -- a pure scale choice that does
        # not change the ranking, only the units.
        w_lin = w / w.sum()
        return values @ w_lin

    if method == "chebyshev":
        # Weighted Chebyshev distance to the ideal point (per-objective max of
        # the supplied values). Natively a quantity to MINIMISE -- the gap to the
        # ideal. The loop MAXIMISES, so return the NEGATED gap: closer-to-ideal
        # (a variant high on both objectives) gets the HIGHEST scalar.
        ideal = values.max(axis=0)
        gaps = w[None, :] * (ideal[None, :] - values)  # >= 0 entrywise.
        chebyshev_gap = gaps.max(axis=1)
        return -chebyshev_gap

    raise ValueError(
        f"method must be 'linear' or 'chebyshev', got {method!r}."
    )


def pareto_front(matrix: ObjectivesInput) -> np.ndarray:
    """Boolean mask of the non-dominated rows for MAXIMISATION.

    A row ``i`` is DOMINATED iff some other row ``j`` is ``>=`` on every
    objective AND ``>`` on at least one. The returned mask is ``True`` for rows
    that are NOT dominated by any other row (the Pareto front). Identical rows
    never dominate each other (domination requires a strict ``>`` somewhere), so
    all duplicates of a Pareto-optimal point are flagged ``True``. ``O(n^2)``.

    Args:
        matrix: ``dict{name: array}`` or ``(n_variants x n_objectives)`` matrix,
            same input convention as :func:`scalarize`.

    Returns:
        A length-``n_variants`` boolean array; ``True`` == on the Pareto front.
    """
    mat, _ = _as_matrix(matrix)
    n = mat.shape[0]
    on_front = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # Does j dominate i?  j >= i everywhere and j > i somewhere.
            if np.all(mat[j] >= mat[i]) and np.any(mat[j] > mat[i]):
                on_front[i] = False
                break
    return on_front


def compose_objectives(
    landscapes: Sequence[Landscape],
    weights: Sequence[float],
    method: str = "linear",
    normalize: bool = True,
) -> Landscape:
    """THE HOOK: fold 2+ Landscapes over the SAME ids into ONE scalar Landscape.

    Every input Landscape must cover the SAME variant id set (fail-fast on any
    disagreement -- a multi-objective campaign measures the SAME variants for
    each property). Each variant's per-objective true values are read out in a
    shared id order, :func:`scalarize`d with ``weights`` / ``method`` /
    ``normalize``, and the result is packed into a plain
    :class:`~plm_evolve.benchmark.oracle.Landscape` via
    :meth:`Landscape.from_mapping`.

    Because the return value IS a ordinary ``Landscape``, every consumer --
    :func:`~plm_evolve.benchmark.simulate.run_simulation`, the strategies,
    ``top_k_ids``, ``true_values`` -- works UNCHANGED. Passing this composed
    landscape is the ONLY thing a multi-objective campaign does differently.

    Args:
        landscapes: 2+ Landscapes over an identical variant id set.
        weights: one weight per landscape, in the given order (see
            :func:`scalarize`).
        method: ``"linear"`` or ``"chebyshev"`` (see :func:`scalarize`).
        normalize: min-max-normalise each objective before combining.

    Returns:
        A :class:`Landscape` whose ``measure(v)`` is the scalarized combined
        objective for ``v`` (higher == better on all objectives).
    """
    landscapes = list(landscapes)
    if len(landscapes) < 2:
        raise ValueError(
            f"compose_objectives needs >= 2 landscapes, got {len(landscapes)}."
        )
    if len(weights) != len(landscapes):
        raise ValueError(
            f"weights length {len(weights)} != number of landscapes "
            f"{len(landscapes)}; pass exactly one weight per landscape."
        )

    ids = landscapes[0].variant_ids()  # sorted, stable order from Landscape.
    id_set = set(ids)
    for k, ls in enumerate(landscapes[1:], start=1):
        other = set(ls.variant_ids())
        if other != id_set:
            missing = id_set - other
            extra = other - id_set
            raise ValueError(
                f"Landscape {k} disagrees on the variant id set with landscape "
                f"0 (missing {sorted(missing)[:3]}..., extra "
                f"{sorted(extra)[:3]}...); compose_objectives requires the SAME "
                f"variants in every objective."
            )

    # Read each objective's true values in the SHARED id order.
    matrix = np.column_stack(
        [ls.measure_many(ids) for ls in landscapes]
    )
    scalar = scalarize(matrix, weights, method=method, normalize=normalize)

    mapping: Dict[str, float] = {vid: float(s) for vid, s in zip(ids, scalar)}
    return Landscape.from_mapping(mapping)


# Name alias requested by the brief: the hook can be referred to as
# ``ScalarizedLandscape``. It is the SAME callable that builds a plain Landscape
# (no subclass: a subclass would have to re-preserve every Landscape invariant
# and method by hand for zero gain, whereas a composed Landscape already plugs
# into every consumer unchanged).
ScalarizedLandscape = compose_objectives
