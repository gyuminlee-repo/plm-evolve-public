"""Multi-objective HOOK tests (GAP 5) -- SYNTHETIC only, no network/model.

The capability under test is OPT-IN multi-objective selection composed into the
EXISTING single-objective loop, with the loop's MAXIMISATION convention (higher
scalar == better) preserved throughout:

* :func:`scalarize`: ``linear`` and ``chebyshev`` give the DOCUMENTED values on a
  tiny known input (incl. the chebyshev sign -- closer-to-ideal == higher);
  weight/length mismatch fails fast (no silent broadcast).
* :func:`pareto_front`: exactly the non-dominated rows are flagged for
  MAXIMISATION, with ties (duplicate front points) both kept.
* :func:`compose_objectives` / :data:`ScalarizedLandscape`: ``measure(v)`` equals
  the :func:`scalarize` of the components; a mismatched id set fails fast; and
  THE HOOK drives the UNCHANGED :func:`run_simulation` loop to completion.
* the ablation DIRECTION: under anti-correlation the balanced (50/50) arm
  achieves objective B far better than activity-only; under alignment they
  roughly tie on B and activity-only is NOT beaten on objective A; the B gap
  shrinks from anti -> aligned (the interaction).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from plm_evolve.benchmark.ablation import run_multiobjective_ablation
from plm_evolve.benchmark.multiobjective import (
    ScalarizedLandscape,
    compose_objectives,
    pareto_front,
    scalarize,
)
from plm_evolve.benchmark.oracle import Landscape, synthetic_landscape
from plm_evolve.benchmark.simulate import run_simulation
from plm_evolve.benchmark.strategies import GreedyTopLayer


# A 3-variant, 2-objective trade-off block used across several tests.
#   variant 0: (objA=0, objB=2)   variant 1: (1, 1)   variant 2: (2, 0)
# min-max-normalised: A -> [0, 0.5, 1], B -> [1, 0.5, 0].
_TRADEOFF = np.array([[0.0, 2.0], [1.0, 1.0], [2.0, 0.0]])


# --------------------------------------------------------------------------
# scalarize: documented values + fail-fast
# --------------------------------------------------------------------------
def test_scalarize_linear_balanced_documented_values():
    """Linear 50/50 of the trade-off block ties every variant at 0.5."""
    s = scalarize(_TRADEOFF, [0.5, 0.5], method="linear", normalize=True)
    # 0.5 * normA + 0.5 * normB = 0.5 * [0,.5,1] + 0.5 * [1,.5,0] = [.5,.5,.5].
    np.testing.assert_allclose(s, [0.5, 0.5, 0.5])


def test_scalarize_linear_weighted_documented_values():
    """Linear 0.8/0.2 favours the high-A variant by the documented amounts."""
    s = scalarize(_TRADEOFF, [0.8, 0.2], method="linear", normalize=True)
    # weights renormalise to [.8,.2]; .8*[0,.5,1] + .2*[1,.5,0] = [.2,.5,.8].
    np.testing.assert_allclose(s, [0.2, 0.5, 0.8])


def test_scalarize_chebyshev_documented_values_and_sign():
    """Chebyshev returns the NEGATED gap: closest-to-ideal == HIGHEST scalar."""
    s = scalarize(_TRADEOFF, [0.5, 0.5], method="chebyshev", normalize=True)
    # ideal = per-objective max of normalised = [1, 1].
    # gap = 0.5 * max(1 - normA, 1 - normB):
    #   v0: .5*max(1,0)=.5 ; v1: .5*max(.5,.5)=.25 ; v2: .5*max(0,1)=.5
    # returned NEGATED -> [-.5, -.25, -.5].
    np.testing.assert_allclose(s, [-0.5, -0.25, -0.5])
    # SIGN GUARD (the real point): the variant balanced on BOTH objectives (v1)
    # gets the HIGHEST scalar; the corner variants the lowest.
    assert s[1] > s[0] and s[1] > s[2]
    assert int(np.argmax(s)) == 1


def test_scalarize_dict_form_aligns_weights_to_key_order():
    """Dict objectives align weights to key/insertion order."""
    objectives = {"activity": np.array([0.0, 1.0, 2.0]),
                  "stability": np.array([2.0, 1.0, 0.0])}
    s_dict = scalarize(objectives, [0.5, 0.5], method="linear")
    s_mat = scalarize(_TRADEOFF, [0.5, 0.5], method="linear")
    np.testing.assert_allclose(s_dict, s_mat)


def test_scalarize_weight_length_mismatch_fails_fast():
    """A weight per objective is required -- no silent broadcast."""
    with pytest.raises(ValueError, match="weights length"):
        scalarize(_TRADEOFF, [1.0], method="linear")  # 1 weight, 2 objectives.
    with pytest.raises(ValueError, match="weights length"):
        scalarize(_TRADEOFF, [0.5, 0.3, 0.2], method="linear")  # 3 weights.


def test_scalarize_ragged_objective_lengths_fail_fast():
    """Objective columns of different lengths fail fast (no broadcast)."""
    with pytest.raises(ValueError, match="same length"):
        scalarize({"a": [0.0, 1.0, 2.0], "b": [0.0, 1.0]}, [0.5, 0.5])


def test_scalarize_requires_two_objectives():
    """A single objective is not a multi-objective composition."""
    with pytest.raises(ValueError, match=">= 2 objectives"):
        scalarize({"a": [0.0, 1.0]}, [1.0])


def test_scalarize_unknown_method_fails_fast():
    with pytest.raises(ValueError, match="linear.*chebyshev"):
        scalarize(_TRADEOFF, [0.5, 0.5], method="nope")


def test_scalarize_constant_objective_is_finite():
    """A constant objective (zero range) contributes 0, not NaN."""
    M = np.array([[5.0, 0.0], [5.0, 1.0], [5.0, 2.0]])  # objA constant.
    s = scalarize(M, [0.5, 0.5], method="linear", normalize=True)
    assert np.all(np.isfinite(s))
    # objA contributes nothing; ranking is driven by objB alone (ascending).
    assert list(np.argsort(s)) == [0, 1, 2]


# --------------------------------------------------------------------------
# pareto_front: non-dominated set for MAXIMISATION
# --------------------------------------------------------------------------
def test_pareto_front_flags_exactly_nondominated():
    """A constructed set: only the dominated row is dropped."""
    # The trade-off block is an all-non-dominated front; add a dominated point.
    M = np.vstack([_TRADEOFF, [0.0, 0.0]])  # [0,0] is dominated by every other.
    mask = pareto_front(M)
    np.testing.assert_array_equal(mask, [True, True, True, False])


def test_pareto_front_dominated_by_single_better_point():
    """A clear winner dominates everything strictly worse on both axes."""
    M = np.array([[3.0, 3.0],   # dominates the rest.
                  [1.0, 2.0],
                  [2.0, 1.0],
                  [0.0, 0.0]])
    mask = pareto_front(M)
    np.testing.assert_array_equal(mask, [True, False, False, False])


def test_pareto_front_keeps_duplicate_front_points():
    """Identical rows do not dominate each other (needs a strict > somewhere)."""
    M = np.array([[2.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
    mask = pareto_front(M)
    np.testing.assert_array_equal(mask, [True, True, True])


def test_pareto_front_tie_on_one_axis_resolved_by_strict_other():
    """Equal on one objective, strictly worse on the other -> dominated."""
    M = np.array([[1.0, 2.0],   # ties A with row 1, but worse on B -> dominated.
                  [1.0, 3.0],
                  [4.0, 0.0]])
    mask = pareto_front(M)
    np.testing.assert_array_equal(mask, [False, True, True])


# --------------------------------------------------------------------------
# compose_objectives / ScalarizedLandscape: THE HOOK
# --------------------------------------------------------------------------
def _two_objective_landscapes(seed: int = 0):
    """Two landscapes over the SAME variant id set, different activity signals.

    Objective A is a plain synthetic landscape. Objective B reuses A's id set
    (same WT sequence -> same ``<WTaa><pos><newaa>`` tokens) but assigns it an
    INDEPENDENT activity per variant, so the two objectives genuinely differ
    while remaining composable (compose_objectives requires the SAME ids).
    """
    lsA = synthetic_landscape(n_positions=8, alphabet_subset="ACDEFG", seed=seed)
    rng = np.random.default_rng(seed + 100)
    ids = lsA.variant_ids()
    lsB = Landscape.from_mapping(
        {vid: float(rng.normal(0.0, 1.0)) for vid in ids}
    )
    return lsA, lsB


def test_compose_objectives_measure_equals_scalarize_of_components():
    """ScalarizedLandscape.measure(v) == scalarize of the per-objective values."""
    lsA, lsB = _two_objective_landscapes(seed=1)
    composed = compose_objectives([lsA, lsB], [0.5, 0.5], method="linear")

    ids = lsA.variant_ids()
    matrix = np.column_stack([lsA.measure_many(ids), lsB.measure_many(ids)])
    expected = scalarize(matrix, [0.5, 0.5], method="linear", normalize=True)

    # The composed landscape is a plain Landscape over the SAME ids.
    assert composed.variant_ids() == ids
    got = composed.measure_many(ids)
    np.testing.assert_allclose(got, expected)
    # ScalarizedLandscape is the same callable (alias) -> identical result.
    composed2 = ScalarizedLandscape([lsA, lsB], [0.5, 0.5], method="linear")
    np.testing.assert_allclose(composed2.measure_many(ids), expected)


def test_compose_objectives_chebyshev_measure_matches():
    """The chebyshev hook composes correctly too (sign carried through)."""
    lsA, lsB = _two_objective_landscapes(seed=2)
    composed = compose_objectives([lsA, lsB], [0.5, 0.5], method="chebyshev")
    ids = lsA.variant_ids()
    matrix = np.column_stack([lsA.measure_many(ids), lsB.measure_many(ids)])
    expected = scalarize(matrix, [0.5, 0.5], method="chebyshev", normalize=True)
    np.testing.assert_allclose(composed.measure_many(ids), expected)


def test_compose_objectives_mismatched_id_sets_fail_fast():
    """Landscapes over different variant id sets cannot be composed."""
    lsA = synthetic_landscape(n_positions=6, alphabet_subset="ACDEFG", seed=3)
    lsB = synthetic_landscape(n_positions=5, alphabet_subset="ACDEFG", seed=3)
    with pytest.raises(ValueError, match="disagrees on the variant id set"):
        compose_objectives([lsA, lsB], [0.5, 0.5])


def test_compose_objectives_requires_two_landscapes_and_matching_weights():
    lsA, lsB = _two_objective_landscapes(seed=4)
    with pytest.raises(ValueError, match=">= 2 landscapes"):
        compose_objectives([lsA], [1.0])
    with pytest.raises(ValueError, match="weights length"):
        compose_objectives([lsA, lsB], [0.5])  # one weight, two landscapes.


def _learnable_embeddings_for(landscape: Landscape, seed: int = 0,
                              n_dims: int = 6, emb_noise: float = 0.05):
    """Embeddings that carry the (composed) signal -- keyed on the composed ids.

    Mirrors the GAP-2 learnable fixture: dim-0 is the true (composed) value plus
    small noise so the top-layer RF can generalise; the rest are distractors.
    """
    single = [v for v in landscape.variant_ids() if v != "WT" and ":" not in v]
    rng = np.random.default_rng(seed)
    rows = {}
    for v in single:
        true = landscape.measure(v)
        signal = true + rng.normal(0.0, emb_noise)
        distractors = rng.normal(0.0, 1.0, size=n_dims - 1)
        rows[v] = np.concatenate([[signal], distractors])
    return pd.DataFrame.from_dict(
        rows, orient="index", columns=[f"e{j}" for j in range(n_dims)]
    )


def test_hook_drives_unchanged_run_simulation_loop():
    """THE HOOK: a composed Landscape plugs into run_simulation unchanged."""
    lsA, lsB = _two_objective_landscapes(seed=5)
    composed = compose_objectives([lsA, lsB], [0.5, 0.5], method="linear")
    emb = _learnable_embeddings_for(composed, seed=5)

    # Identical call signature as any single-objective run -- nothing special.
    df = run_simulation(
        composed, emb, GreedyTopLayer(),
        n_init=8, n_per_round=6, n_rounds=4, seed=5, top_k=10,
        target=float(np.quantile(composed.true_values(), 0.9)),
    )
    # A full round table came back (round 0 seed + 4 AL rounds).
    assert list(df["round"]) == [0, 1, 2, 3, 4]
    # Cumulative best-found is monotonic non-decreasing (a sane loop completed).
    best = df["max_activity_found"].to_numpy()
    assert np.all(np.diff(best) >= -1e-9)
    # The composed oracle only ever measured variants it contains (no KeyError
    # would mean the hook handed the loop a coherent single-scalar landscape).
    assert int(df["n_measured"].iloc[-1]) == 8 + 6 * 4


# --------------------------------------------------------------------------
# Ablation DIRECTION (honest, correlation-swept)
# --------------------------------------------------------------------------
def test_multiobjective_ablation_direction_and_interaction():
    """Anti-corr: balanced >> activity-only on B. Aligned: tie on B, A unbeaten.

    Assertions are on the seed-MEAN with a tolerance (not per-seed) so the
    direction is robust, and the B gap is asserted to SHRINK from anti -> aligned
    (the interaction). The non-rig check: aligned balanced does NOT beat
    activity-only on objective A.
    """
    seeds = list(range(12))
    df = run_multiobjective_ablation(
        rho_list=[-0.9, 0.9], seeds=seeds, pool_size=200, budget=10
    )
    anti = df[df["rho"] == -0.9]
    aligned = df[df["rho"] == 0.9]

    # --- objective B: balanced - activity_only, per regime (seed means).
    gap_b_anti = float(
        (anti["balanced_objB"] - anti["activity_only_objB"]).mean()
    )
    gap_b_aligned = float(
        (aligned["balanced_objB"] - aligned["activity_only_objB"]).mean()
    )

    # Anti-correlated: balanced achieves objective B MUCH better.
    assert gap_b_anti > 0.15, gap_b_anti
    # Aligned: roughly a tie on B (B comes for free with A).
    assert abs(gap_b_aligned) < 0.08, gap_b_aligned
    # The INTERACTION: the B advantage shrinks from anti -> aligned.
    assert gap_b_anti > gap_b_aligned + 0.1

    # --- objective A non-rig: aligned balanced does NOT beat activity-only on A.
    # Activity-only structurally contains the global A-max -> its normalised A
    # achievement is 1.0 and cannot be exceeded.
    assert np.allclose(aligned["activity_only_objA"], 1.0)
    gap_a_aligned = float(
        (aligned["balanced_objA"] - aligned["activity_only_objA"]).mean()
    )
    assert gap_a_aligned <= 1e-9, gap_a_aligned

    # --- balance metric (min of the two normalised maxima) confirms the trade.
    bal_anti = float(
        (anti["balanced_balance"] - anti["activity_only_balance"]).mean()
    )
    assert bal_anti > 0.15, bal_anti


def test_multiobjective_ablation_columns_and_pareto_fraction_bounds():
    """The frame is well-formed and Pareto fractions are in [0, 1]."""
    seeds = list(range(6))
    df = run_multiobjective_ablation(
        rho_list=[-0.5, 0.0, 0.5], seeds=seeds, pool_size=120, budget=8
    )
    assert len(df) == 3 * len(seeds)
    for col in ("activity_only_pareto_frac", "balanced_pareto_frac"):
        assert df[col].between(0.0, 1.0).all()
    # Anti-correlated: balanced recovers more of the true Pareto front on average
    # (real margin, not a 1e-9 knife-edge).
    anti = df[df["rho"] == -0.5]
    assert (
        anti["balanced_pareto_frac"].mean()
        > anti["activity_only_pareto_frac"].mean() + 0.05
    )


def test_multiobjective_ablation_validates_inputs():
    with pytest.raises(ValueError, match="budget must be >= 1"):
        run_multiobjective_ablation(rho_list=[0.0], seeds=[0], budget=0)
    with pytest.raises(ValueError, match="pool_size must be >= budget"):
        run_multiobjective_ablation(
            rho_list=[0.0], seeds=[0], pool_size=5, budget=10
        )
