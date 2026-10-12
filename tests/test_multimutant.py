"""Multi-mutant proposal tests (GAP 4) -- SYNTHETIC only, no network/model.

The capability under test is ADDITIVE-stacking multi-mutant proposal with
epistasis explicitly DE-SCOPED from the predictor:

* enumeration: distinct-position combos, canonical ``':'`` order, count, cap;
* :class:`AdditiveMultiPredictor`: predict(multi) == baseline + sum(components)
  EXACTLY, fail-fast on a missing constituent (no silent 0), and NO epistasis
  term (a constructed interacting pair is predicted as the additive sum);
* the generative oracle: additive at epistasis=0, differs at epistasis>0;
* :func:`propose_multimutants`: distinct, additively ranked, deterministic;
* the ablation DIRECTION: stacking's (scale-free) advantage SHRINKS as epistasis
  grows (the honest bound on the de-scoped additive regime).
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from plm_evolve.benchmark.ablation import run_multimutant_ablation
from plm_evolve.benchmark.multimutant import (
    AdditiveMultiPredictor,
    canonical_multi_token,
    enumerate_multimutants,
    make_generative_landscape,
    propose_multimutants,
)
from plm_evolve.benchmark.oracle import synthetic_landscape
from plm_evolve.variants import parse_variant


# --------------------------------------------------------------------------
# enumerate_multimutants
# --------------------------------------------------------------------------
def test_enumerate_distinct_positions_canonical_order_and_count():
    """Doubles over distinct positions, canonical ':' order, correct count."""
    singles = ["K10R", "M1A", "D5E"]  # deliberately NOT position-sorted.
    multis = enumerate_multimutants(singles, order=2)

    # C(3, 2) = 3 doubles, all at distinct positions.
    assert len(multis) == 3
    assert len(set(multis)) == 3  # distinct ids.

    for token in multis:
        comps = token.split(":")
        positions = [parse_variant(c).position for c in comps]
        # No within-multi position clash.
        assert len(set(positions)) == len(positions)
        # Canonical: components sorted by INTEGER position (10 after 1/5, not "10" < "5").
        assert positions == sorted(positions)

    # The specific canonical ids (position-sorted joins), independent of input order.
    assert set(multis) == {"M1A:D5E", "M1A:K10R", "D5E:K10R"}


def test_enumerate_no_within_multi_position_clash():
    """Two substitutions at the SAME position never form a multi."""
    # A1C and A1D are both at position 1 -> must NOT combine.
    singles = ["A1C", "A1D", "K10R"]
    multis = enumerate_multimutants(singles, order=2)
    # Only A1C:K10R and A1D:K10R survive; A1C:A1D is a same-site clash, dropped.
    assert len(multis) == 2
    for token in multis:
        positions = [parse_variant(c).position for c in token.split(":")]
        assert len(set(positions)) == 2


def test_enumerate_top_k_restricts_to_first_k_singles():
    """top_k uses the first k of the (priority-ordered) singles only."""
    singles = ["M1A", "D5E", "K10R", "S20T"]  # already priority-ordered.
    multis = enumerate_multimutants(singles, order=2, top_k=2)
    # Only M1A and D5E participate -> a single double.
    assert multis == ["M1A:D5E"]


def test_enumerate_order_three_triples():
    """order=3 yields distinct-position triples in canonical order."""
    singles = ["M1A", "D5E", "K10R", "S20T"]
    triples = enumerate_multimutants(singles, order=3)
    assert len(triples) == 4  # C(4, 3).
    for token in triples:
        positions = [parse_variant(c).position for c in token.split(":")]
        assert len(positions) == 3
        assert positions == sorted(positions)


def test_enumerate_order_below_two_raises():
    with pytest.raises(ValueError, match="order must be >= 2"):
        enumerate_multimutants(["M1A", "D5E"], order=1)


def test_enumerate_cap_honored_with_warning_and_deterministic():
    """A binding cap warns (not silent) and is a deterministic rng subset."""
    # 10 singles at distinct positions -> C(10, 2) = 45 doubles.
    singles = [f"A{p}C" for p in range(1, 11)]
    full = enumerate_multimutants(singles, order=2)
    assert len(full) == 45

    with pytest.warns(UserWarning, match="capped"):
        capped_a = enumerate_multimutants(
            singles, order=2, max_candidates=12, rng=np.random.default_rng(0)
        )
    capped_b = enumerate_multimutants(
        singles, order=2, max_candidates=12, rng=np.random.default_rng(0)
    )
    assert len(capped_a) == 12
    assert capped_a == capped_b  # same seed -> identical subset.
    assert set(capped_a).issubset(set(full))


def test_enumerate_cap_without_rng_raises():
    """A binding cap with no rng fails fast rather than truncating silently."""
    singles = [f"A{p}C" for p in range(1, 11)]
    with pytest.raises(ValueError, match="no rng was provided"):
        enumerate_multimutants(singles, order=2, max_candidates=5)


def test_canonical_multi_token_orders_by_integer_position():
    """Position 2 vs 10 sorts numerically, not lexicographically."""
    assert canonical_multi_token(["A10C", "A2C"]) == "A2C:A10C"
    # Same multi from either input order -> one id.
    assert canonical_multi_token(["A2C", "A10C"]) == "A2C:A10C"


# --------------------------------------------------------------------------
# AdditiveMultiPredictor
# --------------------------------------------------------------------------
def test_predict_equals_baseline_plus_sum_exact():
    """predict(multi) == baseline + sum(component effects) EXACTLY."""
    effects = {"M1A": 1.5, "D5E": -0.25, "K10R": 2.0}
    pred = AdditiveMultiPredictor(measured_effects=effects, baseline=0.3)

    assert pred.predict("M1A") == pytest.approx(0.3 + 1.5)
    assert pred.predict("M1A:D5E") == pytest.approx(0.3 + 1.5 - 0.25)
    assert pred.predict("M1A:D5E:K10R") == pytest.approx(0.3 + 1.5 - 0.25 + 2.0)
    # Canonical id and any input order give the same prediction (sum commutes).
    assert pred.predict("K10R:M1A") == pytest.approx(pred.predict("M1A:K10R"))


def test_predict_fail_fast_on_missing_constituent_no_silent_zero():
    """A constituent absent from BOTH measured and model raises (no silent 0)."""
    pred = AdditiveMultiPredictor(measured_effects={"M1A": 1.0})
    with pytest.raises(KeyError, match="No effect available"):
        pred.predict("M1A:K10R")  # K10R unavailable.

    # If it silently used 0 for K10R, this would equal predict("M1A") == 1.0.
    # The raise above proves it does not.


def test_predict_unmeasured_constituent_falls_back_to_model():
    """An unmeasured constituent is sourced from the fallback model."""
    # Callable model.
    pred_call = AdditiveMultiPredictor(
        measured_effects={"M1A": 1.0}, model=lambda t: 3.0 if t == "K10R" else 0.0
    )
    assert pred_call.predict("M1A:K10R") == pytest.approx(1.0 + 3.0)

    # Dict model.
    pred_dict = AdditiveMultiPredictor(
        measured_effects={"M1A": 1.0}, model={"K10R": 3.0}
    )
    assert pred_dict.predict("M1A:K10R") == pytest.approx(1.0 + 3.0)


def test_predictor_has_no_epistasis_term_by_design():
    """A pair whose TRUE value != additive is still predicted as the additive sum.

    The oracle (epistasis>0) gives M1A:K10R a true value that DIFFERS from
    effect(M1A)+effect(K10R). The additive predictor IGNORES that interaction by
    design and predicts exactly the sum -- proving there is no epistasis term.
    """
    landscape, score_token = make_generative_landscape(
        n_positions=10, alphabet_subset="ACDEFG", seed=2, epistasis=3.0
    )
    singles = {
        v: landscape.measure(v)
        for v in landscape.variant_ids()
        if v != "WT" and ":" not in v
    }
    # Pick two singles at distinct positions.
    a, b = next(
        (x, y)
        for x, y in itertools.combinations(list(singles)[:6], 2)
        if parse_variant(x).position != parse_variant(y).position
    )
    multi = canonical_multi_token([a, b])

    true_multi = score_token(multi)
    additive_sum = singles[a] + singles[b]
    # The oracle's true multi is genuinely epistatic (differs from the sum).
    assert true_multi != pytest.approx(additive_sum)

    pred = AdditiveMultiPredictor(measured_effects=singles, baseline=0.0)
    # The predictor returns the additive sum (NOT the true epistatic value).
    assert pred.predict(multi) == pytest.approx(additive_sum)
    assert pred.predict(multi) != pytest.approx(true_multi)


# --------------------------------------------------------------------------
# Generative oracle
# --------------------------------------------------------------------------
def test_generative_oracle_additive_at_epistasis_zero():
    """At epistasis=0, score_token(multi) == sum of constituent singles EXACTLY.

    Also: the single-mutant Landscape is the SAME source of truth as score_token
    (measure(single) == score_token(single)), and WT is exactly 0.
    """
    landscape, score_token = make_generative_landscape(
        n_positions=12, alphabet_subset="ACDEFG", seed=5, epistasis=0.0
    )
    assert score_token("WT") == 0.0
    assert landscape.measure("WT") == 0.0

    singles = [v for v in landscape.variant_ids() if v != "WT" and ":" not in v]
    # Landscape and score_token agree on singles (single source of truth).
    for v in singles[:10]:
        assert landscape.measure(v) == pytest.approx(score_token(v))

    # A few distinct-position multis are EXACTLY the sum of their singles.
    checked = 0
    for a, b in itertools.combinations(singles, 2):
        if parse_variant(a).position == parse_variant(b).position:
            continue
        multi = canonical_multi_token([a, b])
        assert score_token(multi) == pytest.approx(
            score_token(a) + score_token(b), abs=1e-12
        )
        checked += 1
        if checked >= 8:
            break
    assert checked >= 8


def test_generative_oracle_epistasis_makes_multis_differ():
    """At epistasis>0 a multi DIFFERS from the additive sum (term live for multis).

    The single effects are identical across epistasis (the epistasis tensor is
    drawn AFTER the site effects), so ONLY the multi changes -- the epistasis
    term is consulted for multis only, not singles.
    """
    ls0, st0 = make_generative_landscape(
        n_positions=12, alphabet_subset="ACDEFG", seed=5, epistasis=0.0
    )
    ls1, st1 = make_generative_landscape(
        n_positions=12, alphabet_subset="ACDEFG", seed=5, epistasis=2.0
    )
    singles = [v for v in ls0.variant_ids() if v != "WT" and ":" not in v]

    # Singles are unchanged by the epistasis knob (term affects multis only).
    for v in singles[:10]:
        assert st0(v) == pytest.approx(st1(v))

    a, b = next(
        (x, y)
        for x, y in itertools.combinations(singles, 2)
        if parse_variant(x).position != parse_variant(y).position
    )
    multi = canonical_multi_token([a, b])
    # epistasis=0: additive; epistasis>0: differs from the additive sum.
    assert st0(multi) == pytest.approx(st0(a) + st0(b))
    assert st1(multi) != pytest.approx(st1(a) + st1(b))


def test_generative_oracle_matches_synthetic_landscape_at_epistasis_zero():
    """The generative latent REUSES synthetic_landscape's construction (drift guard).

    The latent is DUPLICATED in multimutant.py (not refactored out of
    synthetic_landscape) to keep synthetic_landscape's RNG draw order -- and
    every existing test value -- untouched. This test proves the duplication is
    FUNCTIONALLY the original: at epistasis=0 / noise=0 the generative oracle's
    single-mutant activities are byte-identical to
    :func:`~plm_evolve.benchmark.oracle.synthetic_landscape`. If either latent
    drifts, this fails.
    """
    for seed in (0, 5, 7):
        for alphabet in ("ACDEFG", "ACDE"):
            for n in (8, 12):
                ls_gen, _ = make_generative_landscape(
                    n_positions=n, alphabet_subset=alphabet, seed=seed,
                    epistasis=0.0,
                )
                ls_syn = synthetic_landscape(
                    n_positions=n, alphabet_subset=alphabet, seed=seed,
                    epistasis=0.0, noise=0.0, n_multi_mutants=0,
                )
                singles_gen = {
                    v: ls_gen.measure(v)
                    for v in ls_gen.variant_ids()
                    if v != "WT" and ":" not in v
                }
                singles_syn = {
                    v: ls_syn.measure(v)
                    for v in ls_syn.variant_ids()
                    if v != "WT" and ":" not in v
                }
                assert set(singles_gen) == set(singles_syn)
                assert ls_gen.measure("WT") == ls_syn.measure("WT") == 0.0
                for v in singles_gen:
                    # Byte-identical: same draw order -> same site effects.
                    assert singles_gen[v] == singles_syn[v]


def test_generative_oracle_deterministic():
    """Same seed/params -> identical landscape and score_token outputs."""
    ls_a, st_a = make_generative_landscape(
        n_positions=8, alphabet_subset="ACDE", seed=7, epistasis=1.0
    )
    ls_b, st_b = make_generative_landscape(
        n_positions=8, alphabet_subset="ACDE", seed=7, epistasis=1.0
    )
    np.testing.assert_array_equal(ls_a.true_values(), ls_b.true_values())
    for v in ls_a.variant_ids():
        assert st_a(v) == st_b(v)


# --------------------------------------------------------------------------
# propose_multimutants
# --------------------------------------------------------------------------
def test_propose_returns_distinct_multis_ranked_by_additive_score():
    """The proposals are distinct and ordered by additive (sum) score, desc."""
    effects = {"M1A": 3.0, "D5E": 2.0, "K10R": 1.0, "S20T": 0.5}
    proposals = propose_multimutants(effects, order=2, n_propose=3)

    assert len(proposals) == len(set(proposals))  # distinct.
    # The best double stacks the two best singles.
    assert proposals[0] == "M1A:D5E"  # 3.0 + 2.0 = 5.0 (top additive score).

    pred = AdditiveMultiPredictor(measured_effects=effects)
    scores = [pred.predict(p) for p in proposals]
    assert scores == sorted(scores, reverse=True)  # descending additive score.


def test_propose_is_deterministic():
    """Same inputs -> same proposal (and rng path for a capped enumeration)."""
    effects = {f"A{p}C": float(p) for p in range(1, 11)}  # 10 distinct positions.
    kw = dict(order=2, n_propose=5, max_candidates=20)
    a = propose_multimutants(effects, rng=np.random.default_rng(3), **kw)
    b = propose_multimutants(effects, rng=np.random.default_rng(3), **kw)
    assert a == b


# --------------------------------------------------------------------------
# Ablation DIRECTION (honest interaction, not a magic number)
# --------------------------------------------------------------------------
def test_run_multimutant_ablation_tidy_and_deterministic():
    kw = dict(epistasis_list=[0.0, 4.0], seeds=[0, 1, 2])
    a = run_multimutant_ablation(**kw)
    b = run_multimutant_ablation(**kw)
    expected_cols = [
        "epistasis", "seed", "pool_size", "pool_best", "stack_best_found",
        "random_best_found", "regret_stack", "regret_random", "norm_advantage",
    ]
    assert list(a.columns) == expected_cols
    assert len(a) == 2 * 3  # 2 epistasis x 3 seeds.
    import pandas as pd
    pd.testing.assert_frame_equal(a, b)


def test_multimutant_ablation_direction_advantage_shrinks_with_epistasis():
    """HONEST DIRECTION (interaction), not a magic number.

    Additive stacking's scale-free advantage over random (the fraction of
    random's regret gap it closes) is LARGE when the landscape is additive
    (epistasis=0, where the additive predictor is exactly right) and SHRINKS as
    epistasis grows (the additive predictor degrades because it ignores the
    interaction). Assert:

      * at epistasis=0 stacking REACHES the pool optimum (regret ~ 0), and
      * advantage(epistasis=0) > advantage(epistasis=high).

    This bounds the capability to the additive-dominated regime the user
    de-scoped to; it is NOT rigged to win at high epistasis.
    """
    df = run_multimutant_ablation(
        epistasis_list=[0.0, 4.0], seeds=list(range(8))
    )
    grp = df.groupby("epistasis")

    # At epistasis=0 stacking is exactly right -> reaches the pool optimum.
    add_stack_regret = grp.get_group(0.0)["regret_stack"].mean()
    assert add_stack_regret == pytest.approx(0.0, abs=1e-9)
    add_adv = grp.get_group(0.0)["norm_advantage"].mean()
    high_adv = grp.get_group(4.0)["norm_advantage"].mean()

    # Additive regime: stacking closes the large majority of random's gap. The
    # per-seed norm_advantage is averaged, and a seed where random ALSO happens to
    # land on the pool optimum is a genuine tie (advantage 0 -- no gap to close),
    # which pulls the mean below 1.0 even though stacking is perfect every seed
    # (asserted exactly via regret_stack ~ 0 above). 0.8 is comfortably clear of
    # the high-epistasis value without being a rigged magic number.
    assert add_adv > 0.8
    # Interaction (the load-bearing claim): the advantage SHRINKS as epistasis
    # grows -- a clear margin, not a hairline, so it is robust to seed choice.
    assert high_adv < add_adv - 0.2
    # And at high epistasis the additive predictor has genuinely degraded.
    assert grp.get_group(4.0)["regret_stack"].mean() > add_stack_regret
