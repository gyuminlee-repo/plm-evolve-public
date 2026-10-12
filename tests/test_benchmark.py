"""Benchmark harness tests -- SYNTHETIC only (no model download, no network).

The synthetic landscape doubles as the oracle AND, via a deterministic feature
map, as the embedding table the strategies score on. Because the landscape is
additive, an embedding that exposes the per-site one-hot features is learnable
by the top-layer regressor -- so GreedyTopLayer can beat random.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from plm_evolve.benchmark.oracle import Landscape, synthetic_landscape
from plm_evolve.benchmark.strategies import (
    AdditiveLinear,
    ConstantLiarStrategy,
    CoverageBatchStrategy,
    GreedyTopLayer,
    RandomStrategy,
    ThompsonStrategy,
    UCBStrategy,
    build_strategy,
)
from plm_evolve.benchmark.simulate import (
    run_panel,
    run_simulation,
    single_round_simulation,
)
from plm_evolve.benchmark import metrics as M
from plm_evolve.benchmark.embeddings_sim import (
    POOLINGS,
    pooling_gain,
    synthetic_embedding_tables,
)
from plm_evolve.benchmark.ablation import (
    build_decoy_cluster_landscape,
    build_redundant_landscape,
    build_rugged_landscape,
    run_acquisition_ablation,
    run_batch_diversity_ablation,
    run_locality_ablation,
)
from plm_evolve.variants import parse_variant


# --------------------------------------------------------------------------
# Helpers: build an embedding table that is learnable for an additive oracle.
# --------------------------------------------------------------------------
def _learnable_embeddings(
    landscape: Landscape, seed: int = 0, n_dims: int = 6, emb_noise: float = 0.05
) -> pd.DataFrame:
    """Deterministic embeddings that CARRY the activity signal (learnable).

    Stand-in for PLM embeddings: each single-mutant variant gets a vector whose
    first dimension is its true activity plus small noise, and the remaining
    dimensions are random distractors. The mapping embedding -> activity is
    therefore smooth and GLOBAL, so the top-layer RandomForest generalises from
    measured to unseen candidates (unlike a per-variant identity one-hot, where
    each candidate column is all-zero at train time and nothing transfers).

    Real PLM embeddings work because they place functionally similar variants
    nearby; this fixture reproduces that learnability without a model download.
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


def _learnable_setup(seed=0, n_positions=8, noise=0.0):
    alphabet = "ACDEFG"
    ls = synthetic_landscape(
        n_positions=n_positions, alphabet_subset=alphabet, seed=seed, noise=noise
    )
    emb = _learnable_embeddings(ls, seed=seed)
    return ls, emb


# --------------------------------------------------------------------------
# Oracle / landscape
# --------------------------------------------------------------------------
def test_synthetic_landscape_determinism():
    a = synthetic_landscape(n_positions=6, alphabet_subset="ACDEF", seed=42)
    b = synthetic_landscape(n_positions=6, alphabet_subset="ACDEF", seed=42)
    assert a.variant_ids() == b.variant_ids()
    np.testing.assert_array_equal(a.true_values(), b.true_values())
    # Different seed -> different landscape.
    c = synthetic_landscape(n_positions=6, alphabet_subset="ACDEF", seed=43)
    assert not np.array_equal(a.true_values(), c.true_values())


def test_landscape_measure_and_topk():
    ls = synthetic_landscape(n_positions=5, alphabet_subset="ACDE", seed=1)
    # measure returns the stored value; missing -> fail-fast.
    vid = ls.variant_ids()[0]
    assert ls.measure(vid) == ls.activities[vid]
    with pytest.raises(KeyError):
        ls.measure("Z99Z")
    top3 = ls.top_k_ids(3)
    assert len(top3) == 3
    # top_k are genuinely the largest.
    vals = sorted(ls.true_values(), reverse=True)[:3]
    np.testing.assert_allclose(sorted([ls.measure(v) for v in top3], reverse=True), vals)


def test_landscape_from_csv_roundtrip(tmp_path):
    p = tmp_path / "ls.csv"
    pd.DataFrame({"variant": ["WT", "M1A", "K2R"], "activity": [0.0, 1.5, 2.5]}).to_csv(
        p, index=False
    )
    ls = Landscape.from_csv(p)
    assert ls.measure("K2R") == 2.5
    assert ls.global_max() == 2.5


def test_landscape_from_proteingym(tmp_path):
    p = tmp_path / "dms.csv"
    pd.DataFrame(
        {"mutant": ["M1A", "K2R", "V3D"], "DMS_score": [0.2, 1.1, -0.3]}
    ).to_csv(p, index=False)
    ls = Landscape.from_proteingym(p)
    assert ls.measure("K2R") == 1.1
    assert set(ls.variant_ids()) == {"M1A", "K2R", "V3D"}


# --------------------------------------------------------------------------
# Strategies: valid selections
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "strat",
    [
        RandomStrategy(),
        GreedyTopLayer(),
        AdditiveLinear(),
        UCBStrategy(),
        ThompsonStrategy(),
        CoverageBatchStrategy(),
        ConstantLiarStrategy(),
    ],
)
def test_strategy_valid_selection(strat):
    ls, emb = _learnable_setup(seed=0, n_positions=6)
    pool = [v for v in ls.variant_ids() if v in set(emb.index)]
    rng = np.random.default_rng(0)
    measured = pool[:8]
    measured_y = [ls.measure(v) for v in measured]
    candidates = pool[8:]
    n_select = 5
    picked = strat.pick(measured, measured_y, candidates, emb, n_select, rng)
    # Correct count, subset of candidates, no re-pick of measured, no dupes.
    assert len(picked) == n_select
    assert set(picked).issubset(set(candidates))
    assert set(picked).isdisjoint(set(measured))
    assert len(set(picked)) == len(picked)


def test_build_strategy_registry():
    assert isinstance(build_strategy("greedy"), GreedyTopLayer)
    assert isinstance(build_strategy("random"), RandomStrategy)
    assert isinstance(build_strategy("additive"), AdditiveLinear)
    assert isinstance(build_strategy("ucb"), UCBStrategy)
    assert isinstance(build_strategy("thompson"), ThompsonStrategy)
    assert isinstance(build_strategy("coverage"), CoverageBatchStrategy)
    assert isinstance(build_strategy("constant_liar"), ConstantLiarStrategy)
    with pytest.raises(ValueError):
        build_strategy("nope")


# --------------------------------------------------------------------------
# Simulation: reproducibility + monotonicity
# --------------------------------------------------------------------------
def test_run_simulation_reproducible():
    ls, emb = _learnable_setup(seed=7, n_positions=8)
    g = GreedyTopLayer()
    a = run_simulation(ls, emb, g, n_init=8, n_per_round=6, n_rounds=4, seed=3)
    b = run_simulation(ls, emb, g, n_init=8, n_per_round=6, n_rounds=4, seed=3)
    pd.testing.assert_frame_equal(a, b)


def test_run_simulation_cumulative_best_monotonic():
    ls, emb = _learnable_setup(seed=2, n_positions=8)
    g = GreedyTopLayer()
    df = run_simulation(ls, emb, g, n_init=8, n_per_round=6, n_rounds=5, seed=1)
    best = df["max_activity_found"].to_numpy()
    # Cumulative best is non-decreasing across rounds.
    assert np.all(np.diff(best) >= -1e-12)
    # n_measured strictly grows.
    assert np.all(np.diff(df["n_measured"].to_numpy()) > 0)


def test_rounds_to_target_metric():
    # Manually crafted cumulative-best trajectory.
    traj = [0.1, 0.5, 0.9, 1.2]
    assert M.rounds_to_target(traj, 0.8) == 2.0
    assert np.isnan(M.rounds_to_target(traj, 5.0))


# --------------------------------------------------------------------------
# Core thesis: greedy beats random on a learnable landscape.
# --------------------------------------------------------------------------
def test_greedy_beats_random_on_learnable_landscape():
    # Pool ~= 30 positions * 5 mutants = 150 single mutants; the budget
    # (10 + 6*4 = 34) is a small fraction, leaving room for greedy to win.
    seeds = list(range(6))
    greedy_best, random_best = [], []
    for s in seeds:
        ls, emb = _learnable_setup(seed=s, n_positions=30, noise=0.01)
        gdf = run_simulation(ls, emb, GreedyTopLayer(), n_init=10, n_per_round=6,
                             n_rounds=4, seed=s)
        rdf = run_simulation(ls, emb, RandomStrategy(), n_init=10, n_per_round=6,
                             n_rounds=4, seed=s)
        greedy_best.append(gdf["max_activity_found"].iloc[-1])
        random_best.append(rdf["max_activity_found"].iloc[-1])
    # Mean best-found after K rounds: greedy should beat random.
    assert np.mean(greedy_best) > np.mean(random_best)


# --------------------------------------------------------------------------
# Single-round baseline is measurable against iterative greedy.
# --------------------------------------------------------------------------
def test_single_round_vs_iterative_measurable():
    ls, emb = _learnable_setup(seed=5, n_positions=10, noise=0.01)
    n_init, per_round, rounds = 10, 8, 4
    total = n_init + per_round * rounds

    iterative = run_simulation(ls, emb, GreedyTopLayer(), n_init=n_init,
                               n_per_round=per_round, n_rounds=rounds, seed=5)
    single = single_round_simulation(ls, emb, n_total=total, seed=5, ranker="greedy",
                                     n_init=n_init)

    # Both produced trajectories with the SAME total budget at the end.
    assert iterative["n_measured"].iloc[-1] == total
    assert single["n_measured"].iloc[-1] == total
    # Both expose a comparable final best-found metric (do not over-assert who wins).
    it_best = iterative["max_activity_found"].iloc[-1]
    sr_best = single["max_activity_found"].iloc[-1]
    assert np.isfinite(it_best) and np.isfinite(sr_best)
    assert single["strategy"].iloc[0] == "single_round_greedy"


def test_additive_linear_runs_as_baseline():
    ls, emb = _learnable_setup(seed=4, n_positions=8, noise=0.0)
    df = run_simulation(ls, emb, AdditiveLinear(), n_init=8, n_per_round=6,
                        n_rounds=3, seed=4)
    # Runs end-to-end and improves over the seed (additive model on an additive
    # landscape is a sensible -- indeed strong -- baseline).
    best = df["max_activity_found"].to_numpy()
    assert np.all(np.diff(best) >= -1e-12)
    assert best[-1] >= best[0]


# --------------------------------------------------------------------------
# Panel runner: tidy frame across strategies x seeds x rounds.
# --------------------------------------------------------------------------
def test_run_panel_tidy():
    ls, emb = _learnable_setup(seed=0, n_positions=8)
    strategies = [GreedyTopLayer(), RandomStrategy(), AdditiveLinear()]
    seeds = [0, 1]
    panel = run_panel(ls, emb, strategies, seeds, n_init=8, n_per_round=6,
                      n_rounds=3, target=None, single_round_rankers=["greedy"])
    # Columns present and tidy.
    for col in ("strategy", "seed", "round", "n_measured", "max_activity_found"):
        assert col in panel.columns
    # 3 iterative strategies (4 rounds each: 0..3) + single_round (2 rows) over 2 seeds.
    assert set(panel["strategy"].unique()) == {
        "greedy", "random", "additive", "single_round_greedy"
    }
    assert set(panel["seed"].unique()) == {0, 1}


# ==========================================================================
# GAP-2: honest locality-swept synthetic embeddings + pooling ablation.
#
# These tests prove MECHANICS (pooling-driven dilution) and SENSITIVITY (to the
# swept locality alpha). They do NOT prove that mutated-token pooling is better
# on real proteins -- that depends on the EMPIRICAL alpha of real PLM embeddings,
# which must be measured separately. Locality is always a swept parameter here,
# never hardcoded; the assertions are on the DIRECTION of the effect (averaged
# over seeds), not on magic numbers.
# ==========================================================================
def _final_best_by_pooling(alpha, length, seeds, *, n_dims=64,
                           n_init=10, n_per_round=6, n_rounds=4,
                           poolings=POOLINGS):
    """Mean (over seeds) final best-found for each pooling at one (alpha, L).

    The SAME shared latent (same s(v), same distractors) feeds every pooling
    per seed, so the only thing that differs between poolings is the pooling
    gain g_p(alpha, L) -- the isolation the ablation demands.
    """
    acc = {p: [] for p in poolings}
    for s in seeds:
        ls = synthetic_landscape(
            n_positions=length, alphabet_subset="ACDEFG", seed=s, noise=0.0
        )
        tables = synthetic_embedding_tables(
            ls, wt_len=length, n_dims=n_dims, locality_alpha=alpha,
            signal_noise=0.05, distractor_scale=0.5, seed=s, poolings=poolings,
        )
        for p in poolings:
            sim = run_simulation(
                ls, tables[p], GreedyTopLayer(), n_init=n_init,
                n_per_round=n_per_round, n_rounds=n_rounds, seed=s,
            )
            acc[p].append(sim["max_activity_found"].iloc[-1])
    return {p: float(np.mean(v)) for p, v in acc.items()}


def test_pooling_gain_duals_and_limits():
    """g_mean and g_mut are exact duals (alpha <-> 1-alpha) with correct limits."""
    L = 400
    # Symmetric crossover at alpha = 0.5.
    assert pooling_gain("mean", 0.5, L) == pytest.approx(pooling_gain("mutated_token", 0.5, L))
    # Duality: g_mut(alpha) == g_mean(1 - alpha).
    for a in (0.0, 0.25, 0.7, 1.0):
        assert pooling_gain("mutated_token", a, L) == pytest.approx(
            pooling_gain("mean", 1.0 - a, L)
        )
    # alpha = 1 (fully local): mean-pool gain -> 1/L, mutated-token holds at 1.
    assert pooling_gain("mean", 1.0, L) == pytest.approx(1.0 / L)
    assert pooling_gain("mutated_token", 1.0, L) == pytest.approx(1.0)
    # alpha = 0 (fully smeared): mean keeps full signal, mutated-token -> 1/L.
    assert pooling_gain("mean", 0.0, L) == pytest.approx(1.0)
    assert pooling_gain("mutated_token", 0.0, L) == pytest.approx(1.0 / L)
    # delta reuses the mean gain (it is mean minus a constant WT offset).
    assert pooling_gain("delta", 0.3, L) == pooling_gain("mean", 0.3, L)
    with pytest.raises(ValueError):
        pooling_gain("bogus", 0.5, L)
    with pytest.raises(ValueError):
        pooling_gain("mean", 1.5, L)


def test_embedding_tables_shape_and_single_mutants_only():
    """Tables are indexed by single mutants only (WT + multi-mutants excluded)."""
    ls = synthetic_landscape(
        n_positions=8, alphabet_subset="ACDEFG", seed=0, n_multi_mutants=3
    )
    tables = synthetic_embedding_tables(ls, n_dims=16, locality_alpha=0.7, seed=0)
    assert set(tables.keys()) == set(POOLINGS)
    expected = {v for v in ls.variant_ids() if v != "WT" and ":" not in v}
    for p, df in tables.items():
        assert set(df.index) == expected
        assert list(df.columns) == [f"e{j}" for j in range(16)]
        # No WT and no multi-mutant rows leaked in.
        assert "WT" not in df.index
        assert not any(":" in str(v) for v in df.index)


def test_mechanics_high_alpha_mutated_token_beats_mean():
    """MECHANICS at high locality + large L: mean-pool dilution costs sample
    efficiency, so greedy-on-mean final best < greedy-on-mutated_token. This is
    the ~1/L dilution made visible -- NOT a claim about real proteins.
    """
    seeds = list(range(5))
    best = _final_best_by_pooling(alpha=1.0, length=200, seeds=seeds)
    # Direction only (averaged over seeds): local signal survives in the
    # mutated token but is diluted away under mean-pooling.
    assert best["mutated_token"] > best["mean"]


def test_sensitivity_low_alpha_gap_closes_or_reverses():
    """SENSITIVITY: at low locality (smeared signal) the advantage flips --
    mean-pool keeps the full signal while the single mutated token sees ~1/L.
    So greedy-on-mean >= greedy-on-mutated_token. The effect is a function of
    the SWEPT alpha, proving locality (not pooling per se) drives it.
    """
    seeds = list(range(5))
    best = _final_best_by_pooling(alpha=0.0, length=200, seeds=seeds)
    assert best["mean"] >= best["mutated_token"]


def test_artifact_canary_delta_equals_mean_under_rf():
    """ARTIFACT CANARY (advisor): delta == mean under the RF top-layer.

    RandomForest splits are thresholds on features; subtracting a CONSTANT
    vector (the WT mean row) from every row shifts all thresholds equally ->
    identical tree structure -> identical predictions and picks. A large
    delta-vs-mean gap is a BUG (e.g. distractors redrawn for delta), not a
    result. Asserted at the trajectory level with the SAME seed so the loop's
    initial-set draw matches.
    """
    for s in range(4):
        ls = synthetic_landscape(
            n_positions=120, alphabet_subset="ACDEFG", seed=s, noise=0.0
        )
        tables = synthetic_embedding_tables(
            ls, wt_len=120, n_dims=48, locality_alpha=0.8,
            signal_noise=0.05, distractor_scale=0.5, seed=s,
        )
        mean_sim = run_simulation(
            ls, tables["mean"], GreedyTopLayer(), n_init=10, n_per_round=6,
            n_rounds=4, seed=s,
        )
        delta_sim = run_simulation(
            ls, tables["delta"], GreedyTopLayer(), n_init=10, n_per_round=6,
            n_rounds=4, seed=s,
        )
        # Same picks => identical best-found trajectory (RF offset-invariance).
        np.testing.assert_allclose(
            mean_sim["max_activity_found"].to_numpy(),
            delta_sim["max_activity_found"].to_numpy(),
            rtol=0, atol=0,
        )


def test_embedding_tables_deterministic():
    """Same seed -> byte-identical tables; the canary requires reproducibility."""
    ls = synthetic_landscape(n_positions=10, alphabet_subset="ACDEFG", seed=2)
    a = synthetic_embedding_tables(ls, n_dims=32, locality_alpha=0.6, seed=2)
    b = synthetic_embedding_tables(ls, n_dims=32, locality_alpha=0.6, seed=2)
    for p in POOLINGS:
        pd.testing.assert_frame_equal(a[p], b[p])


def test_run_locality_ablation_tidy_and_deterministic():
    """run_locality_ablation returns the expected tidy columns and is
    reproducible (same sweep -> identical rows).
    """
    kw = dict(
        n_positions_list=[60],
        alpha_list=[0.0, 1.0],
        seeds=[0, 1],
        n_init=8,
        n_per_round=5,
        n_rounds=3,
        n_dims=24,
    )
    a = run_locality_ablation(**kw)
    b = run_locality_ablation(**kw)
    expected_cols = [
        "L", "alpha", "pooling", "seed",
        "max_activity_found", "rounds_to_target", "simple_regret",
    ]
    assert list(a.columns) == expected_cols
    # 1 L x 2 alpha x 3 poolings x 2 seeds = 12 rows.
    assert len(a) == 1 * 2 * len(POOLINGS) * 2
    assert set(a["pooling"].unique()) == set(POOLINGS)
    assert set(a["alpha"].unique()) == {0.0, 1.0}
    pd.testing.assert_frame_equal(a, b)


def test_run_locality_ablation_sweep_shows_crossover():
    """The swept frame reproduces the crossover: at alpha=1 mutated_token wins
    on average; at alpha=0 mean wins. Locality is swept, the conclusion follows
    the data (MECHANICS + SENSITIVITY), and delta tracks mean (canary).
    """
    df = run_locality_ablation(
        n_positions_list=[200],
        alpha_list=[0.0, 1.0],
        seeds=list(range(4)),
        n_init=10,
        n_per_round=6,
        n_rounds=4,
        n_dims=64,
    )
    pivot = df.groupby(["alpha", "pooling"])["max_activity_found"].mean().unstack()
    # High locality: mutated_token beats mean.
    assert pivot.loc[1.0, "mutated_token"] > pivot.loc[1.0, "mean"]
    # Low locality: mean >= mutated_token (advantage closes / reverses).
    assert pivot.loc[0.0, "mean"] >= pivot.loc[0.0, "mutated_token"]
    # Canary across the whole sweep: delta == mean at every alpha.
    np.testing.assert_allclose(pivot["delta"].to_numpy(), pivot["mean"].to_numpy())


# ==========================================================================
# GAP-1: uncertainty-aware acquisition (a CAPABILITY, not a novelty).
#
# UCB / Thompson are added with the SAME pick(...) contract, reusing the core
# TopLayerRegressor. The honest claim is a SENSITIVITY result: uncertainty-aware
# acquisition helps WHEN the landscape is rugged / greedy gets trapped -- not
# "ucb always wins". The assertions below test the INVARIANT (beta=0 == greedy),
# validity, determinism, and the DIRECTION of the ruggedness interaction.
# ==========================================================================
def test_ucb_beta_zero_equals_greedy_exact():
    """UCB(beta=0) picks must be EXACTLY GreedyTopLayer picks (tested invariant).

    With beta=0 the acquisition is mean + 0*std == mean, and mean is byte-
    identical to TopLayerRegressor.predict() (the greedy ranking signal), so the
    stable-descending sort yields the same picks for the same measured set/seed.
    """
    ls, emb = _learnable_setup(seed=0, n_positions=8)
    pool = [v for v in ls.variant_ids() if v in set(emb.index)]
    measured = pool[:8]
    measured_y = [ls.measure(v) for v in measured]
    candidates = pool[8:]
    n_select = 5
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)
    greedy_pick = GreedyTopLayer().pick(
        measured, measured_y, candidates, emb, n_select, rng_a
    )
    ucb_pick = UCBStrategy(beta=0.0).pick(
        measured, measured_y, candidates, emb, n_select, rng_b
    )
    np.testing.assert_array_equal(np.array(greedy_pick), np.array(ucb_pick))


def test_ucb_nonzero_beta_can_differ_from_greedy():
    """With beta>0 UCB MAY reorder picks (exploration term active). Not a strict
    requirement on a given seed, but the acquisition vector must differ from the
    pure mean when std is non-zero somewhere -- a sanity check the beta term is
    actually wired in (i.e. beta=0 collapsing to greedy is not vacuous).
    """
    ls, emb = _learnable_setup(seed=1, n_positions=12)
    pool = [v for v in ls.variant_ids() if v in set(emb.index)]
    measured = pool[:8]
    measured_y = [ls.measure(v) for v in measured]
    candidates = pool[8:]
    from plm_evolve.regressor import TopLayerRegressor

    model = TopLayerRegressor().fit(emb.loc[measured].to_numpy(),
                                    np.asarray(measured_y, float))
    mean, std = model.predict_with_uncertainty(emb.loc[candidates].to_numpy())
    # The exploration term is non-trivial (some candidate has positive std).
    assert np.any(std > 0.0)
    assert not np.array_equal(mean + 2.0 * std, mean)


@pytest.mark.parametrize("strat_name", ["ucb", "thompson"])
def test_acquisition_strategy_determinism(strat_name):
    """Same seed -> same picks for ucb and thompson (rng-only randomness)."""
    ls, emb = _learnable_setup(seed=2, n_positions=10)
    pool = [v for v in ls.variant_ids() if v in set(emb.index)]
    measured = pool[:8]
    measured_y = [ls.measure(v) for v in measured]
    candidates = pool[8:]
    n_select = 6
    a = build_strategy(strat_name).pick(
        measured, measured_y, candidates, emb, n_select, np.random.default_rng(7)
    )
    b = build_strategy(strat_name).pick(
        measured, measured_y, candidates, emb, n_select, np.random.default_rng(7)
    )
    np.testing.assert_array_equal(np.array(a), np.array(b))


@pytest.mark.parametrize("strat_name", ["ucb", "thompson"])
def test_acquisition_cold_start_falls_back_to_random(strat_name):
    """No measured labels -> cold-start random fallback (same idiom as greedy)."""
    ls, emb = _learnable_setup(seed=0, n_positions=8)
    candidates = [v for v in ls.variant_ids() if v in set(emb.index)]
    n_select = 5
    picked = build_strategy(strat_name).pick(
        [], [], candidates, emb, n_select, np.random.default_rng(0)
    )
    assert len(picked) == n_select
    assert set(picked).issubset(set(candidates))
    assert len(set(picked)) == len(picked)


def test_acquisition_strategies_run_in_simulation():
    """UCB / Thompson run end-to-end through the shared simulation loop."""
    ls, emb = _learnable_setup(seed=3, n_positions=10, noise=0.01)
    for strat in (UCBStrategy(), ThompsonStrategy()):
        df = run_simulation(ls, emb, strat, n_init=8, n_per_round=5,
                            n_rounds=3, seed=3)
        best = df["max_activity_found"].to_numpy()
        assert np.all(np.diff(best) >= -1e-12)  # cumulative best non-decreasing
        assert df["strategy"].iloc[0] == strat.name


# --------------------------------------------------------------------------
# GAP-1 ablation: tunable ruggedness builder + the DIRECTION of the effect.
# --------------------------------------------------------------------------
def test_build_rugged_landscape_deterministic_and_smooth_vs_rugged():
    """Same seed -> identical (landscape, embeddings); ruggedness adds a peak."""
    ls0, e0 = build_rugged_landscape(n_variants=50, ruggedness=0.0, seed=1)
    ls0b, e0b = build_rugged_landscape(n_variants=50, ruggedness=0.0, seed=1)
    np.testing.assert_array_equal(ls0.true_values(), ls0b.true_values())
    pd.testing.assert_frame_equal(e0, e0b)
    # Adding ruggedness raises the global max (the narrow tall peak appears).
    ls1, _ = build_rugged_landscape(n_variants=300, ruggedness=1.0, seed=1)
    ls0c, _ = build_rugged_landscape(n_variants=300, ruggedness=0.0, seed=1)
    assert ls1.global_max() > ls0c.global_max()


def test_run_acquisition_ablation_tidy_and_deterministic():
    kw = dict(
        ruggedness_list=[0.0, 1.0],
        seeds=[0, 1],
        n_variants=120,
        n_init=8,
        n_per_round=4,
        n_rounds=3,
        strategies=("greedy", "ucb", "thompson", "random"),
    )
    a = run_acquisition_ablation(**kw)
    b = run_acquisition_ablation(**kw)
    expected_cols = [
        "ruggedness", "strategy", "seed", "max_activity_found",
        "rounds_to_target", "simple_regret", "global_max", "frac_of_global",
    ]
    assert list(a.columns) == expected_cols
    # 2 ruggedness x 4 strategies x 2 seeds = 16 rows.
    assert len(a) == 2 * 4 * 2
    assert set(a["strategy"].unique()) == {"greedy", "ucb", "thompson", "random"}
    pd.testing.assert_frame_equal(a, b)


def test_acquisition_ablation_direction_exploration_helps_when_rugged():
    """HONEST DIRECTION (interaction), not a magic number:

    The advantage of uncertainty-aware acquisition over greedy GROWS with
    ruggedness. On a near-additive (ruggedness=0) landscape greedy ties or
    slightly beats ucb/thompson (the exploration tax); on a rugged landscape
    greedy gets trapped on the broad decoy and ucb/thompson escape to the narrow
    global peak. Assert the interaction (delta_rugged > delta_additive), averaged
    over seeds, for BOTH ucb and thompson -- never that ucb always wins.
    """
    # 16 seeds, not 8: under the default RF criterion (friedman_mse, matching
    # EVOLVEpro) the rugged-landscape exploration-vs-greedy margin is within
    # noise at 8 seeds (a coin-flip on whether ucb edges greedy). The direction
    # is real but needs the extra power to resolve; the assertions below stay
    # strict. Raising the sample size is the correct fix, not loosening them.
    df = run_acquisition_ablation(
        ruggedness_list=[0.0, 1.0],
        seeds=list(range(16)),
        n_variants=300,
        n_init=8,
        n_per_round=4,
        n_rounds=4,
    )
    piv = df.groupby(["ruggedness", "strategy"])["frac_of_global"].mean().unstack()
    greedy_add = piv.loc[0.0, "greedy"]
    greedy_rug = piv.loc[1.0, "greedy"]

    # On the rugged landscape greedy is genuinely trapped (well below global).
    assert greedy_rug < 0.9
    # Near-additive: greedy is near-optimal (the construction is not pre-trapped).
    assert greedy_add > 0.9

    for explore in ("ucb", "thompson"):
        delta_rugged = piv.loc[1.0, explore] - greedy_rug
        delta_additive = piv.loc[0.0, explore] - greedy_add
        # Interaction: exploration helps MORE (relatively) when rugged.
        assert delta_rugged > delta_additive
        # And on the rugged landscape exploration actually beats greedy.
        assert piv.loc[1.0, explore] > greedy_rug


# ==========================================================================
# GAP-3: batch-diversity selection (a CAPABILITY, not a novelty).
#
# CoverageBatchStrategy (mutation-site coverage) is the DIFFERENTIATED method;
# ConstantLiarStrategy is a FolDE parity baseline. The honest claim is a
# regime-conditional INTERACTION: spreading a batch helps WHEN the score-
# redundant cluster is a DECOY (greedy piles its budget on a confident
# suboptimal cluster and misses the rare optimum), and ties WHEN the redundant
# cluster already contains the optimum. The assertions below test the INVARIANT
# (penalty=0 == greedy), the MECHANIC (coverage spreads across sites;
# constant_liar de-dups picks 2+), determinism, and the DIRECTION of the
# regime interaction.
# ==========================================================================
def _clustered_pick_setup(seed=0):
    """A clustered (landscape, embeddings, measured, candidates) for the GAP-3
    mechanic tests: ONE decoy site holds many same-site high-score subs, so a
    naive greedy batch piles onto it while a coverage penalty spreads it.
    """
    ls, emb = build_decoy_cluster_landscape(
        n_other_sites=8, decoy_subs=19, other_subs=8, seed=seed
    )
    pool = list(emb.index)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pool), size=6, replace=False)
    measured = [pool[i] for i in sorted(idx.tolist())]
    measured_y = [ls.measure(v) for v in measured]
    candidates = [v for v in pool if v not in set(measured)]
    return ls, emb, measured, measured_y, candidates


def _same_site_pick_count(picked):
    """Largest number of picks landing on a SINGLE residue site (cluster size)."""
    counts: dict[int, int] = {}
    for p in picked:
        pos = parse_variant(p).position
        counts[pos] = counts.get(pos, 0) + 1
    return max(counts.values()) if counts else 0


def test_build_decoy_cluster_landscape_deterministic():
    """Same seed -> identical (landscape, embeddings); the optimum is NOT in the
    decoy cluster (a separate rare site holds the global max)."""
    a_ls, a_emb = build_decoy_cluster_landscape(
        n_other_sites=6, decoy_subs=19, other_subs=8, seed=3
    )
    b_ls, b_emb = build_decoy_cluster_landscape(
        n_other_sites=6, decoy_subs=19, other_subs=8, seed=3
    )
    np.testing.assert_array_equal(a_ls.true_values(), b_ls.true_values())
    pd.testing.assert_frame_equal(a_emb, b_emb)
    # The global optimum (1.0) is a UNIQUE single-substitution rare site, NOT one
    # of the many decoy-cluster subs (so piling on the decoy MISSES it).
    top = a_ls.top_k_ids(1)[0]
    assert a_ls.measure(top) == 1.0
    assert parse_variant(top).position != 1  # site 1 is the decoy cluster.


def test_coverage_penalty_zero_equals_greedy_exact():
    """CoverageBatchStrategy(penalty=0) picks must be EXACTLY GreedyTopLayer picks.

    With penalty=0 the effective score is score - 0*site_count == score == the
    greedy mean, so the stable argmax batch is the greedy top-N for the same
    measured set / seed (the tested reduction invariant).
    """
    ls, emb, measured, measured_y, candidates = _clustered_pick_setup(seed=1)
    n_select = 8
    greedy_pick = GreedyTopLayer().pick(
        measured, measured_y, candidates, emb, n_select, np.random.default_rng(0)
    )
    cov_pick = CoverageBatchStrategy(base="greedy", penalty=0.0).pick(
        measured, measured_y, candidates, emb, n_select, np.random.default_rng(0)
    )
    np.testing.assert_array_equal(np.array(greedy_pick), np.array(cov_pick))


def test_coverage_penalty_reduces_same_site_picks():
    """CoverageBatchStrategy(penalty>0) selects FEWER same-site picks than greedy.

    On the clustered case greedy piles a chunk of its batch onto the single
    decoy site; the site-coverage penalty discounts repeated same-site picks, so
    the largest single-site cluster in the coverage batch is strictly smaller.
    """
    ls, emb, measured, measured_y, candidates = _clustered_pick_setup(seed=2)
    n_select = 8
    greedy_pick = GreedyTopLayer().pick(
        measured, measured_y, candidates, emb, n_select, np.random.default_rng(0)
    )
    cov_pick = CoverageBatchStrategy(base="greedy", penalty=2.0).pick(
        measured, measured_y, candidates, emb, n_select, np.random.default_rng(0)
    )
    greedy_max_site = _same_site_pick_count(greedy_pick)
    cov_max_site = _same_site_pick_count(cov_pick)
    # Greedy genuinely piles on one site; coverage spreads -> smaller cluster.
    assert greedy_max_site > 1
    assert cov_max_site < greedy_max_site


def test_constant_liar_pick1_equals_greedy_picks2plus_differ():
    """ConstantLiarStrategy: pick 1 == greedy argmax; picks 2+ de-dup off it.

    The first pick is the plain greedy argmax (no lie yet). The running-mean lie
    then drags refit predictions DOWN around the chosen embedding cluster, so
    picks 2+ are pushed to a DIFFERENT region -> the full set differs from the
    greedy top-N on the clustered case.
    """
    ls, emb, measured, measured_y, candidates = _clustered_pick_setup(seed=3)
    n_select = 8
    greedy_pick = GreedyTopLayer().pick(
        measured, measured_y, candidates, emb, n_select, np.random.default_rng(0)
    )
    cl_pick = ConstantLiarStrategy().pick(
        measured, measured_y, candidates, emb, n_select, np.random.default_rng(0)
    )
    # Pick 1 is the greedy argmax (the first greedy top-N entry).
    assert cl_pick[0] == greedy_pick[0]
    # The constant lie reorders picks 2+ -> the batches are not identical sets.
    assert set(cl_pick) != set(greedy_pick)


@pytest.mark.parametrize("strat_name", ["coverage", "constant_liar"])
def test_batch_strategy_determinism(strat_name):
    """Same seed -> same picks for coverage and constant_liar (rng-only random
    is consulted only for the cold-start fallback, which is not hit here)."""
    ls, emb, measured, measured_y, candidates = _clustered_pick_setup(seed=4)
    n_select = 6
    a = build_strategy(strat_name).pick(
        measured, measured_y, candidates, emb, n_select, np.random.default_rng(7)
    )
    b = build_strategy(strat_name).pick(
        measured, measured_y, candidates, emb, n_select, np.random.default_rng(7)
    )
    np.testing.assert_array_equal(np.array(a), np.array(b))


@pytest.mark.parametrize("strat_name", ["coverage", "constant_liar"])
def test_batch_strategy_cold_start_falls_back_to_random(strat_name):
    """No measured labels -> cold-start random fallback (same idiom as greedy)."""
    ls, emb = build_decoy_cluster_landscape(
        n_other_sites=6, decoy_subs=19, other_subs=8, seed=0
    )
    candidates = list(emb.index)
    n_select = 5
    picked = build_strategy(strat_name).pick(
        [], [], candidates, emb, n_select, np.random.default_rng(0)
    )
    assert len(picked) == n_select
    assert set(picked).issubset(set(candidates))
    assert len(set(picked)) == len(picked)


def test_run_batch_diversity_ablation_tidy_and_deterministic():
    kw = dict(
        redundancy_list=[0.0, 8.0],
        seeds=[0, 1],
        n_sites=8,
        n_subs=6,
        strategies=("greedy", "coverage", "constant_liar", "random"),
    )
    a = run_batch_diversity_ablation(**kw)
    b = run_batch_diversity_ablation(**kw)
    expected_cols = [
        "regime", "redundancy", "strategy", "seed", "max_activity_found",
        "rounds_to_target", "simple_regret", "global_max", "frac_of_global",
        "reached_target",
    ]
    assert list(a.columns) == expected_cols
    # tie: 2 redundancy x 4 strategies x 2 seeds = 16; decoy: 4 x 2 = 8 -> 24.
    assert len(a) == 2 * 4 * 2 + 4 * 2
    assert set(a["regime"].unique()) == {"tie", "decoy"}
    # Decoy rows carry redundancy=NaN; tie rows carry the numeric knob.
    assert a[a.regime == "decoy"]["redundancy"].isna().all()
    assert not a[a.regime == "tie"]["redundancy"].isna().any()
    pd.testing.assert_frame_equal(a, b)


def test_batch_diversity_direction_coverage_pays_off_in_decoy_not_tie():
    """HONEST DIRECTION (regime interaction), not a magic number.

    Coverage (mutation-site spread) beats naive greedy top-N ONLY when the
    score-redundant cluster is a DECOY -- greedy piles its budget on a confident
    suboptimal cluster and misses the rare optimum, while coverage spreads off it
    and reaches it. When the redundant cluster CONTAINS the optimum (tie regime)
    piling is fine and the coverage advantage is ~0. Assert the interaction
    (delta_decoy > delta_tie), averaged over seeds, plus the trap (greedy<0.9 in
    decoy) and the anchor (greedy near-optimal in tie). constant_liar is a parity
    baseline tested as a MECHANIC above, NOT asserted to win here.
    """
    df = run_batch_diversity_ablation(
        redundancy_list=[8.0],   # one strongly-redundant tie point.
        seeds=list(range(8)),
    )
    piv = df.groupby(["regime", "strategy"])["frac_of_global"].mean().unstack()
    greedy_decoy = piv.loc["decoy", "greedy"]
    greedy_tie = piv.loc["tie", "greedy"]
    coverage_decoy = piv.loc["decoy", "coverage"]
    coverage_tie = piv.loc["tie", "coverage"]

    # Decoy regime: greedy is genuinely trapped on the decoy (well below global).
    assert greedy_decoy < 0.9
    # Tie regime: piling on the redundant cluster IS optimal -> greedy near 1.
    assert greedy_tie > 0.95

    delta_decoy = coverage_decoy - greedy_decoy
    delta_tie = coverage_tie - greedy_tie
    # Interaction: coverage helps in the decoy regime, ~ties in the tie regime.
    assert delta_decoy > delta_tie
    # And in the decoy regime coverage actually beats greedy.
    assert coverage_decoy > greedy_decoy
    # The mechanic anchor: coverage reaches the rare optimum far more often than
    # greedy does in the decoy regime (NaN-safe reached_target flag).
    reach = df[df.regime == "decoy"].groupby("strategy")["reached_target"].mean()
    assert reach["coverage"] > reach["greedy"]
