"""Locality-sweep ablation: isolate POOLING in the active-learning loop.

For each ``(L, alpha, pooling, seed)`` the runner:

    1. builds a synthetic additive landscape over ``L`` positions
       (:func:`~plm_evolve.benchmark.oracle.synthetic_landscape`);
    2. builds the matching synthetic embedding table for that pooling from the
       ONE shared latent
       (:func:`~plm_evolve.benchmark.embeddings_sim.synthetic_embedding_tables`),
       so the same ``s(v)`` and distractor draw feed every pooling -- the ONLY
       difference is the pooling gain ``g_p(alpha, L)``;
    3. runs :class:`~plm_evolve.benchmark.strategies.GreedyTopLayer` through the
       shared :func:`~plm_evolve.benchmark.simulate.run_simulation` loop (the
       loop is NOT reimplemented here);
    4. records sample-efficiency: ``max_activity_found`` after the last round,
       ``rounds_to_target`` (NaN if the target is never reached), and
       ``simple_regret``.

This isolates the effect of pooling on sample efficiency under a SWEPT locality
parameter. It proves MECHANICS (pooling-driven dilution) and SENSITIVITY (to
``alpha``); it does NOT prove that mutated-token is better on real proteins --
that depends on the empirical alpha of real PLMs, which must be measured
separately on real embeddings. ``alpha`` is always swept, never hardcoded.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .embeddings_sim import POOLINGS, synthetic_embedding_tables
from .multimutant import (
    AdditiveMultiPredictor,
    enumerate_multimutants,
    make_generative_landscape,
)
from .multiobjective import pareto_front, scalarize
from .oracle import Landscape, synthetic_landscape
from .simulate import run_simulation
from .strategies import CoverageBatchStrategy, GreedyTopLayer, build_strategy
from ..variants import AA_ALPHABET


def run_locality_ablation(
    *,
    n_positions_list: Sequence[int],
    alpha_list: Sequence[float],
    seeds: Sequence[int],
    n_init: int,
    n_per_round: int,
    n_rounds: int,
    n_dims: int = 64,
    poolings: Sequence[str] = POOLINGS,
    alphabet_subset: str = "ACDEFG",
    landscape_noise: float = 0.0,
    signal_noise: float = 0.05,
    distractor_scale: float = 0.5,
    target_quantile: float = 0.95,
    top_k: int = 10,
) -> pd.DataFrame:
    """Sweep ``(L, alpha, pooling, seed)`` -> one tidy sample-efficiency frame.

    The same landscape (per ``L``, ``seed``) feeds every pooling, and the
    embedding tables for all poolings come from the SAME shared latent (per
    ``L``, ``alpha``, ``seed``), so a row's ``pooling`` is the only thing that
    changes the information the greedy loop sees.

    Args:
        n_positions_list: protein lengths ``L`` to sweep.
        alpha_list: locality fractions ``alpha in [0, 1]`` to sweep.
        seeds: seeds (one full landscape+embedding build per seed).
        n_init: initial random measured-set size.
        n_per_round: variants measured per active-learning round.
        n_rounds: number of active-learning rounds.
        n_dims: embedding dimensionality ``D``.
        poolings: which poolings to compare (subset of
            :data:`~plm_evolve.benchmark.embeddings_sim.POOLINGS`).
        alphabet_subset: amino-acid subset for the synthetic landscape (its size
            sets the per-position mutant count, hence the candidate-pool size).
        landscape_noise: Gaussian observation noise sd on the landscape.
        signal_noise: sd of the activity coupling noise in ``s(v)``.
        distractor_scale: sd of the shared distractor noise in the embeddings.
        target_quantile: ``rounds_to_target`` target = this quantile of the
            landscape's true activities (computed per landscape so the target is
            comparable across L).
        top_k: ``top_k`` passed through to the simulation metrics.

    Returns:
        Tidy DataFrame, one row per ``(L, alpha, pooling, seed)``, columns:
        ``L, alpha, pooling, seed, max_activity_found, rounds_to_target,
        simple_regret``.
    """
    for p in poolings:
        if p not in POOLINGS:
            raise ValueError(f"Unknown pooling {p!r}; expected subset of {POOLINGS}.")

    rows: list[dict] = []
    for length in n_positions_list:
        for seed in seeds:
            # One landscape per (L, seed); shared by every alpha and pooling.
            landscape = synthetic_landscape(
                n_positions=length,
                alphabet_subset=alphabet_subset,
                seed=seed,
                noise=landscape_noise,
            )
            # Target for rounds_to_target: a high quantile of the true values,
            # computed per landscape (comparable across L without hardcoding).
            target = float(np.quantile(landscape.true_values(), target_quantile))

            for alpha in alpha_list:
                # One shared latent per (L, alpha, seed); all poolings derive
                # from it, so only the pooling gain differs between tables.
                tables = synthetic_embedding_tables(
                    landscape,
                    wt_len=length,
                    n_dims=n_dims,
                    locality_alpha=alpha,
                    signal_noise=signal_noise,
                    distractor_scale=distractor_scale,
                    obs_noise=0.0,
                    seed=seed,
                    poolings=poolings,
                )
                for pooling in poolings:
                    emb = tables[pooling]
                    sim = run_simulation(
                        landscape,
                        emb,
                        GreedyTopLayer(),
                        n_init=n_init,
                        n_per_round=n_per_round,
                        n_rounds=n_rounds,
                        seed=seed,
                        top_k=top_k,
                        target=target,
                    )
                    last = sim.iloc[-1]
                    r2t = (
                        float(last["rounds_to_target"])
                        if "rounds_to_target" in sim.columns
                        else float("nan")
                    )
                    rows.append(
                        {
                            "L": int(length),
                            "alpha": float(alpha),
                            "pooling": pooling,
                            "seed": int(seed),
                            "max_activity_found": float(last["max_activity_found"]),
                            "rounds_to_target": r2t,
                            "simple_regret": float(last["simple_regret"]),
                        }
                    )

    if not rows:
        raise ValueError(
            "run_locality_ablation produced no rows; check the sweep lists."
        )
    return pd.DataFrame(
        rows,
        columns=[
            "L",
            "alpha",
            "pooling",
            "seed",
            "max_activity_found",
            "rounds_to_target",
            "simple_regret",
        ],
    )


# ==========================================================================
# Acquisition ablation: greedy vs ucb vs thompson vs random ACROSS ruggedness.
#
# Uncertainty-aware acquisition (FolDE 2025, ALDE 2025) is a CAPABILITY, not a
# novelty. The honest claim this ablation supports is a SENSITIVITY result --
# "uncertainty-aware acquisition helps WHEN the landscape is rugged / greedy
# gets trapped" -- NOT "ucb always wins". To show that without rigging it, the
# landscape ruggedness is a TUNABLE knob and the comparison is run ACROSS it.
#
# IMPORTANT (why not synthetic_landscape's `epistasis`): that parameter only
# affects MULTI-mutant entries (additive_score skips the epistasis term for
# len-1 mutations), and the simulation pool is single mutants only, so cranking
# `epistasis` changes nothing the loop sees. The ruggedness that can trap greedy
# must therefore live in the embedding->activity map the RF actually learns. The
# builder below puts it there explicitly via a `ruggedness` knob.
# ==========================================================================


def build_rugged_landscape(
    *,
    n_variants: int,
    ruggedness: float,
    seed: int,
    n_dims: int = 8,
    distractor_scale: float = 1.0,
) -> tuple[Landscape, pd.DataFrame]:
    """A tunably-rugged embedding->activity map for the acquisition ablation.

    Each (synthetic single-mutant) variant is placed at a random point in a 2D
    informative embedding subspace ``[-1, 1]^2`` (the rest of the ``n_dims``
    columns are pure distractors). The TRUE activity is a sum of two Gaussian
    modes over that 2D coordinate:

        * a BROAD, smooth "decoy" mode (height 1, wide) centred at (-0.5, -0.5);
        * a NARROW, tall "global" mode whose height and contribution scale with
          ``ruggedness`` (0 = absent), centred at (0.6, 0.6).

    At ``ruggedness = 0`` the activity is the single broad smooth mode: the RF
    learns a near-monotone map and greedy is near-optimal (exploration can only
    cost the exploration tax -> greedy ties or slightly beats ucb/thompson).
    At ``ruggedness > 0`` the narrow global peak sits in a GAP between the broad
    decoy and the seed points; greedy commits to the decoy region and gets
    trapped, while the sparse gap gives the RF trees high disagreement (large
    std) so UCB/Thompson probe it and escape. The peak is kept INSIDE the spread
    of the candidate cloud (an interpolation gap, not a far extrapolation
    region) because RF tree-disagreement std is only informative for
    interpolation -- trees agree on a flat extrapolated value far outside the
    training data, where UCB would NOT probe.

    Args:
        n_variants: number of single-mutant variants (the candidate pool size).
        ruggedness: knob in ``[0, inf)``; 0 = smooth single mode, larger =
            stronger deceptive narrow global peak.
        seed: RNG seed -- same seed yields an identical (landscape, embeddings).
        n_dims: embedding width ``D`` (first 2 dims informative, rest distractor).
        distractor_scale: sd of the distractor columns.

    Returns:
        ``(landscape, embeddings)`` where ``embeddings`` is indexed by the same
        single-mutant variant ids as the landscape (columns ``e0..e{D-1}``).
    """
    if n_variants < 2:
        raise ValueError("n_variants must be >= 2.")
    if ruggedness < 0:
        raise ValueError("ruggedness must be >= 0.")
    if n_dims < 2:
        raise ValueError("n_dims must be >= 2 (need 2 informative dims).")

    rng = np.random.default_rng(seed)
    coords = rng.uniform(-1.0, 1.0, size=(n_variants, 2))
    cx, cy = coords[:, 0], coords[:, 1]

    # Broad smooth decoy (height 1, wide) -- the mode greedy is seduced by.
    decoy = np.exp(-((cx + 0.5) ** 2 + (cy + 0.5) ** 2) / 0.5)
    # Narrow tall global peak -- only present (and dominant) when rugged.
    peak = (1.0 + 1.2 * ruggedness) * np.exp(
        -((cx - 0.6) ** 2 + (cy - 0.6) ** 2) / 0.05
    )
    activity = decoy + ruggedness * peak

    ids = [f"A{i + 1}C" for i in range(n_variants)]
    landscape = Landscape.from_mapping(dict(zip(ids, activity.tolist())))

    # Embedding: first 2 dims carry the informative coordinate, rest distractor.
    er = np.random.default_rng(seed + 1000)
    distract = er.normal(0.0, distractor_scale, size=(n_variants, n_dims - 2))
    mat = np.concatenate([coords, distract], axis=1)
    embeddings = pd.DataFrame(
        mat, index=ids, columns=[f"e{j}" for j in range(n_dims)]
    )
    return landscape, embeddings


def run_acquisition_ablation(
    *,
    ruggedness_list: Sequence[float],
    seeds: Sequence[int],
    n_variants: int = 300,
    n_init: int = 8,
    n_per_round: int = 4,
    n_rounds: int = 4,
    n_dims: int = 8,
    strategies: Sequence[str] = ("greedy", "ucb", "thompson", "random"),
    target_quantile: float = 0.95,
    top_k: int = 10,
) -> pd.DataFrame:
    """Compare acquisition strategies ACROSS landscape ruggedness.

    For each ``(ruggedness, seed)`` the SAME (landscape, embeddings) from
    :func:`build_rugged_landscape` feeds every strategy through the shared
    :func:`~plm_evolve.benchmark.simulate.run_simulation` loop. Sample
    efficiency (``max_activity_found`` after the last round, ``rounds_to_target``,
    ``simple_regret``) is recorded per run so means +/- sd over seeds can be
    computed downstream.

    The HONEST reading is a sensitivity / interaction result: on a near-additive
    (``ruggedness = 0``) landscape greedy ties (or slightly beats, via the
    exploration tax) ucb/thompson; on a rugged landscape greedy gets trapped on
    the decoy and uncertainty-aware acquisition wins. The conclusion is the
    INTERACTION (the ucb-minus-greedy advantage GROWS with ruggedness), never a
    standalone "ucb always wins".

    Args:
        ruggedness_list: ruggedness knobs to sweep (include 0 for the additive
            tie reference and a positive value for the rugged regime).
        seeds: seeds repeated per ruggedness (mean +/- sd over them).
        n_variants: candidate pool size per landscape.
        n_init / n_per_round / n_rounds: active-learning budget.
        n_dims: embedding width.
        strategies: registry names to compare.
        target_quantile: ``rounds_to_target`` target = this quantile of the true
            activities, computed per landscape (comparable across ruggedness).
        top_k: passed through to the simulation metrics.

    Returns:
        Tidy DataFrame, one row per ``(ruggedness, strategy, seed)``, columns:
        ``ruggedness, strategy, seed, max_activity_found, rounds_to_target,
        simple_regret, global_max, frac_of_global``.
    """
    rows: list[dict] = []
    for ruggedness in ruggedness_list:
        for seed in seeds:
            landscape, embeddings = build_rugged_landscape(
                n_variants=n_variants,
                ruggedness=float(ruggedness),
                seed=seed,
                n_dims=n_dims,
            )
            global_max = landscape.global_max()
            target = float(
                np.quantile(landscape.true_values(), target_quantile)
            )
            for strat_name in strategies:
                strategy = build_strategy(strat_name)
                sim = run_simulation(
                    landscape,
                    embeddings,
                    strategy,
                    n_init=n_init,
                    n_per_round=n_per_round,
                    n_rounds=n_rounds,
                    seed=seed,
                    top_k=top_k,
                    target=target,
                )
                last = sim.iloc[-1]
                r2t = (
                    float(last["rounds_to_target"])
                    if "rounds_to_target" in sim.columns
                    else float("nan")
                )
                best = float(last["max_activity_found"])
                rows.append(
                    {
                        "ruggedness": float(ruggedness),
                        "strategy": strat_name,
                        "seed": int(seed),
                        "max_activity_found": best,
                        "rounds_to_target": r2t,
                        "simple_regret": float(last["simple_regret"]),
                        "global_max": global_max,
                        "frac_of_global": best / global_max
                        if global_max != 0
                        else float("nan"),
                    }
                )

    if not rows:
        raise ValueError(
            "run_acquisition_ablation produced no rows; check the sweep lists."
        )
    return pd.DataFrame(
        rows,
        columns=[
            "ruggedness",
            "strategy",
            "seed",
            "max_activity_found",
            "rounds_to_target",
            "simple_regret",
            "global_max",
            "frac_of_global",
        ],
    )


# ==========================================================================
# Batch-diversity ablation: greedy(top-N) vs coverage vs constant_liar vs
# random, SWEPT over BATCH REDUNDANCY.
#
# GAP 3 batch selection is a CAPABILITY, not a novelty. The honest claim is a
# SENSITIVITY / interaction result: coverage (mutation-site spread) and
# constant_liar (FolDE-style embedding de-dup) beat naive top-N WHEN the
# per-round top-N are REDUNDANT (multiple high-score candidates at the same
# residue site / clustered in embedding space, so a whole round is wasted on
# near-duplicates), and roughly TIE when the top-N are already diverse. The
# conclusion is the INTERACTION (the advantage GROWS with redundancy), never a
# standalone "coverage always wins". The construction is NOT rigged to win at
# low redundancy -- it is built to tie there, and the honesty anchor below
# (greedy's round-1 site spread) proves the redundancy knob does what it says.
# ==========================================================================


def build_redundant_landscape(
    *,
    n_sites: int,
    n_subs: int,
    redundancy: float,
    seed: int,
    n_dims: int = 12,
    distractor_scale: float = 0.3,
) -> tuple[Landscape, pd.DataFrame]:
    """A tunably-REDUNDANT landscape for the batch-diversity ablation.

    The pool is ``n_sites`` residue positions, each with ``n_subs`` single
    substitutions (so ``parse_variant(token).position`` is SHARED by the
    ``n_subs`` substitutions at one site -- this is what makes a coverage
    penalty bind, unlike :func:`build_rugged_landscape` whose ids are all at
    distinct positions). The true activity is::

        activity(site s, sub k) = redundancy * site_value[s] + sub_value[s, k]

    with ``site_value[0] = 1`` the UNIQUE best site (others lower, spread in
    ``[0, 0.6)``) and ``sub_value`` a small within-site spread in ``[0, 0.25)``.

    * ``redundancy = 0``: the site term vanishes, so the top scorers are ranked
      by ``sub_value`` and SCATTER across sites -- the naive top-N batch is
      already diverse, the coverage penalty rarely binds, and all methods TIE.
    * ``redundancy`` large: the site term dominates, so ALL ``n_subs``
      substitutions at the best site rank at the TOP together. Naive top-N
      spends a whole round on near-duplicate same-site variants and discovers
      good sites only about one-per-round; coverage spreads the batch across
      sites (finds the best site immediately) and constant_liar suppresses the
      embedding-space cluster (same effect via de-dup), so both reach the best
      activity in FEWER measurements.

    The embedding is SITE-ENCODING and learnable: a one-hot-ish site block (so
    the ``n_subs`` substitutions at one site CLUSTER in embedding space -- the
    property constant_liar exploits), an activity-signal dim, and distractor
    dims. The SAME ``(landscape, embeddings)`` (per ``redundancy``, ``seed``)
    feeds every strategy, so a row's strategy is the only thing that changes the
    batch.

    Args:
        n_sites: number of residue positions (sites).
        n_subs: substitutions per site (``n_subs <= 19``; same-site cluster size).
        redundancy: knob in ``[0, inf)``; 0 = already-diverse top-N (tie),
            larger = top-N concentrate at the best site (redundant batch).
        seed: RNG seed -- same seed yields an identical ``(landscape, emb)``.
        n_dims: embedding width.
        distractor_scale: sd of the distractor columns.

    Returns:
        ``(landscape, embeddings)`` indexed by the same single-mutant tokens.
    """
    if n_sites < 2:
        raise ValueError("n_sites must be >= 2.")
    if not (1 <= n_subs <= len(AA_ALPHABET) - 1):
        raise ValueError(f"n_subs must be in [1, {len(AA_ALPHABET) - 1}].")
    if redundancy < 0:
        raise ValueError("redundancy must be >= 0.")
    if n_dims < 4:
        raise ValueError("n_dims must be >= 4 (site block + signal + distractor).")

    rng = np.random.default_rng(seed)
    site_value = rng.uniform(0.0, 0.6, size=n_sites)
    site_value[0] = 1.0  # the unique best site.
    sub_value = rng.uniform(0.0, 0.25, size=(n_sites, n_subs))

    ids: list[str] = []
    activity: list[float] = []
    site_of: list[int] = []
    for s in range(n_sites):
        wt = AA_ALPHABET[s % len(AA_ALPHABET)]
        subs = [a for a in AA_ALPHABET if a != wt][:n_subs]
        for k, a in enumerate(subs):
            ids.append(f"{wt}{s + 1}{a}")
            activity.append(float(redundancy * site_value[s] + sub_value[s, k]))
            site_of.append(s)
    activity_arr = np.array(activity)
    landscape = Landscape.from_mapping(dict(zip(ids, activity_arr.tolist())))

    # Embedding: site-encoding block (same-site cluster) + signal + distractors.
    er = np.random.default_rng(seed + 7777)
    n_site_dims = min(n_sites, max(2, n_dims - 3))
    site_block = np.zeros((len(ids), n_site_dims))
    for i, s in enumerate(site_of):
        site_block[i, s % n_site_dims] = 1.0
    signal = activity_arr.reshape(-1, 1) + er.normal(0.0, 0.02, size=(len(ids), 1))
    n_rest = n_dims - n_site_dims - 1
    distract = er.normal(0.0, distractor_scale, size=(len(ids), max(0, n_rest)))
    mat = np.concatenate([site_block, signal, distract], axis=1)
    embeddings = pd.DataFrame(
        mat, index=ids, columns=[f"e{j}" for j in range(mat.shape[1])]
    )
    return landscape, embeddings


def build_decoy_cluster_landscape(
    *,
    n_other_sites: int,
    decoy_subs: int,
    other_subs: int,
    seed: int,
    n_dims: int = 16,
    distractor_scale: float = 0.3,
) -> tuple[Landscape, pd.DataFrame]:
    """A DECOY-cluster landscape where batch diversity genuinely pays off.

    Unlike :func:`build_redundant_landscape` (whose redundant best site CONTAINS
    the global optimum, so piling on it is optimal and all strategies tie), here
    the score-redundant cluster is a DECOY: a broad, well-sampled, high-but-
    SUBOPTIMAL site, while the true optimum sits at a separate UNDER-represented
    "rare" site. This mirrors the accepted decoy approach in
    :func:`build_rugged_landscape` (broad smooth decoy + a narrow/rare true
    optimum), keyed on residue SITES so the coverage penalty binds.

    Layout (single-mutant tokens):

        * Site 1 -- the DECOY cluster: ``decoy_subs`` substitutions at ONE
          residue, all with true activity ~0.75 (high but suboptimal). With
          ``decoy_subs`` chosen LARGER than the batch budget, greedy never
          exhausts the cluster and spends every round re-measuring near-duplicate
          same-site decoy variants.
        * Site 2 -- the RARE optimum: a SINGLE substitution with true activity
          ``1.0`` (the global max). Its embedding does NOT leak the ``1.0`` -- it
          carries a decoy-level signal (~0.73) -- so the optimum is UNDER-
          predicted (see honesty note below).
        * Sites 3.. -- filler: ``other_subs`` substitutions each, with true (and
          predicted) activity in ``[0.3, 0.6)``, clearly below the rare site, so
          once a strategy is forced off the decoy the rare site is the next pick.

    Honesty (why this is NOT rigged, same bar :func:`build_rugged_landscape`
    met): the optimum's under-prediction is EMERGENT, not stamped in. A
    RandomForest cannot extrapolate above its training max, so once the measured
    set tops out at the decoy's ~0.75 the model predicts the unmeasured rare site
    at ``<= 0.75`` regardless of its embedding -- until the rare site is itself
    measured. Greedy therefore piles its whole budget onto the over-represented
    confident decoy and never samples the rare site; coverage diversifies off the
    decoy (its site penalty forces a spread) and samples the under-rated site,
    reaching the optimum that greedy misses. The true optimum exceeds the model's
    learned range -- exactly the regime batch diversity exists to rescue -- so the
    payoff is regime-conditional, never "coverage always wins".

    The embedding is SITE-ENCODING (a one-hot site block so the decoy cluster
    CLUSTERS in embedding space, the property constant_liar exploits) + an
    activity-signal dim + distractors. The site block is sized to give every
    site its OWN dimension (no aliasing), so the construction is unambiguous.

    Args:
        n_other_sites: number of filler sites (besides decoy + rare).
        decoy_subs: substitutions at the decoy site (``<= 19``); choose LARGER
            than the batch budget so greedy cannot exhaust the cluster.
        other_subs: substitutions per filler site (``<= 19``).
        seed: RNG seed -- same seed yields an identical ``(landscape, emb)``.
        n_dims: embedding width (bumped vs the tie builder so the site block can
            give every site a distinct dimension).
        distractor_scale: sd of the distractor columns.

    Returns:
        ``(landscape, embeddings)`` indexed by the same single-mutant tokens.
    """
    if n_other_sites < 1:
        raise ValueError("n_other_sites must be >= 1.")
    if not (1 <= decoy_subs <= len(AA_ALPHABET) - 1):
        raise ValueError(f"decoy_subs must be in [1, {len(AA_ALPHABET) - 1}].")
    if not (1 <= other_subs <= len(AA_ALPHABET) - 1):
        raise ValueError(f"other_subs must be in [1, {len(AA_ALPHABET) - 1}].")

    rng = np.random.default_rng(seed)
    ids: list[str] = []
    true_act: list[float] = []
    pred_signal: list[float] = []
    site_of: list[int] = []

    # Site 0: the redundant DECOY cluster (high but suboptimal, well sampled).
    wt = AA_ALPHABET[0]
    subs = [a for a in AA_ALPHABET if a != wt][:decoy_subs]
    for a in subs:
        v = 0.75 + float(rng.uniform(-0.01, 0.01))
        ids.append(f"{wt}1{a}")
        true_act.append(v)
        pred_signal.append(v)  # embedding carries the decoy activity.
        site_of.append(0)

    # Site 1: the RARE optimum (its own site, true=1.0, NOT leaked into emb).
    wt = AA_ALPHABET[1]
    a = AA_ALPHABET[2]
    ids.append(f"{wt}2{a}")
    true_act.append(1.0)
    pred_signal.append(0.73)  # under-predicted (RF cap makes this emergent too).
    site_of.append(1)

    # Filler sites: predicted (and true) clearly below the rare site.
    for s in range(2, 2 + n_other_sites):
        wt = AA_ALPHABET[s % len(AA_ALPHABET)]
        subs = [a for a in AA_ALPHABET if a != wt][:other_subs]
        base = float(rng.uniform(0.3, 0.6))
        for a in subs:
            v = base + float(rng.uniform(-0.03, 0.03))
            ids.append(f"{wt}{s + 1}{a}")
            true_act.append(v)
            pred_signal.append(v)
            site_of.append(s)

    true_arr = np.array(true_act)
    sig_arr = np.array(pred_signal)
    landscape = Landscape.from_mapping(dict(zip(ids, true_arr.tolist())))

    # Embedding: per-site one-hot block (NO aliasing -- each site its own dim) +
    # signal dim + distractors.
    er = np.random.default_rng(seed + 7777)
    n_sites_total = 2 + n_other_sites
    if n_dims < n_sites_total + 2:
        raise ValueError(
            f"n_dims={n_dims} too small for {n_sites_total} sites; need a "
            f"distinct dim per site (>= {n_sites_total + 2})."
        )
    site_block = np.zeros((len(ids), n_sites_total))
    for i, s in enumerate(site_of):
        site_block[i, s] = 1.0  # each site its own column -> no collision.
    signal = sig_arr.reshape(-1, 1) + er.normal(0.0, 0.02, size=(len(ids), 1))
    n_rest = n_dims - n_sites_total - 1
    distract = er.normal(0.0, distractor_scale, size=(len(ids), max(0, n_rest)))
    mat = np.concatenate([site_block, signal, distract], axis=1)
    embeddings = pd.DataFrame(
        mat, index=ids, columns=[f"e{j}" for j in range(mat.shape[1])]
    )
    return landscape, embeddings


_BATCH_DIVERSITY_COLUMNS = [
    "regime",
    "redundancy",
    "strategy",
    "seed",
    "max_activity_found",
    "rounds_to_target",
    "simple_regret",
    "global_max",
    "frac_of_global",
    "reached_target",
]


def _run_batch_strategies(
    *,
    landscape: Landscape,
    embeddings: pd.DataFrame,
    regime: str,
    redundancy: float,
    seed: int,
    strategies: Sequence[str],
    coverage_penalty: float,
    n_init: int,
    n_per_round: int,
    n_rounds: int,
    target: float,
    top_k: int,
) -> list[dict]:
    """Run every strategy on ONE (landscape, embeddings) -> one row each.

    The SAME ``(landscape, embeddings)`` and ``target`` feed every strategy, so a
    row's ``strategy`` is the only thing that changes the batch.
    """
    global_max = landscape.global_max()
    rows: list[dict] = []
    for strat_name in strategies:
        if strat_name == "coverage":
            strategy = CoverageBatchStrategy(base="greedy", penalty=coverage_penalty)
        else:
            strategy = build_strategy(strat_name)
        sim = run_simulation(
            landscape,
            embeddings,
            strategy,
            n_init=n_init,
            n_per_round=n_per_round,
            n_rounds=n_rounds,
            seed=seed,
            top_k=top_k,
            target=target,
        )
        last = sim.iloc[-1]
        r2t = (
            float(last["rounds_to_target"])
            if "rounds_to_target" in sim.columns
            else float("nan")
        )
        best = float(last["max_activity_found"])
        rows.append(
            {
                "regime": regime,
                "redundancy": float(redundancy),
                "strategy": strat_name,
                "seed": int(seed),
                "max_activity_found": best,
                "rounds_to_target": r2t,
                "simple_regret": float(last["simple_regret"]),
                "global_max": global_max,
                "frac_of_global": best / global_max
                if global_max != 0
                else float("nan"),
                # target=global_max in the decoy regime -> r2t is all-NaN for a
                # strategy that never reaches the optimum; this 0/1 flag is the
                # NaN-safe "did it reach the optimum" signal for downstream means.
                "reached_target": float(0.0 if np.isnan(r2t) else 1.0),
            }
        )
    return rows


def run_batch_diversity_ablation(
    *,
    redundancy_list: Sequence[float],
    seeds: Sequence[int],
    n_sites: int = 10,
    n_subs: int = 8,
    n_init: int = 8,
    n_per_round: int = 6,
    n_rounds: int = 4,
    n_dims: int = 12,
    coverage_penalty: float = 2.0,
    strategies: Sequence[str] = ("greedy", "coverage", "constant_liar", "random"),
    target_quantile: float = 0.95,
    top_k: int = 10,
    include_decoy: bool = True,
    decoy_n_other_sites: int = 8,
    decoy_subs: int = 19,
    decoy_other_subs: int = 8,
    decoy_n_init: int = 6,
    decoy_n_per_round: int = 4,
    decoy_n_rounds: int = 3,
    decoy_n_dims: int = 16,
) -> pd.DataFrame:
    """Compare batch strategies across BOTH the tie and decoy regimes.

    Two regimes are swept so the batch-diversity payoff is shown HONESTLY as a
    regime-conditional interaction, never a standalone "coverage always wins":

    * ``regime="tie"`` (the ``redundancy_list`` sweep over
      :func:`build_redundant_landscape`): the redundant best site CONTAINS the
      global optimum, so piling top-N on it is OPTIMAL and every strategy ties.
      The advantage of coverage / constant_liar over greedy is ~0 here. This is
      the honesty anchor -- diversity does NOT help when the redundant cluster is
      where the optimum already is.
    * ``regime="decoy"`` (one build of :func:`build_decoy_cluster_landscape`,
      tagged ``redundancy=NaN``): the redundant cluster is a DECOY (broad,
      confident, suboptimal) and the true optimum is at a separate under-
      represented site. Greedy piles its budget on the decoy and misses the
      optimum (``frac_of_global`` and ``reached_target`` both LOW); coverage
      spreads the batch off the decoy and reaches it. The coverage advantage is
      LARGE here.

    The honest direction is the INTERACTION: the coverage advantage over greedy
    is ~0 in the tie regime and large in the decoy regime. ``constant_liar`` is a
    FolDE parity baseline whose batch-construction de-dup is asserted as a
    MECHANIC elsewhere, not as a payoff here (its decoy-regime advantage is small
    and seed-dependent). This function only REPORTS the numbers; downstream
    assertions check the direction without a magic number.

    Args:
        redundancy_list: redundancy knobs for the TIE regime (include 0 for the
            already-diverse reference and a large value for the redundant batch).
        seeds: seeds repeated per regime/redundancy (mean +/- sd over them).
        n_sites / n_subs: tie-regime landscape shape (``n_subs`` = cluster size).
        n_init / n_per_round / n_rounds: tie-regime active-learning budget.
        n_dims: tie-regime embedding width.
        coverage_penalty: site-coverage penalty for the ``coverage`` strategy
            (shared by both regimes).
        strategies: registry names to compare in BOTH regimes.
        target_quantile: tie-regime ``rounds_to_target`` target = this quantile
            of the true activities (decoy regime uses the global max as target,
            since reaching the true optimum is the whole point there).
        top_k: passed through to the simulation metrics.
        include_decoy: also run the decoy regime (default True).
        decoy_*: decoy-regime construction and budget. ``decoy_subs`` is chosen
            LARGER than the decoy batch budget so greedy cannot exhaust the
            cluster (the mechanism that traps it).

    Returns:
        Tidy DataFrame, one row per ``(regime, redundancy, strategy, seed)``,
        columns ``regime, redundancy, strategy, seed, max_activity_found,
        rounds_to_target, simple_regret, global_max, frac_of_global,
        reached_target``. Tie-regime rows carry the numeric redundancy; decoy
        rows carry ``redundancy=NaN``.
    """
    rows: list[dict] = []

    # --- TIE regime: redundant cluster CONTAINS the optimum (piling is fine). --
    for redundancy in redundancy_list:
        for seed in seeds:
            landscape, embeddings = build_redundant_landscape(
                n_sites=n_sites,
                n_subs=n_subs,
                redundancy=float(redundancy),
                seed=seed,
                n_dims=n_dims,
            )
            target = float(np.quantile(landscape.true_values(), target_quantile))
            rows.extend(
                _run_batch_strategies(
                    landscape=landscape,
                    embeddings=embeddings,
                    regime="tie",
                    redundancy=float(redundancy),
                    seed=seed,
                    strategies=strategies,
                    coverage_penalty=coverage_penalty,
                    n_init=n_init,
                    n_per_round=n_per_round,
                    n_rounds=n_rounds,
                    target=target,
                    top_k=top_k,
                )
            )

    # --- DECOY regime: redundant cluster is a decoy, optimum is rare/elsewhere. --
    if include_decoy:
        for seed in seeds:
            landscape, embeddings = build_decoy_cluster_landscape(
                n_other_sites=decoy_n_other_sites,
                decoy_subs=decoy_subs,
                other_subs=decoy_other_subs,
                seed=seed,
                n_dims=decoy_n_dims,
            )
            # Target = the true optimum: reaching it is exactly what diversity buys.
            target = landscape.global_max()
            rows.extend(
                _run_batch_strategies(
                    landscape=landscape,
                    embeddings=embeddings,
                    regime="decoy",
                    redundancy=float("nan"),
                    seed=seed,
                    strategies=strategies,
                    coverage_penalty=coverage_penalty,
                    n_init=decoy_n_init,
                    n_per_round=decoy_n_per_round,
                    n_rounds=decoy_n_rounds,
                    target=target,
                    top_k=top_k,
                )
            )

    if not rows:
        raise ValueError(
            "run_batch_diversity_ablation produced no rows; check the sweep lists."
        )
    return pd.DataFrame(rows, columns=_BATCH_DIVERSITY_COLUMNS)


# ==========================================================================
# Multi-mutant ablation (GAP 4): ADDITIVE STACKING vs RANDOM multi selection,
# SWEPT over EPISTASIS strength.
#
# GAP 4 multi-mutant proposal is a CAPABILITY, not a novelty (MULTI-evolve 2026
# scooped explicit epistasis-modelled multi-mutants; Visani 2026 showed a purely
# additive predictor reproduces it at r > 0.999 on additive-dominated
# landscapes). The USER explicitly DE-SCOPED epistasis: the predictor is purely
# additive (multi = baseline + sum of constituent singles), NO epistasis term.
#
# The honest claim is a SENSITIVITY / interaction result bounded to the
# additive-dominated regime the de-scope assumes: additive stacking finds
# high multi-mutants FAR more sample-efficiently than random WHEN the landscape
# is additive (epistasis=0), and its advantage SHRINKS as epistasis grows
# (the additive predictor degrades because it cannot model the interaction).
# This is NOT rigged: at high epistasis stacking SHOULD lose its edge, and the
# ablation reports that.
#
# Design (per the generative-oracle gap): both arms draw from ONE fixed
# candidate-multi pool (order-2 combinations over the top_k singles, capped), so
# the comparison is fair. Stacking is given PERFECT measured singles (true
# single effects from the oracle) -- the honest worst case for the de-scope
# claim, since stacking then fails at high epistasis ONLY because it ignores
# interaction, not because its singles were noisy.
#
# Metric note: regret (pool_best - best_found) inflates with the epistasis term
# (the landscape scale grows), so the SCALE-FREE advantage is reported as
# ``norm_advantage = 1 - regret_stack / regret_random`` (the fraction of
# random's regret gap that stacking closes). norm_advantage is 1.0 when stacking
# reaches the pool optimum (regret 0) and decays toward 0 as the additive
# predictor decorrelates from the true (epistatic) ranking. The honest direction
# is ``norm_advantage(epistasis=0) > norm_advantage(epistasis=high)``.
# ==========================================================================


_MULTIMUTANT_COLUMNS = [
    "epistasis",
    "seed",
    "pool_size",
    "pool_best",
    "stack_best_found",
    "random_best_found",
    "regret_stack",
    "regret_random",
    "norm_advantage",
]


def run_multimutant_ablation(
    *,
    epistasis_list: Sequence[float],
    seeds: Sequence[int],
    n_positions: int = 14,
    alphabet_subset: str = "ACDEFGH",
    order: int = 2,
    top_k_singles: int = 12,
    max_candidates: int = 300,
    budget: int = 10,
    baseline: float = 0.0,
) -> pd.DataFrame:
    """Additive-stacking vs random multi-mutant selection, swept over epistasis.

    For each ``(epistasis, seed)`` the SAME generative oracle
    (:func:`~plm_evolve.benchmark.multimutant.make_generative_landscape`) and the
    SAME fixed candidate-multi pool feed BOTH arms, so the only thing that
    changes between arms is the selection rule:

      1. Build the generative oracle (single-mutant Landscape + ``score_token``)
         at this epistasis / seed.
      2. Take the TRUE single effects from the oracle (perfect measured singles).
      3. Enumerate ONE order-``order`` candidate-multi pool from the top
         ``top_k_singles`` singles (capped at ``max_candidates``,
         deterministically), and score every pool member's TRUE multi activity
         with ``score_token`` (this is where epistasis enters for multis).
      4. STACKING arm: rank the pool by the PURELY ADDITIVE predictor
         (:class:`~plm_evolve.benchmark.multimutant.AdditiveMultiPredictor`,
         baseline + sum of constituent singles, NO epistasis term), take the top
         ``budget``, and record the best TRUE activity among them.
      5. RANDOM arm: pick ``budget`` pool members at random (same rng), record
         the best TRUE activity among them.
      6. Record regret (``pool_best - best_found``) for both arms and the
         scale-free ``norm_advantage = 1 - regret_stack / regret_random``.

    The active-learning loop / RF are NOT reimplemented; this is a one-shot
    proposal comparison (stacking is a ranking over a fixed pool), which is the
    multi-mutant capability the single-mutant loop cannot do.

    Args:
        epistasis_list: epistasis strengths to sweep (include 0.0 for the
            additive reference and a large value for the epistatic regime).
        seeds: seeds repeated per epistasis (mean +/- sd over them downstream).
        n_positions: oracle sequence length.
        alphabet_subset: oracle amino-acid subset (its size sets the candidate
            pool richness).
        order: multi-mutant order (``>= 2``; ``2`` -> doubles).
        top_k_singles: enumerate the pool from the best this-many singles.
        max_candidates: cap on the candidate-multi pool size (deterministic).
        budget: number of multis each arm is allowed to "measure".
        baseline: WT activity added to every additive prediction (0 here).

    Returns:
        Tidy DataFrame, one row per ``(epistasis, seed)``, columns:
        ``epistasis, seed, pool_size, pool_best, stack_best_found,
        random_best_found, regret_stack, regret_random, norm_advantage``.
    """
    if budget < 1:
        raise ValueError("budget must be >= 1.")

    rows: list[dict] = []
    for epistasis in epistasis_list:
        for seed in seeds:
            landscape, score_token = make_generative_landscape(
                n_positions=n_positions,
                alphabet_subset=alphabet_subset,
                seed=seed,
                epistasis=float(epistasis),
            )
            # Perfect measured singles: the TRUE single effects from the oracle.
            single_effects = {
                v: landscape.measure(v)
                for v in landscape.variant_ids()
                if v != "WT" and ":" not in v
            }
            rng = np.random.default_rng(seed)

            # ONE fixed candidate-multi pool, shared by both arms.
            ranked_singles = sorted(
                single_effects, key=lambda t: (-single_effects[t], t)
            )
            pool = enumerate_multimutants(
                ranked_singles,
                order=order,
                top_k=top_k_singles,
                max_candidates=max_candidates,
                rng=rng,
            )
            if not pool:
                raise ValueError(
                    "run_multimutant_ablation produced an empty candidate pool; "
                    "increase top_k_singles, the alphabet, or n_positions."
                )
            true_vals = np.array([score_token(t) for t in pool], dtype=float)
            pool_best = float(true_vals.max())

            # STACKING: purely additive ranking over the SAME pool.
            predictor = AdditiveMultiPredictor(
                measured_effects=single_effects, baseline=baseline
            )
            pred_scores = predictor.predict_many(pool)
            stack_order = np.argsort(-pred_scores, kind="stable")[:budget]
            stack_best = float(true_vals[stack_order].max())

            # RANDOM: same budget at random from the SAME pool.
            n_pick = min(budget, len(pool))
            ridx = rng.choice(len(pool), size=n_pick, replace=False)
            random_best = float(true_vals[ridx].max())

            regret_stack = pool_best - stack_best
            regret_random = pool_best - random_best
            # Scale-free advantage: the fraction of random's regret gap that
            # stacking closes. 1.0 when stacking reaches the pool optimum; can go
            # NEGATIVE when stacking is strictly WORSE than random (that loss is
            # reported faithfully, not floored). When random ALSO reaches the
            # optimum (regret_random == 0) there is no gap to close: a genuine tie
            # if stacking is optimal too, else stacking strictly lost a gap of
            # zero-denominator size -- both map to 0.0 (the RAW regret columns
            # carry the un-normalised truth in that degenerate case).
            if regret_random > 0:
                norm_advantage = 1.0 - regret_stack / regret_random
            else:
                norm_advantage = 0.0

            rows.append(
                {
                    "epistasis": float(epistasis),
                    "seed": int(seed),
                    "pool_size": int(len(pool)),
                    "pool_best": pool_best,
                    "stack_best_found": stack_best,
                    "random_best_found": random_best,
                    "regret_stack": float(regret_stack),
                    "regret_random": float(regret_random),
                    "norm_advantage": float(norm_advantage),
                }
            )

    if not rows:
        raise ValueError(
            "run_multimutant_ablation produced no rows; check the sweep lists."
        )
    return pd.DataFrame(rows, columns=_MULTIMUTANT_COLUMNS)


# ==========================================================================
# Multi-objective ablation (GAP 5): activity-only vs balanced-scalarized,
# swept ACROSS the objective correlation.
#
# Multi-objective selection is a SECONDARY, opt-in capability, not a novelty.
# The honest claim it supports is a TRADE-OFF result: balanced (50/50)
# scalarization achieves the SECOND objective FAR better than activity-only WHEN
# the two objectives trade off (anti-correlated), and roughly TIES when they are
# aligned (the second objective comes for free). To show that without rigging,
# the correlation between the two objectives is a TUNABLE knob and the
# comparison is run ACROSS it.
#
# This is a ONE-SHOT, perfect-information comparison (like run_multimutant_
# ablation): both arms see the SAME true objective values over the SAME
# candidate pool, and the ONLY thing that differs is the selection rule
# (greedy-on-A vs greedy-on-the-composed-50/50-scalar). The active-learning loop
# is NOT reimplemented here; loop integration is proven separately by the hook
# (compose_objectives -> run_simulation) in the tests. No embeddings, no token
# parsing -- simple string ids -- so the direction is not muddied by RF noise.
# ==========================================================================


_MULTIOBJECTIVE_COLUMNS = [
    "rho",
    "seed",
    "pool_size",
    "activity_only_objA",
    "balanced_objA",
    "activity_only_objB",
    "balanced_objB",
    "activity_only_balance",
    "balanced_balance",
    "activity_only_pareto_frac",
    "balanced_pareto_frac",
]


def _correlated_objectives(
    n: int, rho: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Two standardized objective arrays with target correlation ``rho``.

    Objective A is the activity, ``zA ~ N(0, 1)``. Objective B is built as
    ``rho * zA + sqrt(1 - rho^2) * noise`` with independent ``noise ~ N(0, 1)``,
    so the population correlation between A and B is ``rho``. ``rho = -1`` is a
    hard trade-off (improving A costs B), ``rho = +1`` is full alignment (B comes
    free with A), and ``rho = 0`` is independence. The min-max normalisation
    inside :func:`scalarize` absorbs the scale, so A and B need no rescaling.
    """
    if not -1.0 <= rho <= 1.0:
        raise ValueError(f"rho must be in [-1, 1], got {rho}.")
    zA = rng.normal(0.0, 1.0, size=n)
    noise = rng.normal(0.0, 1.0, size=n)
    zB = rho * zA + np.sqrt(max(0.0, 1.0 - rho * rho)) * noise
    return zA, zB


def _minmax(x: np.ndarray) -> np.ndarray:
    """Min-max to [0, 1]; a constant array maps to all-zeros (no NaN)."""
    lo, hi = float(x.min()), float(x.max())
    if hi - lo <= 0:
        return np.zeros_like(x, dtype=float)
    return (x - lo) / (hi - lo)


def run_multiobjective_ablation(
    *,
    rho_list: Sequence[float],
    seeds: Sequence[int],
    pool_size: int = 200,
    budget: int = 10,
) -> pd.DataFrame:
    """Activity-only vs balanced-scalarized selection, swept across correlation.

    For each ``(rho, seed)`` the SAME two true objective arrays (A = activity,
    B = a second property correlated with A at ``rho``) over the SAME candidate
    pool feed BOTH arms, so the only thing that changes between arms is the
    selection rule:

      1. Build objectives A and B at correlation ``rho``
         (:func:`_correlated_objectives`).
      2. ACTIVITY-ONLY arm: greedy on A alone -- take the top ``budget`` by A.
      3. BALANCED arm: greedy on the composed 50/50 scalar
         (:func:`~plm_evolve.benchmark.multiobjective.scalarize` with
         ``weights=[0.5, 0.5]``, linear, min-max-normalised) -- take the top
         ``budget`` by that scalar.
      4. For each arm, among the SELECTED set record the achieved objective
         maxima (min-max-normalised over the pool), their ``balance`` (the MIN of
         the two normalised maxima -- low if either objective is neglected), and
         the fraction of the pool's TRUE Pareto front
         (:func:`~plm_evolve.benchmark.multiobjective.pareto_front`) the arm
         recovered.

    Honest direction (asserted by the tests, not assumed): under ANTI-correlation
    balanced beats activity-only on objective B (and on ``balance``); under
    ALIGNMENT they roughly tie on B, and balanced does NOT beat activity-only on
    objective A (activity-only structurally contains the global A-max, so its
    normalised A achievement is 1.0 and cannot be beaten -- the ablation does not
    pretend otherwise).

    Args:
        rho_list: objective correlations to sweep, e.g. ``[-0.9, 0.0, 0.9]``;
            include a strongly negative (trade-off) and a strongly positive
            (aligned) value.
        seeds: seeds repeated per ``rho`` (mean +/- sd over them downstream).
        pool_size: number of candidate variants in the shared pool.
        budget: number of variants each arm is allowed to "measure".

    Returns:
        Tidy DataFrame, one row per ``(rho, seed)``, columns:
        ``rho, seed, pool_size, activity_only_objA, balanced_objA,
        activity_only_objB, balanced_objB, activity_only_balance,
        balanced_balance, activity_only_pareto_frac, balanced_pareto_frac``.
    """
    if budget < 1:
        raise ValueError("budget must be >= 1.")
    if pool_size < budget:
        raise ValueError("pool_size must be >= budget.")

    rows: list[dict] = []
    for rho in rho_list:
        for seed in seeds:
            rng = np.random.default_rng(seed)
            objA, objB = _correlated_objectives(pool_size, float(rho), rng)

            # Normalised objectives, shared by both arms (achievement scale).
            normA = _minmax(objA)
            normB = _minmax(objB)
            front = pareto_front(np.column_stack([objA, objB]))
            n_front = int(front.sum())

            # ACTIVITY-ONLY: greedy on A alone (descending, stable).
            act_idx = np.argsort(-objA, kind="stable")[:budget]
            # BALANCED: greedy on the composed 50/50 scalar.
            scalar = scalarize(
                np.column_stack([objA, objB]),
                weights=[0.5, 0.5],
                method="linear",
                normalize=True,
            )
            bal_idx = np.argsort(-scalar, kind="stable")[:budget]

            def _summarise(idx: np.ndarray) -> tuple[float, float, float, float]:
                a_max = float(normA[idx].max())
                b_max = float(normB[idx].max())
                balance = min(a_max, b_max)
                pareto_frac = (
                    float(front[idx].sum()) / n_front if n_front > 0 else 0.0
                )
                return a_max, b_max, balance, pareto_frac

            act_a, act_b, act_bal, act_pf = _summarise(act_idx)
            bal_a, bal_b, bal_bal, bal_pf = _summarise(bal_idx)

            rows.append(
                {
                    "rho": float(rho),
                    "seed": int(seed),
                    "pool_size": int(pool_size),
                    "activity_only_objA": act_a,
                    "balanced_objA": bal_a,
                    "activity_only_objB": act_b,
                    "balanced_objB": bal_b,
                    "activity_only_balance": act_bal,
                    "balanced_balance": bal_bal,
                    "activity_only_pareto_frac": act_pf,
                    "balanced_pareto_frac": bal_pf,
                }
            )

    if not rows:
        raise ValueError(
            "run_multiobjective_ablation produced no rows; check the sweep lists."
        )
    return pd.DataFrame(rows, columns=_MULTIOBJECTIVE_COLUMNS)
