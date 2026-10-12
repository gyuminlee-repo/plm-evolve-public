"""The in-silico simulation harness: sim == real, oracle replaces wet-lab.

Every round runs the SAME active-learning logic as the real wet-lab loop
(:func:`plm_evolve.evolve.run_round`): a strategy trains on the measured set,
predicts/ranks the unmeasured candidates, and proposes a batch. The only
difference is that the proposed batch is "measured" by the oracle
(:class:`~plm_evolve.benchmark.oracle.Landscape`) instead of by an experiment.

The harness exists to measure SAMPLE EFFICIENCY: how few measurements does a
strategy need to reach a target. :func:`single_round_simulation` provides the
non-iterative baseline (spend the whole budget after ONE model fit) so that
"iterative beats single-round" is directly measurable.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .metrics import compute_round_metrics, rounds_to_target
from .oracle import Landscape
from .strategies import GreedyTopLayer, RandomStrategy


def _eligible_candidates(landscape: Landscape, embeddings: pd.DataFrame) -> List[str]:
    """Variants present in BOTH the landscape and the embedding table.

    A strategy can only score variants it has embeddings for, and the oracle
    can only measure variants it contains. The candidate pool is the
    intersection (fail-fast if empty). Multi-mutant oracle entries with no
    embedding row are naturally excluded here.
    """
    emb_ids = set(embeddings.index)
    eligible = [v for v in landscape.variant_ids() if v in emb_ids]
    if not eligible:
        raise ValueError(
            "No variant is present in both the landscape and the embeddings; "
            "check that variant ids use the same token convention."
        )
    return eligible


def _seed_initial(
    pool: Sequence[str], n_init: int, rng: np.random.Generator
) -> List[str]:
    """Pick the initial random measured set (without replacement)."""
    pool = list(pool)
    if n_init > len(pool):
        raise ValueError(
            f"n_init={n_init} exceeds the candidate pool size {len(pool)}."
        )
    idx = rng.choice(len(pool), size=n_init, replace=False)
    return [pool[i] for i in sorted(idx.tolist())]


def run_simulation(
    landscape: Landscape,
    embeddings: pd.DataFrame,
    strategy,
    n_init: int,
    n_per_round: int,
    n_rounds: int,
    seed: int,
    *,
    top_k: int = 10,
    target: Optional[float] = None,
) -> pd.DataFrame:
    """Run an iterative active-learning simulation, one strategy, one seed.

    Loop:
      1. Seed-initialise ``n_init`` random measured variants (measured by oracle).
      2. Each round: strategy picks ``n_per_round`` from the unmeasured pool,
         the oracle measures them, append to the cumulative measured set.
      3. Recompute metrics on the cumulative measured set vs the full landscape.

    Returns:
        A per-round metrics DataFrame with columns ``round`` (0 == seed,
        1.. == AL rounds), ``n_measured``, ``rounds_to_target`` (cumulative,
        NaN until reached), and all per-round metrics.
    """
    rng = np.random.default_rng(seed)
    pool = _eligible_candidates(landscape, embeddings)

    measured: List[str] = _seed_initial(pool, n_init, rng)
    measured_set = set(measured)
    measured_y: List[float] = [landscape.measure(v) for v in measured]

    rows: List[Dict[str, float]] = []
    best_trajectory: List[float] = []

    def record(round_idx: int) -> None:
        m = compute_round_metrics(
            landscape, measured, top_k=top_k, target=target
        )
        best_trajectory.append(m["max_activity_found"])
        row: Dict[str, float] = {"round": round_idx, "n_measured": len(measured)}
        row.update(m)
        if target is not None:
            row["rounds_to_target"] = rounds_to_target(best_trajectory, target)
        rows.append(row)

    # Round 0: the seed set.
    record(0)

    for r in range(1, n_rounds + 1):
        candidates = [v for v in pool if v not in measured_set]
        if not candidates:
            break
        selected = strategy.pick(
            measured, measured_y, candidates, embeddings, n_per_round, rng
        )
        for v in selected:
            measured.append(v)
            measured_set.add(v)
            measured_y.append(landscape.measure(v))
        record(r)

    df = pd.DataFrame(rows)
    df.insert(0, "strategy", getattr(strategy, "name", strategy.__class__.__name__))
    df.insert(1, "seed", seed)
    return df


def single_round_simulation(
    landscape: Landscape,
    embeddings: pd.DataFrame,
    n_total: int,
    seed: int,
    ranker: str = "greedy",
    *,
    n_init: Optional[int] = None,
    top_k: int = 10,
    target: Optional[float] = None,
) -> pd.DataFrame:
    """The non-iterative baseline: spend the whole budget after ONE model fit.

    Measure ``n_init`` random variants, fit ONCE, then pick the remaining
    ``n_total - n_init`` in a single batch (no iteration / no re-fit). This is
    the MULTI-evolve-style single-round control the iterative loop is compared
    against on the same total budget.

    Args:
        n_total: total measurement budget (seed + single batch).
        ranker: ``'greedy'`` (fit top-layer once, pick top batch) or
            ``'random'`` (the whole budget spent at random).
        n_init: size of the initial random fit set (default ``n_total // 2``,
            at least 1, leaving at least one for the batch).

    Returns:
        A two-row DataFrame (round 0 = seed, round 1 = the single batch) with
        the same metric columns as :func:`run_simulation`, tagged
        ``strategy='single_round_<ranker>'``.
    """
    rng = np.random.default_rng(seed)
    pool = _eligible_candidates(landscape, embeddings)
    if n_total > len(pool):
        raise ValueError(
            f"n_total={n_total} exceeds the candidate pool size {len(pool)}."
        )
    if n_init is None:
        n_init = max(1, min(n_total - 1, n_total // 2))
    if not (1 <= n_init < n_total):
        raise ValueError("single_round needs 1 <= n_init < n_total.")

    measured: List[str] = _seed_initial(pool, n_init, rng)
    measured_set = set(measured)
    measured_y: List[float] = [landscape.measure(v) for v in measured]

    rows: List[Dict[str, float]] = []
    best_trajectory: List[float] = []

    def record(round_idx: int) -> None:
        m = compute_round_metrics(
            landscape, measured, top_k=top_k, target=target
        )
        best_trajectory.append(m["max_activity_found"])
        row: Dict[str, float] = {"round": round_idx, "n_measured": len(measured)}
        row.update(m)
        if target is not None:
            row["rounds_to_target"] = rounds_to_target(best_trajectory, target)
        rows.append(row)

    record(0)  # seed

    n_batch = n_total - n_init
    candidates = [v for v in pool if v not in measured_set]
    if ranker == "random":
        strat = RandomStrategy()
    elif ranker == "greedy":
        strat = GreedyTopLayer()
    else:
        raise ValueError(f"ranker must be 'greedy' or 'random', got {ranker!r}.")

    # ONE fit, ONE pick of the whole remaining budget.
    selected = strat.pick(
        measured, measured_y, candidates, embeddings, n_batch, rng
    )
    for v in selected:
        measured.append(v)
        measured_set.add(v)
        measured_y.append(landscape.measure(v))
    record(1)

    df = pd.DataFrame(rows)
    df.insert(0, "strategy", f"single_round_{ranker}")
    df.insert(1, "seed", seed)
    return df


def run_panel(
    landscape: Landscape,
    embeddings: pd.DataFrame,
    strategies: Sequence,
    seeds: Sequence[int],
    n_init: int,
    n_per_round: int,
    n_rounds: int,
    *,
    top_k: int = 10,
    target: Optional[float] = None,
    single_round_rankers: Sequence[str] = (),
) -> pd.DataFrame:
    """Run a panel across strategies x seeds x rounds -> one tidy DataFrame.

    Args:
        strategies: iterable of strategy instances for the iterative loop.
        seeds: seeds to repeat each strategy over (for mean +/- sd downstream).
        single_round_rankers: optional rankers (``'greedy'``/``'random'``) run
            as single-round baselines on the same total budget
            ``n_init + n_per_round * n_rounds``.

    Returns:
        Long/tidy DataFrame: one row per (strategy, seed, round).
    """
    frames: List[pd.DataFrame] = []
    for strategy in strategies:
        for seed in seeds:
            frames.append(
                run_simulation(
                    landscape, embeddings, strategy, n_init, n_per_round,
                    n_rounds, seed, top_k=top_k, target=target,
                )
            )

    n_total = n_init + n_per_round * n_rounds
    for ranker in single_round_rankers:
        for seed in seeds:
            frames.append(
                single_round_simulation(
                    landscape, embeddings, n_total, seed, ranker=ranker,
                    n_init=n_init, top_k=top_k, target=target,
                )
            )

    if not frames:
        raise ValueError("run_panel produced no runs; check strategies/seeds.")
    return pd.concat(frames, ignore_index=True)
