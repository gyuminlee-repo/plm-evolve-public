#!/usr/bin/env python3
"""Benchmark acquisition strategies in the in-silico active-learning simulator.

Compares greedy, ucb, thompson, and coverage strategies on a real DMS fitness
landscape, reporting mean best activity found, simple regret, and rounds to
reach a target quantile -- averaged over multiple seeds.

Usage:
    python3 scripts/sim_acq_compare.py --dataset kelsic --model esmc_300m
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path bootstrap so plm_evolve and scripts/ both resolve regardless of cwd.
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "scripts"))
sys.path.insert(0, str(_ROOT))

import figure_lib  # noqa: E402 -- scripts/ is on path above
from plm_evolve.benchmark.oracle import Landscape  # noqa: E402
from plm_evolve.benchmark.simulate import run_panel  # noqa: E402
from plm_evolve.benchmark.strategies import build_strategy  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare acquisition strategies in the AL simulator."
    )
    p.add_argument("--dataset", default="kelsic")
    p.add_argument("--model", default="esmc_300m")
    p.add_argument("--n-init", type=int, default=16, dest="n_init")
    p.add_argument("--per-round", type=int, default=12, dest="per_round")
    p.add_argument("--rounds", type=int, default=8)
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--target-quantile", type=float, default=0.95, dest="target_quantile")
    p.add_argument("--out", default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    dataset = args.dataset
    model = args.model
    out_dir = Path(
        args.out if args.out is not None
        else f"outputs/acq_compare/{dataset}_{model}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load embeddings and oracle; fail-fast if missing.
    # ------------------------------------------------------------------
    if dataset not in figure_lib.DATASET_HASH:
        raise ValueError(
            f"Dataset {dataset!r} not in DATASET_HASH; "
            f"available: {sorted(figure_lib.DATASET_HASH)}"
        )
    hash_ = figure_lib.DATASET_HASH[dataset]
    emb_path = figure_lib.cache_path(dataset, model, hash_, pooling="mean")
    if not emb_path.exists():
        raise FileNotFoundError(
            f"Embedding cache not found: {emb_path}\n"
            "Run the embed step first or check --dataset/--model."
        )

    emb_df = figure_lib.from_csv(emb_path)
    oracle = figure_lib.load_oracle(dataset)

    # ------------------------------------------------------------------
    # 2. Intersect ids (exclude WT); fail-fast if empty.
    # ------------------------------------------------------------------
    emb_ids = set(emb_df.index)
    oracle_ids = set(oracle.keys())
    common_ids = sorted(
        v for v in (emb_ids & oracle_ids) if v != "WT"
    )
    if not common_ids:
        raise ValueError(
            f"Intersection of embeddings and oracle is empty for "
            f"dataset={dataset!r} model={model!r}. "
            "Check that variant-id conventions match."
        )

    activities = [float(oracle[v]) for v in common_ids]

    # ------------------------------------------------------------------
    # 3. Write temp landscape CSV from the intersected set; load it.
    # ------------------------------------------------------------------
    landscape_csv = out_dir / "_landscape_tmp.csv"
    pd.DataFrame(
        {"variant": common_ids, "activity": activities}
    ).to_csv(landscape_csv, index=False)
    landscape = Landscape.from_csv(landscape_csv)

    # Subset embeddings to the same id set (same order as common_ids).
    emb_subset = emb_df.loc[common_ids]

    # ------------------------------------------------------------------
    # 4. Target at the requested quantile of the intersected activities.
    # ------------------------------------------------------------------
    target = float(np.quantile(activities, args.target_quantile))

    # ------------------------------------------------------------------
    # 5. Run the panel.
    # ------------------------------------------------------------------
    strategy_names = ["greedy", "ucb", "thompson", "coverage"]
    strategies = [build_strategy(n) for n in strategy_names]
    seeds = list(range(args.seeds))

    panel = run_panel(
        landscape=landscape,
        embeddings=emb_subset,
        strategies=strategies,
        seeds=seeds,
        n_init=args.n_init,
        n_per_round=args.per_round,
        n_rounds=args.rounds,
        target=target,
        single_round_rankers=[],
    )

    # ------------------------------------------------------------------
    # 6. Write full tidy panel.
    # ------------------------------------------------------------------
    panel_path = out_dir / "panel.csv"
    panel.to_csv(panel_path, index=False)

    # ------------------------------------------------------------------
    # 7. Aggregate exactly like _cmd_simulate (cli.py lines 103-121).
    # ------------------------------------------------------------------
    final = (
        panel.sort_values("round")
        .groupby(["strategy", "seed"], as_index=False)
        .tail(1)
    )

    # Build per-strategy summary dict.
    summary: dict[str, dict] = {}
    for name, grp in final.groupby("strategy"):
        mean_best = float(np.nanmean(grp["max_activity_found"]))
        mean_regret = float(np.nanmean(grp["simple_regret"]))
        if "rounds_to_target" in grp.columns:
            mean_rtt = float(np.nanmean(grp["rounds_to_target"]))
        else:
            mean_rtt = float("nan")
        summary[name] = {
            "mean_best": mean_best,
            "mean_regret": mean_regret,
            "mean_rounds_to_target": mean_rtt,
        }

    # ------------------------------------------------------------------
    # 8. Print context and comparison table.
    # ------------------------------------------------------------------
    global_max = landscape.global_max()
    print(f"landscape global_max={global_max:.4f}  target(q={args.target_quantile})={target:.4f}")
    print(f"n_variants={len(common_ids)}  n_init={args.n_init}  "
          f"per_round={args.per_round}  rounds={args.rounds}  seeds={args.seeds}")
    print()

    hdr = (f"{'strategy':<12}  {'mean_best':>9}  {'mean_regret':>11}  "
           f"{'mean_rtt':>8}  {'delta_best':>10}  {'delta_rtt':>9}")
    print(hdr)
    print("-" * len(hdr))

    greedy = summary.get("greedy", {})
    for name in strategy_names:
        s = summary.get(name, {})
        mean_best = s.get("mean_best", float("nan"))
        mean_regret = s.get("mean_regret", float("nan"))
        mean_rtt = s.get("mean_rounds_to_target", float("nan"))

        delta_best = mean_best - greedy.get("mean_best", float("nan"))
        delta_rtt = mean_rtt - greedy.get("mean_rounds_to_target", float("nan"))

        if name == "greedy":
            delta_best_str = "baseline"
            delta_rtt_str = "baseline"
        else:
            delta_best_str = f"{delta_best:+.4f}"
            delta_rtt_str = f"{delta_rtt:+.4f}" if not np.isnan(delta_rtt) else "nan"

        print(f"{name:<12}  {mean_best:>9.4f}  {mean_regret:>11.4f}  "
              f"{mean_rtt:>8.4f}  {delta_best_str:>10}  {delta_rtt_str:>9}")

    print()
    print(f"Wrote tidy panel: {panel_path}  shape={panel.shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
