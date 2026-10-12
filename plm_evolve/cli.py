"""Command-line interface for plm-evolve.

Three subcommands:

``embed``     generate all single mutants from a WT FASTA, embed them with a
              PLM, and write an embedding CSV.
``rank``      train the top-layer regressor on measured rounds and propose the
              next-round candidates.
``simulate``  run the in-silico active-learning panel against an oracle
              landscape and report sample-efficiency metrics per strategy.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

from . import __version__


def _cmd_embed(args: argparse.Namespace) -> int:
    from .embeddings import ESM2Extractor, to_csv
    from .variants import generate_single_mutant_fasta

    out_path = Path(args.out)

    # Write the single-mutant FASTA next to the output, then embed it.
    with tempfile.TemporaryDirectory() as tmp:
        mutant_fasta = Path(tmp) / "single_mutants.fasta"
        n = generate_single_mutant_fasta(args.wt, mutant_fasta)
        print(f"Generated {n} sequences (19*L + WT) -> embedding with {args.model}")

        extractor = ESM2Extractor(model_name=args.model, pooling=args.pooling)
        df = extractor.embed_fasta(mutant_fasta)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    to_csv(df, out_path)
    print(f"Wrote embeddings: {out_path}  shape={df.shape}")
    return 0


def _cmd_rank(args: argparse.Namespace) -> int:
    from .embeddings import from_csv
    from .evolve import run_round

    embeddings_df = from_csv(args.embeddings)
    result = run_round(
        embeddings_df=embeddings_df,
        round_files=args.rounds,
        wt_fasta=args.wt,
        top_n=args.top,
        out_dir=args.out,
        acquisition=args.acquisition,
        beta=args.beta,
        penalty=args.penalty,
        seed=args.seed,
    )

    print(f"Trained on {result.n_train} measured variant(s).")
    print(f"\nTop {len(result.next_round)} candidates for the next round:")
    for _, row in result.next_round.iterrows():
        print(f"  {row['variant']:<12} predicted={row['predicted']:.4f}")

    if args.out:
        print(f"\nWrote ranked.csv and next_round.csv to {args.out}")
    return 0


def _cmd_simulate(args: argparse.Namespace) -> int:
    import numpy as np

    from .embeddings import from_csv
    from .benchmark.oracle import Landscape
    from .benchmark.simulate import run_panel
    from .benchmark.strategies import build_strategy

    embeddings_df = from_csv(args.embeddings)
    landscape = Landscape.from_csv(args.landscape)

    # Split requested strategies into iterative ones and the single-round
    # baseline ('single_round' -> greedy single-shot on the same total budget).
    iterative_names = [s for s in args.strategies if s != "single_round"]
    single_rankers = ["greedy"] if "single_round" in args.strategies else []
    strategies = [build_strategy(name) for name in iterative_names]
    seeds = list(range(args.seeds))

    panel = run_panel(
        landscape=landscape,
        embeddings=embeddings_df,
        strategies=strategies,
        seeds=seeds,
        n_init=args.n_init,
        n_per_round=args.per_round,
        n_rounds=args.rounds,
        target=args.target,
        single_round_rankers=single_rankers,
    )

    # Per-strategy summary at the FINAL round of each (strategy, seed) run.
    final = (
        panel.sort_values("round")
        .groupby(["strategy", "seed"], as_index=False)
        .tail(1)
    )
    print(f"Landscape: {len(landscape)} variants, "
          f"global_max={landscape.global_max():.4f}")
    print(f"Budget: n_init={args.n_init} per_round={args.per_round} "
          f"rounds={args.rounds} seeds={args.seeds}"
          + (f" target={args.target}" if args.target is not None else ""))
    print("\nstrategy                 mean_best   mean_regret"
          + ("   mean_rounds_to_target" if args.target is not None else ""))
    for name, grp in final.groupby("strategy"):
        mean_best = float(np.nanmean(grp["max_activity_found"]))
        mean_regret = float(np.nanmean(grp["simple_regret"]))
        line = f"{name:<24} {mean_best:>9.4f}   {mean_regret:>11.4f}"
        if args.target is not None and "rounds_to_target" in grp:
            line += f"   {float(np.nanmean(grp['rounds_to_target'])):>21.4f}"
        print(line)

    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        panel_path = out_dir / "panel.csv"
        panel.to_csv(panel_path, index=False)
        print(f"\nWrote tidy panel: {panel_path}  shape={panel.shape}")
    return 0


def _cmd_propose_multi(args: argparse.Namespace) -> int:
    import numpy as np
    import pandas as pd

    from .embeddings import from_csv
    from .regressor import TopLayerRegressor
    from .rounds import build_training_table, load_round_data
    from .variants import WT_TOKEN
    from .benchmark.multimutant import AdditiveMultiPredictor, propose_multimutants

    embeddings_df = from_csv(args.embeddings)
    measured = load_round_data(args.rounds, args.wt)
    table = build_training_table(measured, embeddings_df.index.tolist())

    train_mask = table["measured"].to_numpy()
    if train_mask.sum() == 0:
        raise ValueError(
            "No measured variant matched the embedding index; nothing to fit "
            "the single-mutant model on."
        )

    # Fit the single-mutant model on the MEASURED singles' embeddings -> activity.
    train_variants = table.loc[train_mask, "variant"]
    X_train = embeddings_df.loc[train_variants].to_numpy()
    y_train = table.loc[train_mask, "activity"].to_numpy()
    model = TopLayerRegressor()
    model.fit(X_train, y_train)

    # Single-effect source, MEASURED-first then model-predicted (never silent 0).
    # WT is excluded from the stacking source (parse_variant('WT') is invalid);
    # WT activity feeds the additive baseline instead (rank-invariant).
    measured_map = dict(zip(table.loc[train_mask, "variant"], y_train))
    baseline = float(measured_map.get(WT_TOKEN, 0.0))

    single_effects: dict[str, float] = {}
    unmeasured_singles: List[str] = []
    for vid in embeddings_df.index:
        if vid == WT_TOKEN or ":" in str(vid):
            continue  # skip WT and any multi tokens present in the table.
        if vid in measured_map:
            single_effects[vid] = float(measured_map[vid])
        else:
            unmeasured_singles.append(vid)

    if unmeasured_singles:
        X_unmeasured = embeddings_df.loc[unmeasured_singles].to_numpy()
        predicted = model.predict(X_unmeasured)
        for vid, pred in zip(unmeasured_singles, predicted):
            single_effects[vid] = float(pred)

    if not single_effects:
        raise ValueError("No single-mutant candidates available to stack.")

    tokens = propose_multimutants(
        single_effects,
        order=args.order,
        n_propose=args.top,
        top_k=args.top_singles,
        baseline=baseline,
        rng=np.random.default_rng(args.seed),
    )
    if not tokens:
        raise ValueError(
            f"No order-{args.order} multi-mutants could be enumerated from the "
            f"available singles (need at least {args.order} distinct positions)."
        )

    predictor = AdditiveMultiPredictor(
        measured_effects=dict(single_effects), baseline=baseline
    )
    scores = predictor.predict_many(tokens)
    proposals = pd.DataFrame({"variant": tokens, "additive_predicted": scores})

    print(
        f"Fitted single-mutant model on {int(train_mask.sum())} measured "
        f"variant(s); {len(single_effects)} candidate single(s) "
        f"(baseline={baseline:.4f})."
    )
    print(f"\nTop {min(args.top, len(proposals))} order-{args.order} multi-mutants:")
    for _, row in proposals.head(args.top).iterrows():
        print(f"  {row['variant']:<20} additive_predicted={row['additive_predicted']:.4f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    proposals.to_csv(out_path, index=False)
    print(f"\nWrote proposals: {out_path}  shape={proposals.shape}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="plm-evolve",
        description=(
            "Frozen PLM embeddings + top-layer RandomForest active-learning "
            "loop for experimental directed evolution (EVOLVEpro core)."
        ),
    )
    parser.add_argument("--version", action="version", version=f"plm-evolve {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_embed = sub.add_parser("embed", help="generate single mutants and embed them")
    p_embed.add_argument("--wt", required=True, help="wild-type FASTA path")
    p_embed.add_argument(
        "--model",
        default="facebook/esm2_t33_650M_UR50D",
        help="HuggingFace ESM-2 model id",
    )
    p_embed.add_argument("--out", default="embeddings.csv", help="output CSV path")
    p_embed.add_argument(
        "--pooling",
        choices=["mean", "delta", "mutated_token"],
        default="mean",
        help=(
            "embedding pooling mode (OPT-IN; default 'mean' = original "
            "behavior). 'delta' = variant mean minus WT mean (canary, "
            "equivalent to 'mean' under a RandomForest top-layer). "
            "'mutated_token' = hidden state at the mutated position (GAP-2)."
        ),
    )
    p_embed.set_defaults(func=_cmd_embed)

    p_rank = sub.add_parser("rank", help="rank candidates and propose next round")
    p_rank.add_argument("--embeddings", required=True, help="embedding CSV path")
    p_rank.add_argument("--wt", required=True, help="wild-type FASTA path")
    p_rank.add_argument(
        "--rounds", required=True, nargs="+", help="measured-round files (xlsx/csv)"
    )
    p_rank.add_argument("--top", type=int, default=12, help="next-round candidate count")
    p_rank.add_argument("--out", default=None, help="output directory")
    p_rank.add_argument(
        "--acquisition",
        choices=["greedy", "ucb", "thompson", "coverage"],
        default="greedy",
        help=(
            "next-round selection (OPT-IN; default 'greedy' = original "
            "behaviour). 'ucb'/'thompson'/'coverage' select via the benchmark "
            "strategy; the ranked table is built the same way in every case."
        ),
    )
    p_rank.add_argument(
        "--beta", type=float, default=1.0, help="exploration weight (ucb/coverage)"
    )
    p_rank.add_argument(
        "--penalty",
        type=float,
        default=1.0,
        help="mutation-site coverage penalty (coverage only)",
    )
    p_rank.add_argument(
        "--seed", type=int, default=0, help="RNG seed (thompson determinism)"
    )
    p_rank.set_defaults(func=_cmd_rank)

    p_sim = sub.add_parser(
        "simulate",
        help="run the in-silico active-learning panel against an oracle landscape",
    )
    p_sim.add_argument(
        "--landscape", required=True, help="oracle landscape CSV (variant,activity)"
    )
    p_sim.add_argument("--embeddings", required=True, help="embedding CSV path")
    p_sim.add_argument(
        "--strategies",
        nargs="+",
        default=["greedy", "random", "additive", "single_round"],
        help="strategies: greedy random additive single_round",
    )
    p_sim.add_argument("--n-init", type=int, default=16, help="initial random set size")
    p_sim.add_argument("--per-round", type=int, default=16, help="picks per round")
    p_sim.add_argument("--rounds", type=int, default=10, help="number of AL rounds")
    p_sim.add_argument("--seeds", type=int, default=10, help="number of seeds (0..N-1)")
    p_sim.add_argument(
        "--target", type=float, default=None, help="target activity for rounds-to-target"
    )
    p_sim.add_argument("--out", default=None, help="output directory for panel.csv")
    p_sim.set_defaults(func=_cmd_simulate)

    p_multi = sub.add_parser(
        "propose-multi",
        help="propose multi-mutants by additive stacking of the best singles (GAP-4)",
    )
    p_multi.add_argument("--embeddings", required=True, help="embedding CSV path")
    p_multi.add_argument("--wt", required=True, help="wild-type FASTA path")
    p_multi.add_argument(
        "--rounds", required=True, nargs="+", help="measured-round files (xlsx/csv)"
    )
    p_multi.add_argument(
        "--order", type=int, default=2, help="multi-mutant order (>=2; 2 = doubles)"
    )
    p_multi.add_argument(
        "--top-singles",
        type=int,
        default=None,
        dest="top_singles",
        help="restrict stacking to the best K singles (default: all)",
    )
    p_multi.add_argument(
        "--top", type=int, default=12, help="number of multi-mutants to propose"
    )
    p_multi.add_argument(
        "--seed", type=int, default=0, help="RNG seed (enumeration cap sampling)"
    )
    p_multi.add_argument(
        "--out", default="multimutant_proposals.csv", help="output CSV path"
    )
    p_multi.set_defaults(func=_cmd_propose_multi)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
