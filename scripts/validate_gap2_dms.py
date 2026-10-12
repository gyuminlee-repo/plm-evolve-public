#!/usr/bin/env python3
"""Real-DMS validation for GAP-2: greedy on 'mean' vs 'mutated_token' embeddings.

Answers the novelty question on REAL EVOLVEpro DMS oracle data: in the few-shot
active-learning loop, does greedy on ``mutated_token`` embeddings reach the top
variants in fewer measurements than greedy on ``mean`` embeddings, on the SAME
variants / same real ESM-2 model / same loop / same oracle? Mean-vs-mutated_token
is the ONLY difference between the two greedy arms. Floors (random, additive
one-hot ridge) bracket the result; the additive baseline may WIN if the DMS is
additive-dominated, which is an honest finding reported plainly.

The active-learning loop logic is NOT reimplemented here: it lives in
``plm_evolve.benchmark.simulate.run_simulation`` and is reused verbatim. This
script only (1) loads the real oracle, (2) builds the candidate pool, (3) computes
+ caches the two real ESM-2 embedding tables, (4) runs the shared loop across
strategies x seeds x k, and (5) aggregates a tidy CSV + summary.

ASSUMPTIONS (explicit, per karpathy-guidelines):
- Variant tokens follow the package convention ``<WTaa><pos><newaa>`` (1-based),
  validated by ``plm_evolve.variants.parse_variant``.
- The candidate pool is single mutants whose ``<WTaa><pos>`` matches the WT FASTA;
  WT itself is excluded from the landscape/top-k (targets are high-activity singles).
- ``target`` and top-k are derived from the ACTUAL run landscape (post-subsample),
  not the full dataset, so the target is always reachable within the pool.
- The MEAN and MUTATED_TOKEN tables must share an identical index set; this IS the
  "only difference is pooling" guarantee and is asserted (fail-fast) before running.
- The EVOLVEpro DMS oracle root is env-driven (``PLM_EVOLVE_DMS_ROOT`` or
  ``$WORKSPACE_ROOT/030.repos_clone/EvolvePro/data/dms``). Read-only; never modified.

No silent fallbacks: parse failures, WT mismatches, NaN activities and duplicate
tokens are counted and reported separately, never dropped-to-zero.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Reuse the package: loop, oracle, strategies, embeddings, variant parsing.
from plm_evolve.benchmark.oracle import Landscape
from plm_evolve.benchmark.simulate import run_simulation
from plm_evolve.benchmark.strategies import (
    AdditiveLinear,
    GreedyTopLayer,
    RandomStrategy,
)
from plm_evolve.embeddings import make_extractor, from_csv as emb_from_csv, to_csv as emb_to_csv
from plm_evolve.variants import WT_TOKEN, parse_variant, read_wt_sequence


# --- dataset presets carrying the verified inline ground-truth mapping --------

def _dms_root() -> Path:
    """Read-only EVOLVEpro DMS oracle root (env-driven; never modified).

    Resolution order (no hardcoded absolute paths, per multi-environment rule):
      1. ``PLM_EVOLVE_DMS_ROOT`` (points directly at .../data/dms), else
      2. ``$WORKSPACE_ROOT/030.repos_clone/EvolvePro/data/dms``.
    If neither env var is set the script stops with a message.
    """
    env = os.environ.get("PLM_EVOLVE_DMS_ROOT")
    if env:
        return Path(env)
    workspace = os.environ.get("WORKSPACE_ROOT")
    if not workspace:
        raise SystemExit("Set PLM_EVOLVE_DMS_ROOT or WORKSPACE_ROOT to locate the DMS data.")
    return Path(workspace) / "030.repos_clone" / "EvolvePro" / "data" / "dms"


_DMS_ROOT = _dms_root()


@dataclass(frozen=True)
class DatasetPreset:
    name: str
    wt_fasta: str
    activity_file: str
    sheet: str
    variant_col: str
    activity_col: str
    subsample_default: int  # 0 == use all single mutants


def _build_presets() -> Dict[str, DatasetPreset]:
    return {
        "kelsic": DatasetPreset(
            name="kelsic",
            wt_fasta=str(_DMS_ROOT / "wt_fasta" / "kelsic_WT.fasta"),
            activity_file=str(_DMS_ROOT / "activity" / "Source.xlsx"),
            sheet="infA",
            variant_col="variant",
            activity_col="DMS_rich",
            subsample_default=0,  # L=72, 1368 singles; full is cheap enough.
        ),
        "cas12f": DatasetPreset(
            name="cas12f",
            wt_fasta=str(_DMS_ROOT / "wt_fasta" / "cas12f_WT.fasta"),
            activity_file=str(_DMS_ROOT / "activity" / "DMS_AsCas12f_preprocessed.xlsx"),
            sheet="Sheet1",
            variant_col="variant",
            activity_col="avg_activity",
            subsample_default=350,  # L=422, ~7941 singles; subsample for CPU.
        ),
        "stiffler": DatasetPreset(
            name="stiffler",
            wt_fasta=str(_DMS_ROOT / "wt_fasta" / "stiffler_WT.fasta"),
            activity_file=str(_DMS_ROOT / "activity" / "Source.xlsx"),
            sheet="bla",
            variant_col="variant",
            activity_col="DMS_amp_2500_(b)",
            subsample_default=350,  # L=286, ~4978 singles; subsample for CPU.
        ),
        "brenan": DatasetPreset(
            name="brenan",
            wt_fasta=str(_DMS_ROOT / "wt_fasta" / "brenan_WT.fasta"),
            activity_file=str(_DMS_ROOT / "activity" / "Source.xlsx"),
            sheet="MAPK1",
            variant_col="variant",
            activity_col="DMS_SCH",
            subsample_default=350,  # L=360, ~6810 singles; subsample for CPU.
        ),
        "giacomelli": DatasetPreset(
            name="giacomelli",
            wt_fasta=str(_DMS_ROOT / "wt_fasta" / "giacomelli_WT.fasta"),
            activity_file=str(_DMS_ROOT / "activity" / "Source.xlsx"),
            sheet="P53",
            variant_col="variant",
            activity_col="DMS_null_etoposide",
            subsample_default=350,  # L=393, ~7448 singles; subsample for CPU.
        ),
    }


PRESETS: Dict[str, DatasetPreset] = _build_presets()


@dataclass
class DropReport:
    """Counts of variants dropped at each filter (never silent)."""

    total_rows: int = 0
    parse_fail: int = 0
    wt_mismatch: int = 0
    nan_activity: int = 0
    duplicate: int = 0
    wt_rows: int = 0
    kept: int = 0

    def summary(self) -> str:
        return (
            f"total_rows={self.total_rows} kept={self.kept} | dropped: "
            f"parse_fail={self.parse_fail} wt_mismatch={self.wt_mismatch} "
            f"nan_activity={self.nan_activity} duplicate={self.duplicate} "
            f"wt_rows={self.wt_rows}"
        )


# --- step 1: load the real oracle (single mutants matching the WT) ------------

def load_single_mutant_activities(
    activity_file: str,
    sheet: str,
    variant_col: str,
    activity_col: str,
    wt_sequence: str,
) -> Tuple[Dict[str, float], DropReport]:
    """variant_token -> activity for single mutants whose <WTaa><pos> match WT.

    Fail-fast loud reporting: every row that is dropped is counted in a
    DropReport (parse failure, WT-position mismatch, NaN activity, duplicate
    token, or a WT row). Nothing is silently coerced to 0.
    """
    df = pd.read_excel(activity_file, sheet_name=sheet)
    for col in (variant_col, activity_col):
        if col not in df.columns:
            raise ValueError(
                f"Column {col!r} not in {activity_file} sheet {sheet!r}; "
                f"available: {list(df.columns)}"
            )

    report = DropReport(total_rows=len(df))
    mapping: Dict[str, float] = {}

    for raw_tok, raw_act in zip(df[variant_col].astype(str), df[activity_col]):
        tok = raw_tok.strip()
        if tok in ("", "WT", WT_TOKEN, "nan"):
            report.wt_rows += 1
            continue
        try:
            mut = parse_variant(tok)
        except ValueError:
            report.parse_fail += 1
            continue
        # <WTaa><pos> must match the WT fasta (1-based). Do NOT silently drop-to-0.
        if mut.position > len(wt_sequence) or wt_sequence[mut.position - 1] != mut.wt_aa:
            report.wt_mismatch += 1
            continue
        try:
            act = float(raw_act)
        except (TypeError, ValueError):
            report.nan_activity += 1
            continue
        if np.isnan(act):
            report.nan_activity += 1
            continue
        if mut.token in mapping:
            report.duplicate += 1
            continue
        mapping[mut.token] = act

    report.kept = len(mapping)
    if not mapping:
        raise ValueError(
            f"No usable single-mutant rows in {activity_file} sheet {sheet!r}."
        )
    return mapping, report


# --- step 2: candidate pool (subsample, seed-fixed) ---------------------------

def subsample_candidates(
    tokens: List[str], max_variants: int, seed: int
) -> List[str]:
    """Seed-fixed subsample of single-mutant tokens (sorted, deterministic)."""
    tokens = sorted(tokens)
    if max_variants <= 0 or max_variants >= len(tokens):
        return tokens
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(tokens), size=max_variants, replace=False)
    return sorted(tokens[i] for i in idx)


# --- step 3: embeddings (real ESM-2), cached per pooling + variant-set hash ----

def _variant_set_hash(candidate_tokens: List[str]) -> str:
    """Stable short hash of the sorted candidate-id list that gets embedded."""
    h = hashlib.sha256()
    for tok in sorted(candidate_tokens):
        h.update(tok.encode())
        h.update(b"\n")
    return h.hexdigest()[:16]


def generate_mutant_fasta(
    wt_sequence: str, candidate_tokens: List[str], out_fasta: Path
) -> None:
    """Write WT + each candidate's full-length mutated sequence to FASTA."""
    from Bio.Seq import Seq
    from Bio.SeqIO import write as seqio_write
    from Bio.SeqRecord import SeqRecord

    records = [SeqRecord(Seq(wt_sequence), id=WT_TOKEN, description="")]
    for tok in candidate_tokens:
        mut = parse_variant(tok)
        i = mut.position - 1
        seq = wt_sequence[:i] + mut.new_aa + wt_sequence[i + 1 :]
        records.append(SeqRecord(Seq(seq), id=tok, description=""))
    out_fasta.parent.mkdir(parents=True, exist_ok=True)
    with open(out_fasta, "w") as handle:
        seqio_write(records, handle, "fasta")


def compute_or_load_embeddings(
    pooling: str,
    model: str,
    dataset_name: str,
    candidate_tokens: List[str],
    wt_sequence: str,
    cache_dir: Path,
) -> pd.DataFrame:
    """Real ESM-2 embeddings for {WT + candidates}, cached to disk; resumes.

    Cache key = dataset + model + pooling + hash(sorted candidate ids). On a
    cache hit the loaded index is asserted to equal the expected id set before
    use -- a stale cache from a different variant set is refused, not trusted.
    """
    expected_ids = set(candidate_tokens) | {WT_TOKEN}
    vhash = _variant_set_hash(candidate_tokens)
    model_tag = model.replace("/", "_")
    cache_path = cache_dir / f"emb_{dataset_name}_{model_tag}_{pooling}_{vhash}.csv"

    if cache_path.exists():
        df = emb_from_csv(cache_path)
        if set(df.index) != expected_ids:
            raise ValueError(
                f"Cache {cache_path} index does not match the expected variant "
                f"set (have {len(df.index)} ids, want {len(expected_ids)}); "
                f"refusing a stale cache. Delete it to recompute."
            )
        print(f"[cache hit] {pooling}: {cache_path.name} ({len(df)} rows)")
        return df

    print(f"[compute] {pooling}: embedding {len(expected_ids)} seqs with {model} ...")
    fasta_path = cache_dir / f"fasta_{dataset_name}_{vhash}.fasta"
    if not fasta_path.exists():
        generate_mutant_fasta(wt_sequence, candidate_tokens, fasta_path)
    extractor = make_extractor(model, pooling=pooling)
    df = extractor.embed_fasta(fasta_path)
    if set(df.index) != expected_ids:
        raise ValueError(
            f"{pooling} embeddings index mismatch after compute "
            f"(have {len(df.index)}, want {len(expected_ids)})."
        )
    cache_dir.mkdir(parents=True, exist_ok=True)
    emb_to_csv(df, cache_path)
    print(f"[cache write] {cache_path.name}")
    return df


# --- step 5: run the shared loop across strategies x seeds x k ----------------

def build_strategies() -> Dict[str, object]:
    """Strategy instances with EXPLICIT names so greedy_mean != greedy_mutated.

    GreedyTopLayer.name is 'greedy' regardless of the embedding table it runs on;
    set distinct names so run_simulation tags the output rows unambiguously.
    """
    g_mean = GreedyTopLayer()
    g_mean.name = "greedy_mean"
    g_mut = GreedyTopLayer()
    g_mut.name = "greedy_mutated"
    rnd = RandomStrategy()
    rnd.name = "random"
    add = AdditiveLinear()
    add.name = "additive"
    return {
        "greedy_mean": g_mean,
        "greedy_mutated": g_mut,
        "random": rnd,
        "additive": add,
    }


def run_all(
    landscape: Landscape,
    emb_mean: pd.DataFrame,
    emb_mut: pd.DataFrame,
    seeds: List[int],
    topks: List[int],
    target: float,
    n_init: int,
    n_per_round: int,
    n_rounds: int,
) -> pd.DataFrame:
    """Run every (strategy, seed, k) and collect final-round metrics (tidy).

    The embedding table each strategy sees:
      - greedy_mean    -> MEAN table
      - greedy_mutated -> MUTATED_TOKEN table
      - random         -> MEAN table (ignored by RandomStrategy)
      - additive       -> MEAN table (AdditiveLinear re-encodes one-hot tokens,
                          ignores the embedding values; only the index is used)
    """
    strategies = build_strategies()
    table_for = {
        "greedy_mean": emb_mean,
        "greedy_mutated": emb_mut,
        "random": emb_mean,
        "additive": emb_mean,
    }

    rows: List[Dict[str, float]] = []
    for sname, strat in strategies.items():
        emb = table_for[sname]
        for seed in seeds:
            # max_activity_found / rounds_to_target are k-independent (the
            # measured trajectory is fixed by seed); top_k_recall needs a run
            # per k. Determinism (same seed -> same rng) keeps the trajectory
            # identical across the per-k runs.
            recall_by_k: Dict[int, float] = {}
            max_found = float("nan")
            rtt = float("nan")
            for k in topks:
                df = run_simulation(
                    landscape, emb, strat,
                    n_init=n_init, n_per_round=n_per_round, n_rounds=n_rounds,
                    seed=seed, top_k=k, target=target,
                )
                final = df.iloc[-1]
                recall_by_k[k] = float(final["top_k_recall"])
                max_found = float(final["max_activity_found"])
                rtt = float(final["rounds_to_target"])
            row: Dict[str, float] = {"strategy": sname, "seed": seed}
            for k in topks:
                row[f"recall@{k}"] = recall_by_k[k]
            row["max_found"] = max_found
            row["rounds_to_target"] = rtt
            rows.append(row)
    return pd.DataFrame(rows)


# --- step 6: aggregate + verdict ----------------------------------------------

def summarize(tidy: pd.DataFrame, topks: List[int]) -> pd.DataFrame:
    """mean +/- sd over seeds per strategy (rows=strategy).

    ``rounds_to_target`` is NaN for any seed that never reaches the target, and
    ``mean``/``std`` skip those NaNs (pandas default). The reported mean rounds
    is therefore conditioned on the seeds that DID reach the target -- a
    survivorship bias that can make a weak strategy look fast on the needle.
    ``reach_rate`` (fraction of seeds with a non-NaN rounds_to_target) exposes
    that conditioning so the rounds column is read with its denominator.
    """
    metric_cols = [f"recall@{k}" for k in topks] + ["max_found", "rounds_to_target"]
    agg = tidy.groupby("strategy")[metric_cols].agg(["mean", "std"])
    reach = tidy.groupby("strategy")["rounds_to_target"].apply(
        lambda s: float(s.notna().mean())
    )
    agg[("reach_rate", "mean")] = reach
    agg[("reach_rate", "std")] = float("nan")
    return agg


def format_summary(agg: pd.DataFrame, topks: List[int]) -> str:
    """Render a compact mean+/-sd table (no em-dash)."""
    metric_cols = (
        [f"recall@{k}" for k in topks]
        + ["max_found", "rounds_to_target", "reach_rate"]
    )
    order = ["greedy_mean", "greedy_mutated", "random", "additive"]
    strategies = [s for s in order if s in agg.index]
    header = f"{'strategy':<16}" + "".join(f"{c:>20}" for c in metric_cols)
    lines = [header, "-" * len(header)]
    for s in strategies:
        cells = []
        for c in metric_cols:
            m = agg.loc[s, (c, "mean")]
            sd = agg.loc[s, (c, "std")]
            sd = 0.0 if pd.isna(sd) else sd
            if pd.isna(m):
                cells.append(f"{'nan':>20}")
            elif c == "reach_rate":
                cells.append(f"{m:>20.2f}")
            else:
                cells.append(f"{m:>13.4f}+/-{sd:<4.3f}")
        lines.append(f"{s:<16}" + "".join(cells))
    return "\n".join(lines)


def verdict_line(tidy: pd.DataFrame, primary_k: int) -> str:
    """Honest one-line comparison of greedy_mean vs greedy_mutated on recall@k."""
    col = f"recall@{primary_k}"
    gm = tidy.loc[tidy["strategy"] == "greedy_mean", col]
    gt = tidy.loc[tidy["strategy"] == "greedy_mutated", col]
    if gm.empty or gt.empty:
        return "VERDICT: cannot compare (a greedy arm is missing)."
    m_mean, m_sd = gm.mean(), gm.std(ddof=0)
    t_mean, t_sd = gt.mean(), gt.std(ddof=0)
    diff = t_mean - m_mean
    pooled = float(np.hypot(m_sd, t_sd))
    if abs(diff) < 1e-9 or (pooled > 0 and abs(diff) < 0.5 * pooled):
        return (
            f"VERDICT (recall@{primary_k}): no clear separation -- "
            f"greedy_mean={m_mean:.3f}+/-{m_sd:.3f}, "
            f"greedy_mutated={t_mean:.3f}+/-{t_sd:.3f} (diff={diff:+.3f})."
        )
    winner = "greedy_mutated" if diff > 0 else "greedy_mean"
    return (
        f"VERDICT (recall@{primary_k}): {winner} wins by {abs(diff):.3f} -- "
        f"greedy_mean={m_mean:.3f}+/-{m_sd:.3f}, "
        f"greedy_mutated={t_mean:.3f}+/-{t_sd:.3f}."
    )


# --- CLI ----------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_cache = script_dir.parent / "outputs" / "gap2_dms_cache"

    p = argparse.ArgumentParser(
        description="Real-DMS GAP-2 validation: greedy mean vs mutated_token.",
    )
    p.add_argument("--dataset", choices=sorted(PRESETS), default=None,
                   help="Preset carrying verified ground-truth mapping.")
    p.add_argument("--model", default="facebook/esm2_t12_35M_UR50D")
    p.add_argument("--max-variants", type=int, default=None,
                   help="0=all; if >0 subsample single mutants (seed-fixed). "
                        "Default: preset value.")
    p.add_argument("--subsample-seed", type=int, default=0)
    p.add_argument("--n-init", type=int, default=10)
    p.add_argument("--n-per-round", type=int, default=8)
    p.add_argument("--n-rounds", type=int, default=6)
    p.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    p.add_argument("--topk", default="10,20")
    p.add_argument("--cache-dir", default=str(default_cache))
    p.add_argument("--out-csv", default=None,
                   help="Tidy per-(strategy,seed) CSV. Default under cache-dir.")
    # generic overrides (when --dataset is not a preset fit)
    p.add_argument("--wt-fasta", default=None)
    p.add_argument("--activity-file", default=None)
    p.add_argument("--sheet", default=None)
    p.add_argument("--variant-col", default=None)
    p.add_argument("--activity-col", default=None)
    return p.parse_args(argv)


def resolve_config(args: argparse.Namespace) -> DatasetPreset:
    """Merge preset + generic overrides into one config (fail-fast on gaps)."""
    if args.dataset is not None:
        base = PRESETS[args.dataset]
        name = base.name
        wt = args.wt_fasta or base.wt_fasta
        act = args.activity_file or base.activity_file
        sheet = args.sheet or base.sheet
        vcol = args.variant_col or base.variant_col
        acol = args.activity_col or base.activity_col
        sub_default = base.subsample_default
    else:
        missing = [n for n, v in (
            ("--wt-fasta", args.wt_fasta),
            ("--activity-file", args.activity_file),
            ("--sheet", args.sheet),
            ("--variant-col", args.variant_col),
            ("--activity-col", args.activity_col),
        ) if v is None]
        if missing:
            raise SystemExit(
                f"Without --dataset, these are required: {', '.join(missing)}"
            )
        name = Path(args.activity_file).stem
        wt, act, sheet = args.wt_fasta, args.activity_file, args.sheet
        vcol, acol = args.variant_col, args.activity_col
        sub_default = 0
    max_variants = sub_default if args.max_variants is None else args.max_variants
    args.max_variants = max_variants
    return DatasetPreset(name, wt, act, sheet, vcol, acol, max_variants)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    cfg = resolve_config(args)
    seeds = [int(s) for s in str(args.seeds).split(",") if s.strip() != ""]
    topks = [int(k) for k in str(args.topk).split(",") if k.strip() != ""]
    cache_dir = Path(args.cache_dir)

    print(f"=== GAP-2 real-DMS validation: dataset={cfg.name} model={args.model} ===")

    # Step 1: oracle (single mutants matching WT).
    wt_sequence = read_wt_sequence(cfg.wt_fasta)
    activities, report = load_single_mutant_activities(
        cfg.activity_file, cfg.sheet, cfg.variant_col, cfg.activity_col, wt_sequence
    )
    print(f"[load] WT len={len(wt_sequence)} | {report.summary()}")

    # Step 2: candidate pool (subsample, seed-fixed). WT excluded from landscape.
    all_tokens = list(activities.keys())
    candidate_tokens = subsample_candidates(
        all_tokens, args.max_variants, args.subsample_seed
    )
    print(f"[pool] {len(candidate_tokens)} candidate single mutants "
          f"(from {len(all_tokens)}; max_variants={args.max_variants}, "
          f"seed={args.subsample_seed})")

    pool_size = len(candidate_tokens)
    if args.n_init + args.n_per_round * args.n_rounds > pool_size:
        raise SystemExit(
            f"Budget n_init({args.n_init}) + n_per_round({args.n_per_round})*"
            f"n_rounds({args.n_rounds}) exceeds pool size {pool_size}."
        )

    # Step 4: landscape from the FINAL candidate set; target from THIS set.
    run_activities = {tok: activities[tok] for tok in candidate_tokens}
    landscape = Landscape.from_mapping(run_activities)
    sorted_acts = sorted(run_activities.values(), reverse=True)
    target = float(sorted_acts[min(19, len(sorted_acts) - 1)])  # oracle top-20 cutoff
    print(f"[target] oracle top-20 activity cutoff (post-subsample) = {target:.4f}")

    # Step 3: real ESM-2 embeddings, both poolings, cached + resumable.
    emb_mean = compute_or_load_embeddings(
        "mean", args.model, cfg.name, candidate_tokens, wt_sequence, cache_dir
    )
    emb_mut = compute_or_load_embeddings(
        "mutated_token", args.model, cfg.name, candidate_tokens, wt_sequence, cache_dir
    )

    # Fairness assert: identical index sets => mean-vs-mutated is the ONLY diff.
    if set(emb_mean.index) != set(emb_mut.index):
        raise ValueError(
            "MEAN and MUTATED_TOKEN embedding tables have different index sets; "
            "the greedy arms would not start from the same seed set / pool."
        )

    # Step 5: run the shared loop across strategies x seeds x k.
    tidy = run_all(
        landscape, emb_mean, emb_mut, seeds, topks, target,
        args.n_init, args.n_per_round, args.n_rounds,
    )

    # Step 6: aggregate, print table, write CSV, verdict.
    agg = summarize(tidy, topks)
    print("\n=== summary (mean +/- sd over {} seeds) ===".format(len(seeds)))
    print(format_summary(agg, topks))

    out_csv = args.out_csv or str(cache_dir / f"gap2_results_{cfg.name}_"
                                   f"{args.model.replace('/', '_')}.csv")
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    tidy.to_csv(out_csv, index=False)
    print(f"\n[csv] tidy per-(strategy,seed) results -> {out_csv}")

    primary_k = topks[0]
    print("\n" + verdict_line(tidy, primary_k))

    # Sanity: recall must be a fraction in [0,1].
    recall_cols = [f"recall@{k}" for k in topks]
    rvals = tidy[recall_cols].to_numpy()
    if not np.all((rvals >= 0.0) & (rvals <= 1.0)):
        raise ValueError(f"Recall outside [0,1]; check metrics. cols={recall_cols}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
