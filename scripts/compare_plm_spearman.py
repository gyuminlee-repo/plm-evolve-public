#!/usr/bin/env python3
"""Cross-PLM held-out Spearman comparison on a real DMS oracle.

Answers "is PLM B more activity-predictive than PLM A under the EVOLVEpro core
(mean-pool + RandomForest top-layer)?" by a PAIRED held-out Spearman: for each
(train_size, seed) one random split on the shared variant order, fit the core
TopLayerRegressor on the train subset of EACH model's mean embeddings, and
Spearman(pred, true) on the identical held-out set. Pairing by split seed cancels
the split noise. All embeddings must be the SAME variant-set hash and generated
under the SAME environment (no library skew) for the comparison to be valid.

Usage:
  python3 scripts/compare_plm_spearman.py --dataset kelsic --hash 0befd64b54da7283 \\
      --models facebook/esm2_t30_150M_UR50D esmc_300m facebook/esm2_t33_650M_UR50D esmc_600m

Reuses the cached mean embeddings written by scripts/validate_gap2_dms.py
(outputs/gap2_dms_cache/emb_{dataset}_{model_tag}_mean_{hash}.csv).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from plm_evolve.embeddings import from_csv  # noqa: E402
from validate_gap2_dms import PRESETS, load_single_mutant_activities  # noqa: E402
from plm_evolve.variants import read_wt_sequence  # noqa: E402
import figure_lib  # noqa: E402  shared CV (single source of truth)


def _oracle(dataset: str) -> dict:
    cfg = PRESETS[dataset]
    wt = read_wt_sequence(cfg.wt_fasta)
    acts, _ = load_single_mutant_activities(
        cfg.activity_file, cfg.sheet, cfg.variant_col, cfg.activity_col, wt
    )
    return acts


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Cross-PLM held-out Spearman on real DMS.")
    p.add_argument("--dataset", choices=sorted(PRESETS), required=True)
    p.add_argument("--hash", required=True, help="variant-set hash shared by all model caches")
    p.add_argument("--models", nargs="+", required=True, help="model tags (as used in --model)")
    p.add_argument("--train-sizes", default="16,32,64")
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--cache-dir", default=str(Path(__file__).resolve().parent.parent / "outputs" / "gap2_dms_cache"))
    args = p.parse_args(argv)

    cache = Path(args.cache_dir)
    tags = {m: m.replace("/", "_") for m in args.models}
    embs = {}
    for m, tag in tags.items():
        f = cache / f"emb_{args.dataset}_{tag}_mean_{args.hash}.csv"
        if not f.exists():
            raise SystemExit(f"Missing cache: {f}\nGenerate it first with validate_gap2_dms.py --model {m}.")
        embs[m] = from_csv(f)

    act = _oracle(args.dataset)
    ids = set(act)
    for e in embs.values():
        ids &= set(e.index)
    ids = sorted(i for i in ids if i != "WT")
    if not ids:
        raise SystemExit("No common single-mutant ids across the model caches and oracle.")
    y = np.array([float(act[i]) for i in ids])
    X = {m: embs[m].loc[ids].to_numpy() for m in args.models}
    train_sizes = [int(t) for t in args.train_sizes.split(",")]

    print(f"dataset={args.dataset} n={len(ids)} | widths " + ", ".join(f"{m}={X[m].shape[1]}" for m in args.models))

    # Per-seed held-out Spearman from the shared figure_lib routine so the
    # figures, the tables, and this CLI all use one CV implementation.
    curves = {
        m: figure_lib.held_out_spearman_samples(X[m], y, train_sizes, seeds=args.seeds)
        for m in args.models
    }
    print("\nheld-out Spearman (mean +/- sd over %d seeds):" % args.seeds)
    for m in args.models:
        print(f"  {m:<32}" + "  ".join(f"ts{t}={curves[m][t].mean():.3f}+/-{curves[m][t].std():.3f}" for t in train_sizes))

    print("\npaired gaps (later model minus earlier), separated iff |gap| > 2*SE:")
    for i in range(len(args.models) - 1):
        a, b = args.models[i], args.models[i + 1]
        for t in train_sizes:
            d = curves[b][t] - curves[a][t]
            se = d.std(ddof=0) / np.sqrt(len(d))
            verdict = ("higher" if d.mean() > 0 else "lower") if abs(d.mean()) > 2 * se else "no separation"
            print(f"  {b} vs {a} ts={t}: gap={d.mean():+.3f} SE={se:.3f} -> {b} {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
