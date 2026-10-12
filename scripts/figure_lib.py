#!/usr/bin/env python3
"""Shared config, data loaders, and metrics for the ESM-2 vs ESM-C figures.

Single source of truth for the cross-PLM held-out Spearman and the 96-well
plate-2 precision@95 used by compare_plm_spearman.py, plate_selection.py, and
make_figures.py. Numbers are recomputed from the cached mean embeddings in
outputs/gap2_dms_cache/ so the figures stay reproducible, not hardcoded.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "scripts"))
sys.path.insert(0, str(_ROOT))
from plm_evolve.embeddings import from_csv  # noqa: E402
from plm_evolve.regressor import TopLayerRegressor  # noqa: E402
from plm_evolve.variants import read_wt_sequence  # noqa: E402
from validate_gap2_dms import PRESETS, load_single_mutant_activities  # noqa: E402

CACHE_DIR = _ROOT / "outputs" / "gap2_dms_cache"

DATASET_HASH = {
    "kelsic": "0befd64b54da7283",
    "stiffler": "8ed01cbb75481839",
    "giacomelli": "74f993aa35b3c0a1",
    "cas12f": "fa8cfc248745a91d",
    "brenan": "4a57dd0fb3250ecc",
}
MEDIUM_PAIR = ("facebook/esm2_t30_150M_UR50D", "esmc_300m")
MATCHED_PAIR = ("facebook/esm2_t33_650M_UR50D", "esmc_600m")
HEADLINE_PAIR = ("facebook/esm2_t33_650M_UR50D", "esmc_300m")  # main panels: smaller ESM-C 300M vs larger ESM-2 650M
PROTEIN_ORDER = ["kelsic", "stiffler", "giacomelli", "cas12f", "brenan"]

ESM2_COLOR = "#A8C4E0"
ESMC_COLOR = "#1F4E79"
LABELS = {
    "facebook/esm2_t30_150M_UR50D": "ESM-2 150M",
    "facebook/esm2_t33_650M_UR50D": "ESM-2 650M",
    "esmc_300m": "ESM-C 300M",
    "esmc_600m": "ESM-C 600M",
}


def load_oracle(dataset: str) -> dict:
    cfg = PRESETS[dataset]
    wt = read_wt_sequence(cfg.wt_fasta)
    acts, _ = load_single_mutant_activities(
        cfg.activity_file, cfg.sheet, cfg.variant_col, cfg.activity_col, wt
    )
    return acts


def cache_path(dataset: str, model: str, hash_: str, pooling: str = "mean") -> Path:
    tag = model.replace("/", "_")
    return CACHE_DIR / f"emb_{dataset}_{tag}_{pooling}_{hash_}.csv"


def load_aligned(dataset: str, models, hash_: str, pooling: str = "mean"):
    """Return (ids, y, {model: X}) intersected over oracle and all model caches."""
    embs = {}
    for m in models:
        f = cache_path(dataset, m, hash_, pooling)
        if not f.exists():
            raise FileNotFoundError(f"Missing cache: {f}")
        embs[m] = from_csv(f)
    act = load_oracle(dataset)
    ids = set(act)
    for e in embs.values():
        ids &= set(e.index)
    ids = sorted(i for i in ids if i != "WT")
    if not ids:
        raise ValueError(f"No common single-mutant ids for {dataset} {models}")
    y = np.array([float(act[i]) for i in ids])
    X = {m: embs[m].loc[ids].to_numpy() for m in models}
    return ids, y, X


def held_out_spearman_samples(X, y, train_sizes, seeds=20):
    """Per-seed held-out Spearman(pred, true) per train size.

    Returns {train_size: np.ndarray of length ``seeds``}. Identical split logic
    to compare_plm_spearman.py so pairing across models by seed cancels split
    noise. This is the single source the mean/SE curve and the paired-gap test
    both build on.
    """
    n = len(y)
    out = {}
    for t in train_sizes:
        vals = []
        for s in range(seeds):
            rng = np.random.default_rng(s)
            idx = rng.permutation(n)
            tr, te = idx[:t], idx[t:]
            mdl = TopLayerRegressor(n_estimators=100, random_state=1, n_jobs=-1)
            mdl.fit(X[tr], y[tr])
            vals.append(spearmanr(mdl.predict(X[te]), y[te]).correlation)
        out[t] = np.array(vals, dtype=float)
    return out


def held_out_spearman_curve(X, y, train_sizes, seeds=20):
    """Mean and SE of held-out Spearman(pred, true) per train size over seeds.

    Wraps held_out_spearman_samples. Returns (mean: dict, se: dict) keyed by
    train size.
    """
    samples = held_out_spearman_samples(X, y, train_sizes, seeds)
    mean = {t: float(v.mean()) for t, v in samples.items()}
    se = {t: float(v.std(ddof=0) / np.sqrt(len(v))) for t, v in samples.items()}
    return mean, se


def plate2_precision(X, y, plate_size=95, seeds=30):
    """precision@plate_size for the model top-plate2 selection.

    Per seed: plate1 = random plate_size for training. Fit RF, predict on the
    remaining pool. plate2 = model top-plate_size by predicted activity.
    true_top = top-plate_size by true activity in the same pool. precision =
    overlap count of plate2 and true_top. Returns (mean_count, se) over seeds.
    """
    n = len(y)
    counts = []
    for s in range(seeds):
        rng = np.random.default_rng(1000 + s)
        idx = rng.permutation(n)
        tr, pool = idx[:plate_size], idx[plate_size:]
        if len(pool) < plate_size:
            raise ValueError("pool smaller than plate_size")
        mdl = TopLayerRegressor(n_estimators=100, random_state=1, n_jobs=-1)
        mdl.fit(X[tr], y[tr])
        pred = mdl.predict(X[pool])
        plate2 = set(pool[np.argsort(pred)[::-1][:plate_size]])
        true_top = set(pool[np.argsort(y[pool])[::-1][:plate_size]])
        counts.append(len(plate2 & true_top))
    c = np.array(counts, dtype=float)
    return float(c.mean()), float(c.std(ddof=0) / np.sqrt(len(c)))
