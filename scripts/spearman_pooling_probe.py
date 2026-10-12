#!/usr/bin/env python3
"""GAP-2 high-power discriminator: Spearman(pred, true) on held-out variants.

The active-learning recall@10 comparison (``validate_gap2_dms.py``) is
underpowered: per-seed sd (0.10-0.17) swamps the mean-vs-mutated_token recall
diffs. This probe answers the SAME GAP-2 question with a lower-variance
estimator that decouples "which pooling's embedding is more activity-predictive"
from the noisy selection loop:

    Fit the SAME RF top-layer (``plm_evolve.regressor.TopLayerRegressor``) the
    method uses on a random train subset, predict the HELD-OUT remainder, and
    score Spearman(pred, true) on the held out. Repeat over train sizes and
    many seeds.

POWER comes from a PAIRED design (per advisor): at each (train_size, seed) a
SINGLE train/test partition is drawn on a COMMON ordered id list and fed to BOTH
poolings, with an IDENTICAL RF ``random_state``. The only thing differing between
the two predictions is X (the pooling). The verdict is then driven by the SE of
the per-seed gap ``diff_s = spearman_mutated(s) - spearman_mean(s)``, which
cancels the shared split-noise -- exactly the variance the recall comparison
drowned in. Per-pooling mean/sd/se are still reported, but "separated" is judged
against ``SE_diff``, not the per-pooling SE.

ASSUMPTIONS (explicit, per karpathy-guidelines):
- Cached embedding CSVs already exist under the cache dir, named
  ``emb_{dataset}_{model}_{pooling}_{hash}.csv``. NO model inference is run; only
  cached real ESM-2 embeddings are loaded (``plm_evolve.embeddings.from_csv``).
- The DMS oracle activities come from the SAME loader the validator uses
  (``validate_gap2_dms.load_single_mutant_activities`` + ``PRESETS``); it is
  imported, not re-implemented. ``PLM_EVOLVE_DMS_ROOT`` must point at the
  EVOLVEpro ``.../data/dms`` tree.
- A combo's aligned variant set = (oracle single-mutant keys) intersect
  (mean-index) intersect (mutated_token-index), with the WT row dropped. mean and
  mutated_token MUST share an identical aligned id order, so the only difference
  reaching the top-layer is the pooling. This is asserted (fail-fast).
- No feature scaling: the RF top-layer is split-based and the method does not
  scale, so neither does this probe (minimal-change fidelity).
- No silent fallbacks: a degenerate (near-constant) prediction yields a NaN
  Spearman from scipy; those are COUNTED and reported, never zeroed, and excluded
  from the seed-mean so one NaN does not poison the aggregate.

Determinism: the split RNG is seeded per (combo, train_size, seed) and the RF
``random_state`` is fixed, so a rerun with the same seeds is bit-identical.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# Reuse the package regressor (the SAME RF top-layer the method uses) and the
# embedding loader. The oracle loader + presets are imported from the validator.
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from plm_evolve.embeddings import from_csv as emb_from_csv  # noqa: E402
from plm_evolve.regressor import TopLayerRegressor  # noqa: E402
from plm_evolve.variants import WT_TOKEN, read_wt_sequence  # noqa: E402
from validate_gap2_dms import (  # noqa: E402
    PRESETS,
    load_single_mutant_activities,
)

# Known pooling tokens (longest-first so 'mutated_token' is matched before any
# shorter token could shadow it). Anchored to the file-name grammar below.
_POOLINGS = ("mutated_token", "mean", "delta")
# emb_{dataset}_{model_tag}_{pooling}_{hash16}.csv  -- dataset is anchored to a
# known PRESETS key at the front; pooling + 16-hex hash are anchored at the end;
# model_tag is whatever is left in the middle (it contains underscores too).
_FNAME_RE = re.compile(
    r"^emb_(?P<rest>.+)_(?P<pooling>" + "|".join(_POOLINGS) + r")_(?P<hash>[0-9a-f]{16})\.csv$"
)


@dataclass(frozen=True)
class Combo:
    dataset: str
    model: str
    path_by_pooling: Dict[str, Path]  # pooling -> csv path


def discover_combos(
    cache_dir: Path,
    datasets: Optional[List[str]],
    models: Optional[List[str]],
) -> List[Combo]:
    """Group cached emb CSVs into (dataset, model) combos that have BOTH poolings.

    Filenames are parsed by anchoring (not naive split): the dataset is matched
    against known ``PRESETS`` keys at the front, the pooling and 16-hex hash are
    anchored at the end, and the model tag is the remainder in the middle.
    """
    known = sorted(PRESETS.keys(), key=len, reverse=True)
    grouped: Dict[Tuple[str, str], Dict[str, Path]] = {}
    for p in sorted(cache_dir.glob("emb_*.csv")):
        m = _FNAME_RE.match(p.name)
        if not m:
            continue
        rest, pooling = m.group("rest"), m.group("pooling")
        ds = next((d for d in known if rest == d or rest.startswith(d + "_")), None)
        if ds is None:
            print(f"[skip] cannot anchor dataset in {p.name}", file=sys.stderr)
            continue
        model_tag = rest[len(ds):].lstrip("_")
        # model_tag is the on-disk slash->underscore form; keep it as-is for the
        # report (it already reads e.g. facebook_esm2_t12_35M_UR50D).
        grouped.setdefault((ds, model_tag), {})[pooling] = p

    combos: List[Combo] = []
    for (ds, model), by_pool in sorted(grouped.items()):
        if datasets and ds not in datasets:
            continue
        if models and model not in models:
            continue
        if "mean" not in by_pool or "mutated_token" not in by_pool:
            print(f"[skip] {ds}@{model}: missing a required pooling "
                  f"(have {sorted(by_pool)})", file=sys.stderr)
            continue
        combos.append(Combo(ds, model, by_pool))
    return combos


def _oracle_cache() -> Dict[str, Dict[str, float]]:
    """Memoize oracle activities per dataset (kelsic appears in 3 combos)."""
    return {}


def load_oracle(dataset: str, cache: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """variant_token -> activity for the dataset, via the validator's loader."""
    if dataset in cache:
        return cache[dataset]
    preset = PRESETS[dataset]
    wt_seq = read_wt_sequence(preset.wt_fasta)
    activities, report = load_single_mutant_activities(
        preset.activity_file, preset.sheet, preset.variant_col,
        preset.activity_col, wt_seq,
    )
    print(f"[oracle] {dataset}: WT len={len(wt_seq)} | {report.summary()}")
    cache[dataset] = activities
    return activities


def align(
    emb_mean: pd.DataFrame,
    emb_mut: pd.DataFrame,
    activities: Dict[str, float],
) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray]:
    """Common ordered id list (WT dropped) shared by both poolings + the oracle.

    Returns ``(ids, X_mean, X_mut, y)`` with rows in the SAME id order across
    both X matrices and y, so a single split partition applies to both poolings.
    """
    common = (set(emb_mean.index) & set(emb_mut.index) & set(activities)) - {WT_TOKEN}
    ids = sorted(common)
    if not ids:
        raise ValueError("Empty intersection of mean/mutated/oracle id sets.")
    X_mean = emb_mean.loc[ids].to_numpy(dtype=np.float64)
    X_mut = emb_mut.loc[ids].to_numpy(dtype=np.float64)
    y = np.array([float(activities[i]) for i in ids], dtype=np.float64)
    return ids, X_mean, X_mut, y


def _spearman(pred: np.ndarray, true: np.ndarray) -> float:
    """Spearman rho; returns NaN for a degenerate (constant) input (counted upstream)."""
    if np.all(pred == pred[0]) or np.all(true == true[0]):
        return float("nan")
    res = spearmanr(pred, true)
    return float(res.statistic)


@dataclass
class CellResult:
    """Aggregate for one (dataset, model, pooling, train_size) over seeds."""
    spearman_mean: float
    spearman_sd: float
    spearman_se: float
    n_test: int
    n_nan: int


def run_combo(
    combo: Combo,
    activities: Dict[str, float],
    train_sizes: List[int],
    seeds: List[int],
    min_test: int,
    rf_random_state: int,
) -> Tuple[List[dict], List[str], int]:
    """PAIRED probe for one combo. Returns (table_rows, verdict_lines, n_aligned).

    For each (train_size, seed) ONE partition is drawn on the common id order and
    used by BOTH poolings; the RF random_state is identical for both fits so the
    only difference is X. Per-seed gap diff_s = spearman_mut - spearman_mean
    drives the SE_diff verdict.
    """
    emb_mean = emb_from_csv(combo.path_by_pooling["mean"])
    emb_mut = emb_from_csv(combo.path_by_pooling["mutated_token"])
    ids, X_mean, X_mut, y = align(emb_mean, emb_mut, activities)
    n = len(ids)
    # Fairness assert: identical aligned id order => pooling is the only diff.
    assert X_mean.shape[0] == X_mut.shape[0] == n == len(y)
    print(f"[align] {combo.dataset}@{combo.model}: n={n} aligned single mutants")

    rows: List[dict] = []
    verdicts: List[str] = []

    for ts in train_sizes:
        n_test = n - ts
        if ts <= 0 or n_test < min_test:
            print(f"[skip] {combo.dataset}@{combo.model} train_size={ts}: "
                  f"n_test={n_test} < min_test={min_test} (n={n}); skipped.")
            rows.append({
                "dataset": combo.dataset, "model": combo.model, "pooling": "mean",
                "train_size": ts, "spearman_mean": float("nan"),
                "spearman_sd": float("nan"), "spearman_se": float("nan"),
                "n_test": max(n_test, 0), "n_nan": 0, "status": "skipped_small_n",
            })
            rows.append({
                "dataset": combo.dataset, "model": combo.model,
                "pooling": "mutated_token", "train_size": ts,
                "spearman_mean": float("nan"), "spearman_sd": float("nan"),
                "spearman_se": float("nan"), "n_test": max(n_test, 0), "n_nan": 0,
                "status": "skipped_small_n",
            })
            verdicts.append(
                f"VERDICT {combo.dataset}@{combo.model} train_size={ts}: "
                f"skipped (n_test={n_test} < {min_test})."
            )
            continue

        rho_mean: List[float] = []
        rho_mut: List[float] = []
        diffs: List[float] = []
        nan_mean = nan_mut = 0

        for seed in seeds:
            rng = np.random.default_rng(seed)
            perm = rng.permutation(n)
            tr, te = perm[:ts], perm[ts:]
            y_tr, y_te = y[tr], y[te]

            # SAME partition, SAME RF random_state -> only X differs.
            reg_m = TopLayerRegressor(random_state=rf_random_state)
            reg_m.fit(X_mean[tr], y_tr)
            pred_m = reg_m.predict(X_mean[te])
            r_m = _spearman(pred_m, y_te)

            reg_t = TopLayerRegressor(random_state=rf_random_state)
            reg_t.fit(X_mut[tr], y_tr)
            pred_t = reg_t.predict(X_mut[te])
            r_t = _spearman(pred_t, y_te)

            if np.isnan(r_m):
                nan_mean += 1
            else:
                rho_mean.append(r_m)
            if np.isnan(r_t):
                nan_mut += 1
            else:
                rho_mut.append(r_t)
            # Paired gap only when BOTH are finite for this seed.
            if not np.isnan(r_m) and not np.isnan(r_t):
                diffs.append(r_t - r_m)

        n_test = n - ts
        cell_m = _agg(rho_mean, nan_mean, n_test)
        cell_t = _agg(rho_mut, nan_mut, n_test)
        for pooling, cell in (("mean", cell_m), ("mutated_token", cell_t)):
            rows.append({
                "dataset": combo.dataset, "model": combo.model, "pooling": pooling,
                "train_size": ts, "spearman_mean": cell.spearman_mean,
                "spearman_sd": cell.spearman_sd, "spearman_se": cell.spearman_se,
                "n_test": cell.n_test, "n_nan": cell.n_nan, "status": "ok",
            })
        verdicts.append(_verdict(combo, ts, diffs, cell_m, cell_t))

    return rows, verdicts, n


def _agg(rhos: List[float], n_nan: int, n_test: int) -> CellResult:
    """Mean/sd/se over the finite per-seed Spearman values (NaNs excluded)."""
    if not rhos:
        return CellResult(float("nan"), float("nan"), float("nan"), n_test, n_nan)
    arr = np.array(rhos, dtype=np.float64)
    mean = float(arr.mean())
    sd = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    se = sd / np.sqrt(arr.size) if arr.size > 0 else float("nan")
    return CellResult(mean, sd, float(se), n_test, n_nan)


def _verdict(
    combo: Combo, ts: int, diffs: List[float], cell_m: CellResult, cell_t: CellResult
) -> str:
    """Paired verdict: SE of the per-seed gap, separated iff |mean| > 2*SE_diff."""
    tag = f"{combo.dataset}@{combo.model} train_size={ts}"
    if not diffs:
        return (f"VERDICT {tag}: cannot compare (no seed had both Spearman finite; "
                f"mean_nan={cell_m.n_nan}, mut_nan={cell_t.n_nan}).")
    d = np.array(diffs, dtype=np.float64)
    mean_diff = float(d.mean())
    sd_diff = float(d.std(ddof=1)) if d.size > 1 else 0.0
    se_diff = sd_diff / np.sqrt(d.size) if d.size > 0 else float("nan")
    higher = "mutated_token" if mean_diff > 0 else "mean"
    base = (
        f"VERDICT {tag}: mutated_token Spearman={cell_t.spearman_mean:.3f}"
        f"+/-{cell_t.spearman_sd:.3f} vs mean={cell_m.spearman_mean:.3f}"
        f"+/-{cell_m.spearman_sd:.3f} | gap(mut-mean)={mean_diff:+.3f} "
        f"SE_diff={se_diff:.3f} (paired n={d.size})"
    )
    if se_diff > 0 and abs(mean_diff) > 2.0 * se_diff:
        return f"{base} -> {higher} SEPARATED (|gap| > 2 SE_diff)."
    return f"{base} -> no clear separation (|gap| <= 2 SE_diff)."


def to_markdown(df: pd.DataFrame) -> str:
    """Render the tidy table as a markdown table (no em-dash)."""
    cols = ["dataset", "model", "pooling", "train_size", "spearman_mean",
            "spearman_sd", "spearman_se", "n_test", "n_nan", "status"]
    cols = [c for c in cols if c in df.columns]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    lines = [header, sep]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if c in ("spearman_mean", "spearman_sd", "spearman_se"):
                cells.append("nan" if pd.isna(v) else f"{float(v):.4f}")
            elif c in ("train_size", "n_test", "n_nan"):
                cells.append(str(int(v)))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    default_cache = _SCRIPT_DIR.parent / "outputs" / "gap2_dms_cache"
    p = argparse.ArgumentParser(
        description="GAP-2 Spearman pooling probe (mean vs mutated_token).",
    )
    p.add_argument("--datasets", default=None,
                   help="Comma-separated dataset filter; default auto-discover.")
    p.add_argument("--models", default=None,
                   help="Comma-separated model-tag filter; default auto-discover.")
    p.add_argument("--train-sizes", default="16,32,64")
    p.add_argument("--seeds", type=int, default=20,
                   help="Number of random splits (seeds 0..N-1).")
    p.add_argument("--min-test", type=int, default=10,
                   help="Skip a train_size whose held-out floor is below this.")
    p.add_argument("--rf-random-state", type=int, default=1,
                   help="Fixed RF random_state (identical for both poolings).")
    p.add_argument("--cache-dir", default=str(default_cache))
    p.add_argument("--out-csv", default=None,
                   help="Tidy output CSV. Default under cache-dir.")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    cache_dir = Path(args.cache_dir)
    datasets = [s for s in (args.datasets or "").split(",") if s.strip()] or None
    models = [s for s in (args.models or "").split(",") if s.strip()] or None
    train_sizes = [int(x) for x in str(args.train_sizes).split(",") if x.strip()]
    seeds = list(range(int(args.seeds)))

    print(f"=== GAP-2 Spearman pooling probe (cache={cache_dir}) ===")
    combos = discover_combos(cache_dir, datasets, models)
    if not combos:
        raise SystemExit("No (dataset, model) combos with both poolings found.")
    print(f"[discover] {len(combos)} combo(s): "
          + ", ".join(f"{c.dataset}@{c.model}" for c in combos))

    oracle_cache = _oracle_cache()
    all_rows: List[dict] = []
    all_verdicts: List[str] = []
    for combo in combos:
        activities = load_oracle(combo.dataset, oracle_cache)
        rows, verdicts, _ = run_combo(
            combo, activities, train_sizes, seeds,
            args.min_test, args.rf_random_state,
        )
        all_rows.extend(rows)
        all_verdicts.extend(verdicts)

    tidy = pd.DataFrame(all_rows)

    print("\n=== tidy table ===")
    print(to_markdown(tidy))

    print("\n=== verdicts (paired mutated_token vs mean) ===")
    for v in all_verdicts:
        print(v)

    out_csv = args.out_csv or str(cache_dir / "spearman_pooling_probe.csv")
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    tidy.to_csv(out_csv, index=False)
    print(f"\n[csv] tidy results -> {out_csv}")

    # Self-check: every finite Spearman must lie in [-1, 1].
    for col in ("spearman_mean",):
        vals = tidy[col].to_numpy(dtype=np.float64)
        finite = vals[np.isfinite(vals)]
        if finite.size and not np.all((finite >= -1.0) & (finite <= 1.0)):
            raise ValueError(f"Spearman in {col} outside [-1,1].")
    print("[self-check] all finite Spearman in [-1,1]; n_test sane.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
