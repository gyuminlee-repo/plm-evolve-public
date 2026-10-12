#!/usr/bin/env python3
"""Promote the 96-well plate-2 precision@95 routine to a committed driver.

Metric definition (locked, so it cannot drift):
  - plate_size = 95 (one 96-well plate minus the WT well).
  - Per seed s in 0..29 the RNG is numpy default_rng(1000 + s).
  - plate1 = a random 95-variant training set; the RandomForest top-layer
    (n_estimators=100, random_state=1) is fit on it.
  - pool = the remaining n - 95 unmeasured variants (n is per-protein, set by
    the cached variant subsample: kelsic n=500, the other four n=350).
  - plate2 = the model top-95 of the pool, by predicted activity.
  - precision@95 = overlap of plate2 with true_top, where true_top = the top-95
    of the POOL by true activity (pool-def: plate1 is already measured, so it is
    not a re-selectable candidate). Mean over 30 seeds.

precision@95 alone is NOT comparable across proteins, because the random-pick
expectation = 95^2 / (n - 95) depends on n (kelsic 22.3, the others 35.4). So
this driver also records the random baseline and the enrichment-over-baseline
(precision - baseline), which IS comparable across proteins, plus the paired
ESM-C minus ESM-2 delta (the baseline cancels within a protein).

Runs ESM-2 650M vs ESM-C 300M on all five EVOLVEpro DMS proteins.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figure_lib as fl  # noqa: E402

PLATE = 95
OUT = fl._ROOT / "figures" / "esm2_vs_esmc" / "data" / "plate_precision.json"


def main() -> int:
    rows = {}
    for ds in fl.PROTEIN_ORDER:
        ids, y, X = fl.load_aligned(ds, list(fl.HEADLINE_PAIR), fl.DATASET_HASH[ds])
        n = len(y)
        baseline = PLATE * PLATE / (n - PLATE)
        rec = {"n": n, "pool": n - PLATE, "baseline": round(baseline, 1)}
        prec = {}
        for m in fl.HEADLINE_PAIR:
            mean_c, se = fl.plate2_precision(X[m], y, plate_size=PLATE, seeds=30)
            prec[m] = mean_c
            rec[fl.LABELS[m]] = {
                "precision_at_95": round(mean_c, 1),
                "se": round(se, 2),
                "enrichment": round(mean_c - baseline, 1),
            }
        rec["delta"] = round(prec[fl.HEADLINE_PAIR[1]] - prec[fl.HEADLINE_PAIR[0]], 1)
        rows[ds] = rec
        print(f"{ds:<11} n={n:<4} base={baseline:4.1f}  "
              f"ESM-2 {rec['ESM-2 650M']['precision_at_95']:4.1f} "
              f"(enr {rec['ESM-2 650M']['enrichment']:+.1f})  "
              f"ESM-C {rec['ESM-C 300M']['precision_at_95']:4.1f} "
              f"(enr {rec['ESM-C 300M']['enrichment']:+.1f})  delta {rec['delta']:+.1f}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=2))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
