# ESM-2 vs ESM-C figure bundle

Paper-ready figures and tables for the question: under the EVOLVEpro core
(frozen-PLM mean-pool + RandomForest top-layer), does swapping ESM-2 for ESM-C
improve few-shot variant ranking and per-plate yield? Every number here recomputes
from the cached real-DMS embeddings, not from hardcoded values.

## Files

| file | content |
|---|---|
| `fig1_esmc_vs_esm2.svg` | Main figure (hero-top 3 panels). a: per-protein held-out Spearman at train=95 (ESM-2 650M vs ESM-C 300M, 5 proteins), the smaller ESM-C 300M against the larger ESM-2 650M. b: learning curve (Spearman vs train size, kelsic and cas12f). c: plate-2 enrichment over the random baseline for all 5 proteins, with the ESM-C minus ESM-2 delta annotated. |
| `figS1_size_dims.svg` | Supplementary. a: Spearman vs model size on kelsic (ESM-2 saturates, ESM-C 300M matches 600M). b: Spearman vs embedding dimensionality (the edge is not feature count). |
| `figS2_pooling_null.svg` | Supplementary. Mean vs mutated-token pooling under both PLMs (the GAP-2 pooling trick is null). |
| `tables.md` | Table 1 (per-protein ranking + per-plate enrichment) and Table S1 (full Spearman sweep). |
| `data/spearman_sweep.json` | Memoized per-(protein, model, pooling, train size) held-out Spearman (mean, SE, 20-seed samples, embedding width). |
| `data/plate_precision.json` | Plate-2 precision@95, baseline, enrichment, and delta per protein. |
| `data/fig1_numbers.json` | The train=95 Spearman values rendered in Fig1 panel a. |

## Metric definitions

- Held-out Spearman: RandomForest top-layer (n_estimators=100, random_state=1, n_jobs=-1)
  fit on a random train subset of the mean-pool embeddings, Spearman(predicted, true)
  on the held-out pool, paired by split seed across models, 20 seeds.
- Plate-2 precision@95 (pool-def): plate1 = 95 random training wells, plate2 = the
  model top-95 of the remaining pool, precision@95 = overlap of plate2 with the pool
  true-top-95, 30 seeds (default_rng(1000+s)).
- Enrichment = precision@95 minus the random baseline 95^2/(n-95). Raw precision@95 is
  NOT comparable across proteins (the baseline depends on n: 22.3 for kelsic n=500,
  35.4 for the n=350 proteins), so enrichment and the paired ESM-C minus ESM-2 delta
  are the comparable quantities. brenan enrichment sits near chance under both PLMs.

## Honest framing

- ESM-C beats ESM-2 on 4 of 5 proteins; brenan is the no-win case (both PLMs near
  random, gap not separated at 2 SE). It is shown, never averaged away.
- The advantage is ranking accuracy and per-plate enrichment, not finding the single
  best variant faster. Numbers are in-silico on DMS oracles, not a prospective wet-lab run.
- The additive baseline is degenerate on single-mutant landscapes and is not plotted.

## Regenerate

```bash
python3 scripts/plate_selection.py        # writes data/plate_precision.json
python3 scripts/make_figures.py           # writes the 3 SVGs + tables.md (+ caches the sweep)
```

The first full run computes the sweep (about 4 minutes with n_jobs=-1); later runs
reuse `data/spearman_sweep.json` and finish in seconds. Source embeddings live in
`outputs/gap2_dms_cache/` (real EVOLVEpro DMS, regenerated under one library version).
