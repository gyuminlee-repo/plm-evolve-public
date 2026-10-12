# plm-evolve

An independent implementation of the **core principle** of
EVOLVEpro: an active-learning loop for experimental protein directed evolution
driven by a frozen protein language model (PLM).

This package is a from-scratch reimplementation of the published method. It
shares no source with the original EVOLVEpro codebase. The frozen PLM is
swappable between ESM-2 and ESM-C. Citations to the method and to the PLMs
(below) are standard academic practice.

## The core principle

1. **Freeze** a protein language model (ESM-2). Embed every single-amino-acid
   mutant of a wild-type protein (a masked mean over residue embeddings).
2. **Measure** a small number of variants in the lab (one or more rounds).
3. **Train** a light top-layer regressor (RandomForest) on those measured
   variants, using their frozen embeddings as features.
4. **Predict** activity for every unmeasured mutant, **rank** them, and
   **propose** the top-N not-yet-measured variants for the next round.
5. **Iterate**: feed each round of new measurements back into the loop.

The steps above are the experimental directed-evolution loop, the production
core. A separate research and benchmark layer (`plm_evolve.benchmark`, opt-in,
documented below) adds an in-silico simulation harness, alternate embedding
pooling, uncertainty-aware and batch-diverse acquisition, multi-mutant
stacking, and a multi-objective hook. The production loop and its defaults stay
unchanged by that layer.

## Install

One command (creates an isolated venv, installs everything, no conda):

```bash
./setup.sh
source .venv/bin/activate
```

Manual, if you prefer:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[esmc,viz]"   # core + ESM-C SDK + plotting; drop extras you do not need
```

All pip, no conda. Core deps: `pandas numpy scikit-learn biopython torch transformers openpyxl`;
extras `esm` (ESM-C) and `matplotlib` (figures). ESM-2 weights download from HuggingFace on first
use, ESM-C from the esm SDK. CPU is enough; a GPU only speeds up embedding extraction.

## CLI

### 1. Embed all single mutants

Generate every `19 x L` single-substitution sequence (plus WT) from a
wild-type FASTA and embed them with a protein language model (ESM-2 or ESM-C):

```bash
plm-evolve embed --wt WT.fasta --out emb.csv
# ESM-2 (default family):    --model facebook/esm2_t33_650M_UR50D
# ESM-C (EvolutionaryScale):  --model esmc_300m   (or esmc_600m; needs the esm SDK)
```

`esmc_*` model names route through `ESMCExtractor` automatically; everything
else uses `ESM2Extractor`. A real-DMS comparison
(`notes/esm2_vs_esmc_comparison.md`) found ESM-C more activity-predictive than
ESM-2 on 4 of 5 proteins, with ESM-C 300M the efficient choice.

Output `emb.csv` is indexed by variant id (`<WTaa><pos><newaa>`, 1-based, plus
a `WT` row); columns are embedding dimensions.

### 2. Rank candidates and propose the next round

Given the embeddings and one or more measured-round files (Excel/CSV with a
`Variant` column and an `activity` column):

```bash
plm-evolve rank --embeddings emb.csv --wt WT.fasta \
    --rounds round1.xlsx round2.xlsx --top 12 --out out_dir/
```

The command writes `out_dir/ranked.csv` (all candidates by predicted activity)
and `out_dir/next_round.csv` (the top-N not-yet-measured candidates).

The acquisition is opt-in and defaults to plain greedy (the default invocation
is unchanged). `--acquisition {greedy,ucb,thompson,coverage}` selects the
next-round set by an uncertainty-aware (`ucb` with `--beta`, `thompson`) or
mutation-site-diverse (`coverage` with `--penalty`) rule instead of greedy:

```bash
plm-evolve rank --embeddings emb.csv --wt WT.fasta --rounds round1.xlsx \
    --acquisition ucb --beta 1.0 --top 12 --out out_dir/
```

### 3. Propose multi-mutants

Stack the best measured/predicted single mutants into multi-mutant proposals by
an additive model (no epistasis term):

```bash
plm-evolve propose-multi --embeddings emb.csv --wt WT.fasta --rounds round1.xlsx \
    --order 2 --top-singles 20 --top 24 --out out_dir/
```

## Library

```python
import pandas as pd
from plm_evolve import ESM2Extractor, run_round

emb = ESM2Extractor().embed_fasta("single_mutants.fasta")
outcome = run_round(emb, ["round1.xlsx"], "WT.fasta", top_n=12)
print(outcome.next_round)
```

## Research and benchmark layer (`plm_evolve.benchmark`, opt-in)

A research layer measures the loop in silico and adds optional capabilities. It
never changes the production defaults above.

- **In-silico simulation harness**: replay the same train/predict/rank loop
  against an oracle landscape (synthetic, a generic CSV, or a ProteinGym DMS)
  instead of a wet-lab round, with sample-efficiency metrics (top-k recall,
  rounds-to-target, simple regret, Spearman).
- **Embedding pooling** (opt-in, default `mean` unchanged): `mean`, `delta`
  (variant minus WT), `mutated_token` (the mutated-position embedding). Under
  the RandomForest top-layer, `delta` is provably equal to `mean`.
- **Uncertainty-aware acquisition**: `TopLayerRegressor.predict_with_uncertainty`
  returns the per-tree mean and standard deviation; `UCBStrategy` (with
  `beta=0` reducing exactly to greedy) and `ThompsonStrategy`.
- **Batch diversity**: `CoverageBatchStrategy` (mutation-site coverage) and
  `ConstantLiarStrategy` (a FolDE-style parity baseline).
- **Multi-mutant**: additive stacking of top single mutants
  (`AdditiveMultiPredictor`, no epistasis term by design).
- **Multi-objective**: `compose_objectives` scalarizes several objective
  landscapes into one oracle so the single-objective loop runs unchanged.

These pooling and acquisition features are parity with prior work (FolDE,
Roberts 2025; ALDE, Wittmann 2025; MULTI-evolve, Tran 2026), not novel claims;
each ships with an honest ablation marking where it helps and where it does not.

A real-DMS check of `mutated_token` vs `mean` few-shot selection lives in
`scripts/validate_gap2_dms.py` and `scripts/spearman_pooling_probe.py`; the
finding (`notes/gap2_dms_validation.md`) is that `mutated_token` shows no
advantage on the application metric where it is meaningful, on the two proteins
and small models tested.

A separate PLM comparison (`scripts/compare_plm_spearman.py`,
`notes/esm2_vs_esmc_comparison.md`) finds the PLM family matters more than the
pooling: ESM-C predicts activity better than ESM-2 on 4 of 5 proteins, while the
mutated-token pooling is negative under both PLMs.

## Documentation

- [`docs/USAGE.md`](docs/USAGE.md): end-to-end campaign walkthrough (embed,
  rank with acquisition, propose-multi, iterate, in-silico comparison).
- [`CHANGELOG.md`](CHANGELOG.md): version history.
- [`CONTRIBUTING.md`](CONTRIBUTING.md): development setup, tests, conventions.
- [`notes/gap2_dms_validation.md`](notes/gap2_dms_validation.md): the factual
  real-DMS pooling numbers.
- [`notes/esm2_vs_esmc_comparison.md`](notes/esm2_vs_esmc_comparison.md): the
  ESM-2 vs ESM-C real-DMS comparison numbers.

## Method & Citation

This is an independent implementation of the method described in:

- Jiang, K. et al. (2025). "Rapid in silico directed evolution by a protein
  language model with EVOLVEpro." *Science*.
  https://doi.org/10.1126/science.adr6006

Protein language model used for embeddings:

- Lin, Z. et al. (2023). "Evolutionary-scale prediction of atomic-level protein
  structure with a language model." *Science* 379, 1123-1130.
  ESM-2 model: `facebook/esm2_t33_650M_UR50D`.
- ESM-C (EvolutionaryScale, ESM Cambrian, 2024); models `esmc_300m` / `esmc_600m`
  accessed via the `esm` SDK.

## License

MIT.
