# Usage guide

End-to-end walkthrough of a plm-evolve directed-evolution campaign, from a
wild-type sequence to proposed variants for the next experimental round.

## Install

```bash
pip install -e .
# development (adds pytest):
pip install -e ".[dev]"
```

Dependencies: `pandas numpy scikit-learn biopython torch transformers openpyxl`.
ESM-2 weights download automatically from HuggingFace on first use.

## The loop, in one paragraph

Freeze a protein language model and embed every single-amino-acid mutant of a
wild-type protein. Measure a small set of variants in the lab. Train a light
RandomForest top-layer on those measured embeddings, predict activity for every
unmeasured mutant, rank them, and propose the top-N for the next round. Feed
each new round of measurements back in and repeat. A campaign typically needs a
handful of rounds of ten to twenty variants each.

## Round file format

A measured-round file is an Excel or CSV table with a variant-token column
(default header `Variant`) and an activity column (default header `activity`).
Variant tokens follow `<WTaa><pos><newaa>`, 1-based, for example `M1A`. A `WT`
row is allowed and denotes the wild-type baseline.

| Variant | activity |
| --- | --- |
| WT | 1.00 |
| M1A | 0.62 |
| K2R | 1.45 |

## Step 1: embed the single mutants

Generate every `19 x L` single-substitution sequence (plus WT) from the
wild-type FASTA and embed them:

```bash
plm-evolve embed --wt WT.fasta --out emb.csv
# smaller/faster model:
plm-evolve embed --wt WT.fasta --out emb.csv --model facebook/esm2_t12_35M_UR50D
```

`emb.csv` is indexed by variant id with one row per single mutant plus a `WT`
row; columns are embedding dimensions. This step is the slow one (one forward
pass per mutant); a GPU helps for long proteins.

### PLM backend (ESM-2 or ESM-C)

The PLM is swappable by the `--model` name. ESM-2 models
(`facebook/esm2_t30_150M_UR50D`, `facebook/esm2_t33_650M_UR50D`) load via
HuggingFace transformers; ESM-C models (`esmc_300m`, `esmc_600m`,
EvolutionaryScale) load via the `esm` SDK and route through `ESMCExtractor`
automatically:

```bash
plm-evolve embed --wt WT.fasta --out emb.csv --model esmc_300m
```

A real-DMS comparison (`notes/esm2_vs_esmc_comparison.md`) found ESM-C more
activity-predictive than ESM-2 on 4 of 5 proteins, with ESM-C 300M matching
ESM-C 600M (size saturates). One protein (brenan/MAPK1) was unpredictable for
both PLMs, so a quick PLM-predictability check on related data before committing
a campaign is prudent.

### Pooling modes (opt-in)

The default pooling is a masked mean over residue embeddings. Two opt-in
alternatives exist via `--pooling`:

- `mean` (default): the whole-protein mean embedding.
- `delta`: the variant mean minus the WT mean. Under the RandomForest top-layer
  this is provably equal to `mean`; it exists as a diagnostic and for non-tree
  regressors.
- `mutated_token`: the embedding at the mutated residue position.

A real-DMS check (see below) found no advantage for `mutated_token` over `mean`
on the application metric, so `mean` remains the recommended default.

## Step 2: rank and propose the next round

Given the embeddings and one or more measured-round files:

```bash
plm-evolve rank --embeddings emb.csv --wt WT.fasta \
    --rounds round1.xlsx --top 12 --out out_dir/
```

The command writes `out_dir/ranked.csv` (every candidate by predicted activity)
and `out_dir/next_round.csv` (the top-N not-yet-measured candidates). Hand the
next-round set to the lab, measure it, and pass the new file as an extra
`--rounds` argument in the following round.

### Acquisition: how the next round is chosen (opt-in)

Every acquisition rule trains the SAME top-layer RandomForest on the SAME
measured set; the rules differ only in how a batch is drawn from the fitted
model (`plm_evolve/benchmark/strategies.py`). Two signals drive every rule
(`plm_evolve/regressor.py`, `predict_with_uncertainty`): the per-candidate
`mean` (the RandomForest point prediction, identical to greedy) and `std` (the
spread across the 100 trees, an epistemic "how much the trees disagree"
uncertainty).

Key point: `--acquisition` changes ONLY `next_round.csv` (the proposed batch
handed to the lab). `ranked.csv` (every candidate sorted by predicted mean) is
built the same way in every case (`plm_evolve/evolve.py`, `run_round`). The
overall predicted-activity ranking does not move; the selected next-round subset
does.

| Acquisition | Knob | Next-round batch rule | Reduces to greedy when |
| --- | --- | --- | --- |
| `greedy` (default) | none | top-N of the ranking (predicted mean) | always (it is greedy) |
| `ucb` | `beta` | top-N by `mean + beta * std` | `beta = 0` |
| `thompson` | `seed` | one independent draw from `N(mean, std)` per pick, argmax each | never (always stochastic) |
| `coverage` | `penalty` (plus `beta`) | UCB-base score minus `penalty` per earlier same-residue pick | `penalty = 0` |

- **greedy** picks the top-N predicted-mean candidates. Pure exploitation, the
  original EVOLVEpro behaviour.
- **ucb** ranks by `mean + beta * std`, so candidates the trees disagree on
  (high `std`) rise. Higher `beta` explores more; `beta = 0` is byte-identical
  to greedy.
- **thompson** treats each prediction as a posterior `N(mean, std)` and draws
  one independent sample PER pick (not one shared sample), taking the argmax of
  the not-yet-picked each time. All randomness comes from `seed`, so the batch
  is reproducible. The fresh-sample-per-pick is what spreads the batch instead
  of collapsing onto the single greedy argmax.
- **coverage** assembles the batch by largest EFFECTIVE score, where `effective
  = base_score - penalty * (picks already taken at this residue within this
  batch)`. A second substitution at one site is discounted by `penalty`, a
  third by `2 * penalty`, steering the round across distinct residues. In the
  `rank` path the base score is the UCB score (`mean + beta * std`), and
  `penalty = 0` reduces exactly to that base top-N.

```bash
plm-evolve rank --embeddings emb.csv --wt WT.fasta --rounds round1.xlsx \
    --acquisition ucb --beta 1.0 --top 12 --out out_dir/

plm-evolve rank --embeddings emb.csv --wt WT.fasta --rounds round1.xlsx \
    --acquisition coverage --penalty 1.0 --top 12 --out out_dir/
```

These acquisition rules are parity with prior work (FolDE, ALDE); each improves
sample efficiency only under the conditions noted above, not universally. A cold
start with no measured rows falls back to uniform-random selection.

## Step 3: propose multi-mutants

Once single-mutant effects are known, stack the best singles into multi-mutant
proposals by an additive model (no epistasis term):

```bash
plm-evolve propose-multi --embeddings emb.csv --wt WT.fasta --rounds round1.xlsx \
    --order 2 --top-singles 20 --top 24 --out out_dir/
```

`--order 2` proposes doubles, `--order 3` triples. `--top-singles K` restricts
stacking to the best K single mutants. The additive predictor is sample
efficient on additive-dominated landscapes; its edge shrinks as epistasis grows.

## Step 4: iterate

Add each measured round to `--rounds` and repeat steps 2 and 3:

```bash
plm-evolve rank --embeddings emb.csv --wt WT.fasta \
    --rounds round1.xlsx round2.xlsx round3.xlsx --acquisition ucb --top 12
```

## Library API

```python
import pandas as pd
from plm_evolve import ESM2Extractor, run_round

emb = ESM2Extractor().embed_fasta("single_mutants.fasta")
outcome = run_round(
    emb, ["round1.xlsx"], "WT.fasta", top_n=12, acquisition="ucb", beta=1.0
)
print(outcome.next_round)
```

## In-silico method comparison

The research layer replays the same loop against an oracle landscape (a
synthetic landscape, a generic `variant,activity` CSV, or a ProteinGym DMS) so
strategies can be compared without lab cost:

```bash
plm-evolve simulate --landscape landscape.csv --embeddings emb.csv \
    --strategies greedy random additive single_round --rounds 6 --seeds 10
```

The Python API in `plm_evolve.benchmark` (oracle, strategies, simulate, metrics,
ablation) gives finer control, plus ablation drivers for pooling, acquisition,
batch diversity, multi-mutant, and multi-objective selection.

## Real-DMS validation

`scripts/validate_gap2_dms.py` and `scripts/spearman_pooling_probe.py` run the
loop on real EVOLVEpro DMS oracle data and compare embedding-pooling modes. The
factual numbers live in `notes/gap2_dms_validation.md`.

## Honest status and limits

- The acquisition, batch-diversity, multi-mutant, and multi-objective features
  are parity capabilities, not novel methods.
- `mutated_token` pooling showed no advantage over `mean` on the application
  metric in the small real-DMS pilot; `mean` is the default.
- A prospective wet-lab campaign and a larger GPU multi-protein study are open
  follow-ups, not part of this package.
