# EVOLVEpro vs plm-evolve under ESM-2 vs ESM-C (real-DMS pilot)

The original question: swap the EVOLVEpro PLM for a newer one (ESM-C) and see if
the active-learning campaign improves. plm-evolve's default loop (greedy mean-pool
+ RandomForest top-layer) IS the EVOLVEpro core, faithfully reimplemented, so the
"EVOLVEpro vs plm-evolve" axis is method-equivalence; the informative variable is
the PLM. This note records the ESM-2 vs ESM-C numbers measured through plm-evolve's
validated harness.

## Setup
- Core = greedy mean-pool embedding + RandomForest(n_estimators=100, random_state=1),
  the algorithm EVOLVEpro and plm-evolve share.
- Criterion: the tables below are the `criterion="friedman_mse"` values (the
  current plm-evolve default, matching original EVOLVEpro; figures and tables
  regenerated from the cached embeddings). Under the prior sklearn default
  `squared_error` every cell differs by <=0.01 (mean ~0.002, largest at train
  size 16) with no ESM-C vs ESM-2 verdict flip, so the criterion choice does not
  change any finding here.
- Oracle = real EVOLVEpro DMS (kelsic/infA L=72, n=500 seed-fixed subsample;
  cas12f/AsCas12f L=422, n=350).
- Primary metric = held-out Spearman(predicted, true) on the unmeasured pool, RF
  trained on a random subset (train sizes 16/32/64, 20 seeds). Paired by split seed
  across models so the held-out set is identical and split noise cancels.
- ALL embeddings regenerated FRESH under one environment (transformers 4.48.1,
  torch 2.12.1) so ESM-2 and ESM-C share library versions (no skew).
- Matched scale: ESM-2 650M vs ESM-C 600M. Medium scale: ESM-2 150M vs ESM-C 300M.

## Held-out Spearman (kelsic, n=500, mean-pool)
| model | ts=16 | ts=32 | ts=64 |
|---|---|---|---|
| ESM-2 150M | 0.153 +/- 0.106 | 0.225 +/- 0.095 | 0.305 +/- 0.068 |
| **ESM-C 300M** | **0.444 +/- 0.108** | **0.505 +/- 0.051** | **0.537 +/- 0.035** |
| ESM-2 650M | 0.100 +/- 0.087 | 0.172 +/- 0.086 | 0.287 +/- 0.075 |
| **ESM-C 600M** | **0.381 +/- 0.137** | **0.461 +/- 0.063** | **0.525 +/- 0.050** |

Paired gap (Spearman_ESM-C minus Spearman_ESM-2), separated iff |gap| > 2*SE:
- medium (150M vs 300M): +0.291 / +0.280 / +0.231 -> ESM-C better (>8 SE).
- matched (650M vs 600M): +0.281 / +0.290 / +0.238 -> ESM-C better (>10 SE).

## Held-out Spearman (cas12f, n=350, mean-pool, medium pair)
| model | ts=16 | ts=32 | ts=64 |
|---|---|---|---|
| ESM-2 150M | 0.154 +/- 0.093 | 0.162 +/- 0.078 | 0.214 +/- 0.060 |
| **ESM-C 300M** | **0.177 +/- 0.082** | **0.209 +/- 0.076** | **0.278 +/- 0.050** |

Paired gap: +0.023 / +0.047 / +0.064 -> ESM-C better at ts=16 and ts=64 (ts=32 borderline).

## Generalization across 5 EVOLVEpro DMS proteins (medium pair)

Held-out Spearman, ESM-2 150M vs ESM-C 300M, n=350-500, train size 32:

| protein (L) | ESM-2 150M | ESM-C 300M | paired gap | verdict |
|---|---|---|---|---|
| kelsic (72) | 0.225 | 0.505 | +0.28 | ESM-C much higher |
| stiffler (286) | 0.306 | 0.556 | +0.25 | ESM-C much higher |
| giacomelli (393) | 0.251 | 0.403 | +0.15 | ESM-C higher |
| cas12f (422) | 0.162 | 0.209 | +0.05 | ESM-C higher |
| brenan (360) | 0.078 | 0.044 | -0.03 | no separation |

ESM-C is more activity-predictive on 4 of 5 proteins (large on kelsic/stiffler,
modest on cas12f). brenan is the exception: both PLMs sit near random
(Spearman ~0.05-0.08), so brenan/MAPK1 is a landscape neither frozen PLM predicts
well, not an ESM-2 advantage.

## GAP-2 mutated-token pooling under ESM-C

The mutated-token pooling that was negative under ESM-2 is ALSO negative under
ESM-C. Held-out paired Spearman (mean vs mutated_token): kelsic mean separated
HIGHER at small train sizes under both esmc_300m (gap -0.11/-0.09) and esmc_600m
(-0.09/-0.06); cas12f no separation. So mutated-token does not help under either
PLM. The pooling trick is closed; the PLM family is what moves the number.

## Findings
1. ESM-C predicts DMS activity BETTER than ESM-2 on BOTH proteins (direction
   consistent). The margin is protein-dependent: large on kelsic (~+0.27 Spearman,
   0.50 vs 0.23 at ts=32) and modest on cas12f (~+0.05).
2. ESM-C 300M ~= ESM-C 600M (0.537 vs 0.525 at ts=64): the advantage comes from the
   model FAMILY, not size. ESM-C 300M is the sweet spot (it matches 600M and beats
   every ESM-2).
3. ESM-2 650M is NOT better than ESM-2 150M (0.287 vs 0.305 at ts=64): bigger ESM-2
   saturates for this few-shot frozen-embedding regression, consistent with the
   model-scale analysis.
4. Dimensionality is ruled out as the driver: ESM-2 650M (1280-d) is WORSE than
   ESM-C 300M (960-d), so the ESM-C edge is signal quality, not feature count.

## recall@10 (secondary; the application metric, underpowered here)
greedy_mean recall@10: kelsic ESM-2 150M 0.16 vs ESM-C 300M 0.16 (tie), ESM-2 650M
0.32 vs ESM-C 600M 0.25 (within sd ~0.19); cas12f ESM-2 150M 0.28 vs ESM-C 300M 0.36.
recall@k on kelsic's thin beneficial needle is high-variance (sd ~0.15-0.19), so it
does not resolve the PLM effect; the held-out Spearman does.

## 96-well plate scale (train=95/190 Spearman + plate-2 precision@95)

For the 95-variants-per-plate workflow (WT well + 95 test variants), held-out
Spearman at train=95 (one plate) and 190 (two plates), ESM-2 150M vs ESM-C 300M:

| protein | ESM-2 ts95 | ESM-C ts95 | ESM-2 ts190 | ESM-C ts190 |
|---|---|---|---|---|
| kelsic | 0.339 | 0.571 | 0.447 | 0.625 |
| stiffler | 0.499 | 0.672 | 0.587 | 0.713 |
| giacomelli | 0.364 | 0.467 | 0.440 | 0.511 |
| cas12f | 0.260 | 0.316 | 0.325 | 0.384 |
| brenan | 0.099 | 0.100 | 0.086 | 0.116 |

The ESM-C lead holds at one plate (4 of 5), and shrinks as plates accumulate.

Plate-2 selection (committed driver `scripts/plate_selection.py`, pool-def,
30 seeds, default_rng(1000+s)): plate1 = 95 random training wells, plate2 =
model top-95 of the remaining pool. precision@95 = count of the pool true-top-95
captured in plate2. Raw precision@95 is NOT comparable across proteins because
the random-pick expectation 95^2/(n-95) is 22.3 for kelsic (n=500) and 35.4 for
the n=350 proteins. The comparable quantity is enrichment over that baseline
(precision minus baseline) and the paired ESM-C minus ESM-2 delta (baseline
cancels within a protein).

| protein | n | baseline | ESM-2 prec / enr | ESM-C prec / enr | delta |
|---|---|---|---|---|---|
| kelsic | 500 | 22.3 | 35.6 / +13.3 | 42.5 / +20.2 | +6.9 |
| stiffler | 350 | 35.4 | 56.1 / +20.7 | 63.7 / +28.3 | +7.6 |
| giacomelli | 350 | 35.4 | 46.6 / +11.2 | 47.5 / +12.1 | +0.9 |
| cas12f | 350 | 35.4 | 45.0 / +9.6 | 45.7 / +10.3 | +0.7 |
| brenan | 350 | 35.4 | 36.4 / +1.0 | 38.7 / +3.3 | +2.3 |

ESM-C adds ~7-8 genuinely-top variants per plate on kelsic/stiffler, ~1 on the
weaker-signal giacomelli/cas12f, ~2 on brenan (whose ~3-well enrichment is near
chance under both PLMs), broadly in line with the Spearman gaps. The earlier note values
(27/34 and so on) came from an ad-hoc routine that was never committed; the
reproducible driver above supersedes them.

## FolDE predictor (Bradley-Terry MLP) vs EVOLVEpro-core RF

FolDE (arXiv 2510.24053, JBEI Foldy) keeps the same frozen-PLM mean-pool input
but swaps the EVOLVEpro RandomForest top-layer for a Bradley-Terry ranking MLP
ensemble. Running FolDE's actual predictor (folde.few_shot_models.TorchMLPFewShotModel,
ranking loss, no naturalness warm-start) on the SAME cached embeddings and the
SAME 20-seed train=95 splits gives this held-out Spearman. RF = EVOLVEpro-core.
The FolDE RandomForestFewShotModel harness check that reproduced this column
(kelsic ESM-C 0.572) used the sklearn default `squared_error`; plm-evolve now
defaults to `friedman_mse` (the true EVOLVEpro criterion), which shifts these RF
columns by ~0.001. The exact byte-for-byte match held under `squared_error`; the
RF-vs-BT comparison below is unaffected at this precision:

| protein | ESM-2 RF | ESM-2 BT | ESM-C RF | ESM-C BT |
|---|---|---|---|---|
| kelsic | 0.337 | 0.543 | 0.572 | 0.629 |
| stiffler | 0.498 | 0.543 | 0.672 | 0.700 |
| giacomelli | 0.366 | 0.369 | 0.467 | 0.424 |
| cas12f | 0.263 | 0.264 | 0.316 | 0.298 |
| brenan | 0.101 | 0.141 | 0.099 | 0.102 |

The Bradley-Terry MLP beats the RandomForest on all five proteins under the
weaker ESM-2 embeddings (large on kelsic, +0.207), but under the stronger ESM-C
embeddings the gain shrinks and reverses on giacomelli (-0.043) and cas12f
(-0.019). So the predictor upgrade helps most when the embedding is weak; with
ESM-C the RandomForest stays competitive and wins on two of five proteins. This
is FolDE without its naturalness warm-start (which mostly helps the early rounds).
Numbers from scripts/folde_compare.py (FolDE code at jbei/foldy), saved to
figures/esm2_vs_esmc/data/folde_compare.json.

## Caveats
- Five CPU proteins (medium pair on all five; matched 650M/600M on kelsic only).
  The direction is consistent (4 of 5 favor ESM-C), but the magnitude varies a
  lot by protein and one protein (brenan) shows no PLM separation.
- The matched 650M/600M scale was run on kelsic only.
- The `esm` SDK install mutated the shared environment (transformers 4.57.6 -> 4.48.1,
  torch 2.10 -> 2.12); plm-evolve's 142 tests pass under it, and both PLM arms ran
  under that one environment so the comparison is internally consistent.

## Bottom line
Swapping the EVOLVEpro PLM from ESM-2 to ESM-C improves the core loop's predictive
quality on 4 of 5 tested proteins (large on kelsic/stiffler, modest on cas12f;
brenan a hard landscape where both PLMs sit near random), with ESM-C 300M the
efficient sweet spot. This is a POSITIVE, generalizing result for the PLM swap, in
contrast to the GAP-2 mutated-token pooling trick, which is negative under BOTH
PLMs. A GPU campaign over the full mutant sets and the larger ESM-C is the next
step to firm up the per-protein margin.
