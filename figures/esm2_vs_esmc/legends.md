# Figure and table legends

Publication-style legends for the ESM-2 vs ESM-C figure bundle. Numbers recompute
from the cached real deep mutational scanning embeddings (see README.md). The
shared core is the EVOLVEpro recipe: frozen mean-pool PLM embedding plus a
RandomForest top-layer, 20 random train/test splits unless noted.

## Figure 1. ESM-C improves few-shot ranking and per-plate yield

Swapping the frozen PLM from ESM-2 to ESM-C, under an identical RandomForest
top-layer, makes the few-shot model rank candidate variants more accurately and
fill each 96-well plate with more high-activity variants on four of five proteins,
even though ESM-C 300M is the smaller model and ESM-2 650M the larger.

(a) Held-out Spearman correlation between predicted and measured single-mutant
activity at a training budget of 95 variants (one 96-well plate), ESM-2 650M
(light) versus ESM-C 300M (dark), across five EVOLVEpro deep mutational scanning
proteins ordered by length. The smaller ESM-C 300M matches or exceeds the larger
ESM-2 650M on every protein, with the gap separated beyond 2 SE on kelsic,
stiffler, giacomelli, and cas12f; brenan shows no separation and is the protein
where neither PLM predicts. Bars are the mean over 20 splits, error bars are SE,
the dashed line marks zero. ESM-2 does not gain from 150M to 650M (Fig S1), so
the larger ESM-2 650M is not a weakened baseline.

(b) Held-out Spearman versus training budget (16 to 190 variants) for kelsic and
cas12f, with SE bands; dotted lines mark one plate (95) and two plates (190). The
ESM-C lead holds at one plate and narrows as more variants are measured.

(c) Plate-2 enrichment, defined as the count of genuinely top-95 variants captured
in the model top-95 plate minus the random-pick baseline 95^2/(n-95), as paired
bars per protein with the ESM-C minus ESM-2 delta annotated above each pair.
Enrichment removes the pool-size-dependent baseline so the bars compare across
proteins. ESM-C adds roughly 7 to 8 genuinely top variants per plate on kelsic and
stiffler and about 1 on giacomelli and cas12f, while on brenan both PLMs sit near
the random baseline.

The advantage is ranking accuracy and per-plate enrichment, not speed to the single
best variant. The evaluation is in-silico on the scanning oracle, not a prospective
wet-lab run. The additive baseline is degenerate on single-mutant landscapes and is
not plotted.

## Figure S1. Model size and embedding dimensionality do not explain the gain

Controls on kelsic at a 64-variant budget, family color (light is ESM-2, dark is
ESM-C).

(a) Held-out Spearman for ESM-2 150M and 650M and ESM-C 300M and 600M. The larger
ESM-2 650M does not exceed ESM-2 150M (saturation), and ESM-C 300M matches ESM-C
600M, so ESM-C 300M is the efficient choice and the gain tracks the model family,
not scale.

(b) The same four points against embedding dimensionality (640, 1280, 960, 1152).
ESM-2 650M at 1280 dimensions sits below ESM-C 300M at 960 dimensions, so feature
count does not account for the difference.

## Figure S2. Mutated-token pooling does not beat mean pooling

The GAP-2 pooling control. Held-out Spearman for mean pooling (gray) versus
mutated-token pooling (dark) at a 64-variant budget, for ESM-2 150M and 650M and
ESM-C 300M on kelsic plus ESM-C 300M on cas12f; bars are the mean over 20 splits,
error bars SE. Mutated-token pooling is not above mean pooling under either PLM, so
the pooling variant is closed and the predictive gain tracks the PLM family.

## Table 1. Per-protein ranking and per-plate enrichment

ESM-2 650M versus ESM-C 300M. Columns: protein length L, candidate pool size n,
held-out Spearman at a 95-variant budget (mean +/- SE) for each PLM, the paired
ESM-C minus ESM-2 gap, whether the gap exceeds 2 SE, the random plate baseline
95^2/(n-95), plate-2 precision@95 with enrichment over baseline for each PLM, and
the paired delta. Footnote: the size-matched kelsic pair is ESM-2 650M 0.288 versus
ESM-C 600M 0.528 at a 64-variant budget; enrichment is precision minus baseline; raw
precision is not comparable across proteins because the baseline depends on n.

## Table S1. Full held-out Spearman sweep

Every protein, PLM, and training budget (mean +/- SE over 20 splits).
