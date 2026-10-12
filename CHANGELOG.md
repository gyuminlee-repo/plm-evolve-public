# Changelog

Notable changes to plm-evolve. The git tag scheme is `vA.BB.CC.DD`
(DD = fix/typo, CC = feature/refactor, BB = new architecture).

## [0.01.20.00] WT anchored in the training set (matches EVOLVEpro)
- `run_round` now injects the WT row (`embeddings.loc["WT"]`, activity 1.0) into
  the training set by default (`include_wt_in_training=True`, `wt_activity=1.0`),
  matching the original EVOLVEpro, whose `create_iteration_dataframes` anchors WT
  at iteration 0 so the top-layer always trains on it. WT lands in embedding-index
  order like every other training row.
- This was the DOMINANT remaining gap vs EVOLVEpro, larger than the criterion:
  on an in-house 93-variant dataset (ESM-2 650M) omitting WT changed 9 of the top-12 proposals
  (3/12 overlap), and adding it makes plm-evolve reproduce EVOLVEpro's `top_layer`
  byte-for-byte (all 10547 predictions identical, Spearman 1.0) in the same sklearn.
  So exact EVOLVEpro reproduction needs: criterion=friedman_mse + WT-in-training +
  same embeddings + same sklearn version (a version bump alone moves ~2 of top-12).
- The "no measured variant matched" error now checks for a non-WT measured variant
  (WT alone is not a valid training set). Benchmark layer is unaffected (it never
  calls `run_round`). `include_wt_in_training=False` restores measured-only training.

## [0.01.19.00] RF top-layer criterion matches EVOLVEpro
- `TopLayerRegressor` now defaults `criterion="friedman_mse"`, the criterion the
  original EVOLVEpro top-layer sets explicitly (the sklearn default is
  `squared_error`). This makes the production next-round proposal byte-faithful
  to a reference EVOLVEpro run on identical embeddings, seed, and measured set.
  On an in-house round-1 dataset ~1/3 of the top-12 proposal changed between the two
  criteria; benchmark held-out Spearman shifts <=0.01 (mean ~0.002, largest at
  train size 16) with no ESM-C vs ESM-2 verdict flip. The esm2_vs_esmc figures,
  tables, and note were regenerated under `friedman_mse`. `criterion` is exposed
  as a constructor arg for `squared_error` opt-out.
- Test power: the rugged-landscape acquisition-ablation direction test runs 16
  seeds (was 8); under `friedman_mse` the exploration-vs-greedy margin needs the
  extra power to resolve. Assertions unchanged (not loosened).

## [0.01.09.03] ESM-2 vs ESM-C generalizes
- The ESM-C advantage holds on 4 of 5 EVOLVEpro DMS proteins (kelsic, stiffler,
  giacomelli, cas12f); brenan shows no separation (both PLMs near random). The
  mutated-token pooling is negative under ESM-C as well as ESM-2.

## [0.01.09.02] more DMS presets
- `scripts/validate_gap2_dms.py` gains stiffler/brenan/giacomelli dataset presets.

## [0.01.09.01] ESM-2 vs ESM-C comparison
- `scripts/compare_plm_spearman.py`: paired held-out Spearman across PLMs on a DMS
  oracle. ESM-C predicts activity better than ESM-2 under the core loop; ESM-C 300M
  is the efficient sweet spot (size saturates). Note: `notes/esm2_vs_esmc_comparison.md`.

## [0.01.09.00] ESM-C embedding backend
- `ESMCExtractor` (EvolutionaryScale ESM-C via the esm SDK) and a `make_extractor`
  factory; the PLM is swappable ESM-2 <-> ESM-C. Both backends share a pooling
  mixin (identical mean/delta/mutated_token semantics). `embed --model esmc_300m`.

## [0.01.08.02] distribution docs + packaging
- LICENSE (MIT), CHANGELOG, CONTRIBUTING, docs/USAGE.md, CITATION.cff; pyproject
  classifiers, a `[dev]` extra, and a repository url.

## [0.01.08.01]
- docs: README covers the production `rank --acquisition` flag and the
  `propose-multi` command.

## [0.01.08.00] production integration
- `run_round` and the `rank` CLI gain an opt-in
  `--acquisition {greedy,ucb,thompson,coverage}` (with `--beta`, `--penalty`,
  `--seed`). The default `greedy` path stays byte-identical (a test asserts it
  on both the ranked table and the next-round set).
- New `propose-multi` CLI: additive multi-mutant stacking from the best single
  mutants.
- The benchmark strategies and the regressor are reused unchanged.

## [0.01.07.00] GAP-5 multi-objective hook
- `benchmark/multiobjective.py`: `scalarize` (linear and chebyshev),
  `pareto_front`, and `compose_objectives` (combine two or more objective
  landscapes into one scalar oracle so the single-objective loop runs
  unchanged). v1 stays single-objective.

## [0.01.06.00] GAP-4 multi-mutant
- `benchmark/multimutant.py`: `enumerate_multimutants`, `AdditiveMultiPredictor`
  (purely additive, no epistasis term by design), `propose_multimutants`, and a
  generative oracle (`make_generative_landscape`).

## [0.01.05.00] GAP-3 batch diversity
- `CoverageBatchStrategy` (mutation-site coverage) and `ConstantLiarStrategy`
  (a FolDE-style parity baseline).

## [0.01.04.00] GAP-1 uncertainty-aware acquisition
- `TopLayerRegressor.predict_with_uncertainty` (per-tree mean and standard
  deviation), `UCBStrategy` (`beta=0` reduces exactly to greedy), and
  `ThompsonStrategy`.

## [0.01.03.00] GAP-2 real-DMS validation
- `scripts/validate_gap2_dms.py` and `scripts/spearman_pooling_probe.py`. The
  finding (`notes/gap2_dms_validation.md`): `mutated_token` shows no advantage
  over `mean` on the application metric where it is meaningful, on the two
  proteins and small models tested.

## [0.01.02.00] GAP-2 embedding pooling
- Opt-in pooling modes: `mean` (default, unchanged), `delta`, `mutated_token`.

## [0.01.01.00] in-silico simulation harness
- `benchmark/` oracle, baseline strategies, and sample-efficiency metrics
  (top-k recall, rounds-to-target, simple regret, Spearman).

## [0.01.00.00] independent core
- Frozen PLM embedding, top-layer RandomForest regressor, and the
  active-learning loop (`run_round`).
