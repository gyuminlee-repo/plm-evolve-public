# Contributing

## Development setup

Install the package with the dev extra (an editable install plus pytest):

```bash
pip install -e ".[dev]"
```

## Running the tests

The test suite is synthetic and needs no model download and no network access:

```bash
pytest -q
```

A change is ready for review once the full suite passes.

## Conventions

- **Independent implementation.** This package reimplements a published method. Do not copy
  source, headers, or license blocks from other projects; cite the method.
- **Opt-in, defaults unchanged.** Any new behavior is opt-in. The default
  production path (`run_round` greedy mean-pool, `embed --pooling mean`) stays
  byte-identical, and a test asserts that invariant.
- **Honest ablations.** A research-layer capability ships with an ablation that
  marks where it helps AND where it does not. No rigged demonstrations: a
  capability that wins everywhere, including the regime where the baseline ought
  to tie, is a red flag to investigate.
- **Fail-fast.** No silent fallbacks. A missing required input raises a clear
  error rather than defaulting to zero.
- **Parity is not novelty.** The acquisition, batch-diversity, multi-mutant, and
  multi-objective features are parity with prior work (FolDE, ALDE,
  MULTI-evolve); docstrings frame them as capabilities, not novel claims.

## Layout

- `plm_evolve/` production core (embeddings, variants, rounds, regressor,
  evolve, cli).
- `plm_evolve/benchmark/` research and benchmark layer (oracle, strategies,
  simulate, metrics, ablation, multimutant, multiobjective).
- `tests/` synthetic test suite.
- `scripts/` real-DMS validation drivers.
- `notes/` factual measurement notes.

## Commit messages

Follow `vA.BB.CC.DD: summary in English` (DD = fix/typo, CC = feature/refactor,
BB = new architecture). Increment the highest level that applies and reset the
lower levels to `00`.
