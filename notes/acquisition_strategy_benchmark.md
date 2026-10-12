# Acquisition strategy benchmark (greedy vs exploration)

## Question

Does any exploration acquisition (`ucb`, `thompson`, `coverage`) beat plain
`greedy` at finding the best variant fast, in the iterative cumulative-training
loop, on real DMS fitness landscapes?

## Setup

- Driver: `scripts/sim_acq_compare.py` (in-silico `run_panel` over real DMS oracles).
- 3 DMS landscapes, ESM-C 300M mean-pool embeddings: kelsic, giacomelli, cas12f.
- `n_init=16, per_round=12, rounds=8, seeds=20`, target = 95th-percentile activity.
- Strategy defaults: ucb `beta=1.0`, coverage `base=greedy penalty=1.0`, thompson seeded.
- Primary metric: best activity found by the final round (higher is better),
  averaged over 20 seeds. Significance: `z = delta / SE` across seeds.

## Findings (best-found vs greedy)

| dataset | greedy | ucb | thompson | coverage |
| --- | --- | --- | --- | --- |
| kelsic | 1.0871 | -0.0114 (z=-1.2) | -0.0348 (z=-4.2) | +0.0067 (z=+0.7) |
| giacomelli | 1.5780 | -0.0017 (z=-0.1) | +0.0304 (z=+1.3) | -0.0022 (z=-0.2) |
| cas12f | 1.4462 | +0.0000 (z=+0.0) | -0.0436 (z=-1.6) | +0.0043 (z=+0.2) |

## Verdict

- No exploration strategy SIGNIFICANTLY beats greedy on any landscape; every
  positive delta has `|z| < 1.3` (within seed noise, below the |z|>2 bar).
- thompson is significantly WORSE on kelsic (z=-4.2) and trends worse on cas12f
  (z=-1.6, not significant at this budget); avoid it.
- ucb and coverage are statistically indistinguishable from greedy on best-found.
- greedy is the robust default; the empirical upside of switching is not measurable here.

## Caveats

- Dense sampling: pools are 350 to 500 variants and the budget measures about 22
  to 32 percent of the pool. Exploration matters most when a tiny fraction of a
  LARGE space is sampled; a large single-mutant scan at the same small budget is
  far sparser and is untested here.
- Default beta and penalty (untuned). Coverage here uses `base=greedy`, unlike
  the rank/GUI path (`base=ucb`).
- Proxy single-mutant DMS landscapes, not a multi-round wet-lab campaign.
- Metric is best-found at a fixed 8-round budget.

## Reproduce

```bash
python3 scripts/sim_acq_compare.py --dataset kelsic --model esmc_300m --seeds 20
```
