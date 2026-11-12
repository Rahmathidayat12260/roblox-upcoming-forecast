# Roblox Up-and-Coming Top-10 Forecaster

Entry for make.fun bounty `roblox-up-and-coming-forecast-01`. Python 3.13, stdlib
only, no network, deterministic (two runs produce byte-identical output).

## Run

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads `history.ndjson` (+ reads `task.json` only indirectly — the window is fixed
at 72 hourly slots per the rules), writes `forecast.json`.

## Model

Per-game continuous-time rank drift model, estimated from the chart's own history:

1. **Estimation.** Every pair of consecutive reads with delta ≤ 30 h yields, per
   start rank: mean rank change per hour (drift `mu(r)`), per-hour dispersion
   `sigma(r)`, and an hourly leave-the-chart hazard `h(r)` (exit rate converted
   with `-ln(1-rate)/delta`). Each is rank-smoothed (moving average, fallbacks to
   neighbours/global means when a rank is barely observed).
2. **Momentum tilt (mean reversion).** Each game's drift gets a tilt of
   `-0.15 ×` its recent observed rank velocity (lookback 2 reads). Recent risers
   revert; this consistently beat zero tilt and positive momentum in backtests.
3. **Horizon distribution.** For each horizon d = 1..72 h, a game at rank r lands
   at place k with a two-component Gaussian mixture over rank change
   (0.75 × N(r + mu·d, sigma·√d) + 0.25 × N(..., 2.5·sigma·√d)), times survival
   `(1-h)^d`.
4. **Coverage calibration.** Empirically, place k at t+d is held by a game that
   was already on the chart with probability `c_k(d)` (log-linear fit in d per
   place). Each place's named probabilities are rescaled to sum to `c_k(d)`; the
   remainder is left unnamed, which the scorer distributes over the entrants and
   unnamed chart games — the only honest way to cover games that have not yet
   been invented at forecast time.
5. **Mixture.** The scored window is 72 hourly slots, so final probabilities are
   the uniform mixture of the per-horizon distributions.
6. **Sparsity.** Only the top 12 candidates per place are named. Naming fewer
   games leaves more unnamed share, which empirically scores better than naming
   20–30 long shots (swept: 10/12/14/16 candidates; 12 best).
7. **top10** is the expected rank (current rank + drift over 36 h), survivors only,
   ordering ties broken by universe id (deterministic).

Probabilities are floored away from 0; every place's named mass stays below 1, so
a game that materializes on the chart always earns the unnamed share, never the
0.0001 floor.

## Backtest (walk-forward, public archive, 47 reads 2026-10-30 → 11-11)

For each cutoff read i (10 ≤ i ≤ 45): train on reads ≤ i, score against every
archive read in the following 72 h, replicating the official scorer exactly
(unnamed share, 0.0001 floor, mean over reads).

| model | mean log score |
|---|---|
| this model, all 36 walk-forward cutoffs | **−37.14** |
| house baseline (starter), same cutoffs | −46.85 |
| house baseline at the official practice cutoff | −43.51 (official: −43.512039) |
| **this model at the official practice cutoff** | **−33.75** |

Per horizon over all walk-forward window reads: < 12 h from cutoff −26.37
(47 reads), ≥ 12 h −39.68 (281 reads).

## Determinism

No randomness anywhere; ties broken by universe id. Two runs on the same input
produce identical sha256 (`e11839c2…` on the practice kit input). Runtime ≈ 1 s,
memory < 100 MB.
