# Performance notes

Backtest method: walk-forward over the public archive (github.com/Kenny-Dhub/
roblox-snapshots, git history of `data/discover/latest.json` reconstructed into 47
full Up-and-Coming chart reads, 2026-10-30T07:36Z → 2026-11-11T15:13Z). For each
cutoff i, the model trains only on reads ≤ i and is scored exactly like the
official scorer (per place: named p, else unnamed share split over unnamed chart
games, floored at 0.0001; log summed over places 1–10, mean over the reads in the
following 72 h).

## Headline numbers

| model | mean log score (36 cutoffs) | practice cutoff (2026-11-07T20:53Z) |
|---|---|---|
| **this entry** | **−37.137** | **−33.747** |
| house baseline (starter/main.ts replication) | −46.847 | −43.512 (matches official −43.512039) |

Typical probability per place (exp(score/10)): 0.0243 (model) vs 0.0093 (baseline).

## Per horizon (all walk-forward window reads)

| distance from cutoff | reads | mean log score |
|---|---|---|
| < 12 h | 47 | −26.37 |
| ≥ 12 h | 281 | −39.68 |

## Ablations (mean over the last 9 walk-forward cutoffs, closest to final conditions)

| change | score |
|---|---|
| full model (mw=−0.15, tail 0.25×2.5σ, 12 candidates, coverage(d)) | −34.39 |
| no mean-reversion tilt (mw=0) | −41.83 |
| positive momentum (mw=+1.0) | −40.28 |
| single Gaussian (no heavy-tail component) | −38.82 |
| 20 named candidates per place | −39.44 |
| 50 named candidates per place | −44.76 |
| constant coverage (no d-dependence) | −39.24 |
| hazard as linear rate (not −ln(1−rate)/δ) | ≈ −39.4 |

Key findings:

- **Mean reversion beats momentum.** Games that just climbed the chart fall back;
  the best tilt is a negative fraction of recent rank velocity.
- **Sparsity wins.** Unnamed share is the only way to cover future entrants
  (games that do not exist yet at forecast time). Naming 12 candidates beats
  naming 30–97 because it leaves more share for them.
- **Heavy tails help.** Rank changes are leptokurtic; a 25% wide-σ component cuts
  loss when a game jumps 5+ places in a few hours.
- **Coverage calibration matters.** Rescaling named mass to the empirically
  estimated probability that an "old" game still holds each place at each horizon
  prevents double counting of independent per-game marginals.

## Constraint checks

- Python 3.13, stdlib only, no network, no randomness → two runs byte-identical
  (sha256 e11839c237e72f65a0dc92dd0c9ec2696a8468fc63d6ad567ffa469958135b92 on the
  practice kit input).
- Runtime ≈ 0.6 s on 34 history reads (limit 300 s); memory < 100 MB (limit
  1024 MB); forecast.json ≈ 6.5 KB (limit 1 MiB).
