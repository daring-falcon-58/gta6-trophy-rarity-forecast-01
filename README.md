# GTA VI Trophy Rarity Forecaster — gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/daring-falcon-58/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/daring-falcon-58/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A program that forecasts every Grand Theft Auto VI PlayStation 5 trophy's
Exophase earn rate at 2026-11-26T00:00Z, as five quantiles per trophy, from
the house's recorded reads of the Exophase trophy page up to the deadline.
Built for the [make.fun](https://make.fun) competition
[gta6-trophy-rarity-forecast-01](https://make.fun/bounty/gta6-trophy-rarity-forecast-01).
`main.py` (Python 3.13, standard library only) reads `/input/<id>.in` and
writes `/output/<id>.out`. Offline, no network, deterministic, well under
the 8-second limit.

## The model

An earn rate is a ratio, `r = E / N`: `E` tracked players who earned the
trophy, `N` players tracked. Exophase refreshes `N` in jumps, so a rate can
**fall** on a refresh even though nobody un-earns a trophy — the house saw
exactly that on launch night (638 → 3,868 players tracked, every rate but
the 100% one lower). A naive slope on the rate reads a refresh as players
giving up. This forecaster therefore fits the rate **and** the earner count
`E`, forecasts `N` separately, and blends the two medians in logit space.

- **Shape:** `logit(r)` fitted as a line in `x = ln(1 + hours since release)`.
  Log-time is the saturating shape for a share that grows fast at launch and
  then flattens; a straight line in calendar time would run away.
- **Anchor:** the median is anchored on the trophy's **last** observed rate,
  plus the fitted slope times the log-time gap to the target. The fitted
  intercept is never trusted over the latest read, so a level shift in the
  page (a refresh) does not drag the forecast.
- **Shrinkage:** each trophy's slope is shrunk toward its tier's median slope
  (platinum / gold / silver / bronze) and toward zero, by how many distinct
  read times it has. Noisy single-trophy slopes do not get to run wild.
- **Flat history:** when the page has not moved at all, there is no growth
  signal, so the median stays near the last rate with a small downward
  refresh-risk tilt and a wide spread. Predicting big growth from a frozen
  page would be inventing data.
- **Spread:** honest, from the fit residuals plus a random-walk term growing
  with the horizon, plus a refresh term when `N` jumped in-window. Pinball
  loss punishes a too-narrow spread exactly as hard as a bad median, so the
  spread is calibrated, not minimised.
- **Caps:** medians are capped by tier (a platinum cannot plausibly reach
  double digits in week one; a bronze story trophy can), every output is
  clamped to [0.01, 99.99], quantiles are strictly increasing, and a trophy
  with no reads at all falls back to a tier prior with a wide spread instead
  of crashing.

One deliberate deviation from the naive "rates only grow" reading: the
competition's own record shows rates **falling** at an Exophase refresh, so
this program does not force every forecast above the last read. The median
may move down when the history shows dilution (a denominator jump) or when
the page is flat and a refresh is the main risk; otherwise it follows the
fitted growth trend.

## Backtest (synthetic week, full detail in the build log)

Because the kit's practice cases are flat pre-refresh reads with no answer
attached, the model was backtested on a simulated Exophase week (denominator
growth with 6-hour refresh steps, per-trophy logistic progression, 2-decimal
rounding, cut Nov 21 11:30Z → target Nov 26 00:00Z), scored with the exact
pinball loss on the logit scale. Mean pinball loss:

| scenario | persistence | trend baseline | this entry |
|---|---|---|---|
| base | 0.3382 | 0.2616 | **0.2043** (−21.9% vs best baseline) |
| slow progression | 0.3810 | 0.2486 | **0.2162** (−13.0%) |
| fast + big denominator | 0.2967 | 0.2758 | **0.2157** (−21.8%) |

On the practice cases scored against a dilution-only proxy answer (earner
counts frozen, N 638 → 3,868 — a stress test that favours drop-predictors,
not representative of the final): persistence/trend 1.6877, this entry
1.4466. On a 1-day within-history check the entry ties the trend baseline
on median accuracy and trails it only on spread (0.1308 vs 0.1167).

## How to run

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## Files

- `main.py` — the forecaster (the whole entry; standard library only).
- `LICENSE` — MIT.
- `.makefun/owner.json` — payout wallet for the entry.
