"""GTA VI trophy-rarity forecaster for make.fun gta6-trophy-rarity-forecast-01.

Reads one case per file in /input (MAKEFUN_INPUT_DIR outside the grader) and
writes /output/<id>.out with five quantiles (0.05, 0.25, 0.5, 0.75, 0.95) of
each trophy's Exophase earn rate at the case target. Python 3.13, standard
library only, no network, deterministic.

Model — denominator-aware logistic growth in log-time, per trophy:
  * An earn rate is a ratio, r = E / N: E tracked players who earned the
    trophy, N players tracked. Exophase refreshes N in jumps, so a rate can
    FALL on a refresh even though nobody un-earns a trophy (the house saw
    exactly that on launch night: 638 -> 3,868 players, every rate but the
    100% one lower). A naive slope on the rate therefore reads a refresh as
    players giving up. This program fits the rate AND the earner count E,
    forecasts N separately, and blends the two medians in logit space.
  * Shape: logit(r) is fitted as a line in x = ln(1 + hours since release).
    Log-time is the saturating shape for a share that grows fast at launch
    and then flattens; a straight line in calendar time would run away.
  * Anchor: the median is anchored on the trophy's LAST observed rate, plus
    the fitted slope times the log-time gap to the target. The fitted
    intercept is never trusted over the latest read, so a level shift in the
    page (a refresh) does not drag the forecast.
  * Shrinkage: each trophy's slope is shrunk toward its tier's median slope
    (platinum / gold / silver / bronze) and toward zero, by how many distinct
    read times it has. Noisy single-trophy slopes do not get to run wild.
  * Flat history (no movement at all in the window): there is no growth
    signal, so the median stays near the last rate with a small downward
    refresh-risk tilt and a wide spread. Predicting big growth from a frozen
    page would be inventing data.
  * Spread: honest, from the fit residuals plus a random-walk term that
    grows with the horizon, plus a refresh term when N jumped in-window.
    Pinball loss punishes a too-narrow spread exactly as hard as a bad
    median, so the spread is calibrated, not minimised.
  * Caps: medians are capped by tier (a platinum cannot plausibly reach
    double digits in week one; a bronze story trophy can), every output is
    clamped to [0.01, 99.99], quantiles are strictly increasing, and a
    trophy with no reads at all falls back to a tier prior with a wide
    spread instead of crashing.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Spread of a rate's logit move, per square-root day (random-walk part).
BASE_SIGMA = 0.45
# Hard limits on the logit spread actually emitted.
SIGMA_MIN = 0.30
SIGMA_MAX = 1.50
# How far a fitted slope may move a median, in logit units.
MAX_LOGIT_MOVE = 2.5
# Tier ceilings for the week-one median (percent). A trophy's median is
# never forecast above its tier's ceiling.
TIER_CEIL = {"platinum": 3.0, "gold": 25.0, "silver": 60.0, "bronze": 90.0}
# Tier prior medians (percent) for a trophy with no reads at all.
TIER_PRIOR = {"platinum": 0.3, "gold": 2.0, "silver": 6.0, "bronze": 12.0}
# Slope cap, in logit units per unit of log-time.
SLOPE_CAP = 1.2


def clamp_rate(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    c = clamp_rate(pct)
    return math.log(c / (100.0 - c))


def inv_logit(l):
    if l >= 0:
        return 100.0 / (1.0 + math.exp(-l))
    e = math.exp(l)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up (matches the house starter's rounding)."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value == int(value) else value


def parse_time(s):
    """ISO-8601 to an aware datetime; returns None when unparseable."""
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def ols(xs, ys):
    """Least-squares line y ~ a + b*x. Returns (a, b, resid_var, n)."""
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 0.0, 0
    if n == 1:
        return ys[0], 0.0, 0.0, 1
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return my, 0.0, 0.0, n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    resid = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / n
    return a, b, resid, n


def median_of(values):
    vals = sorted(values)
    if not vals:
        return 0.0
    m = len(vals) // 2
    if len(vals) % 2:
        return vals[m]
    return 0.5 * (vals[m - 1] + vals[m])


def forecast_trophy(series, tier, x_target, horizon_days):
    """series: list of (x, rate, n_players) oldest first, rate in percent.

    Returns (median_percent, sigma_logit).
    """
    ceil = TIER_CEIL.get(str(tier).lower(), 90.0)
    prior = TIER_PRIOR.get(str(tier).lower(), 12.0)
    if not series:
        return prior, 1.40

    last_x, last_rate, _last_n = series[-1]
    last_logit = logit(last_rate)

    # --- window: the last 24 hours of reads (10-minute reads), or the
    # whole series when it is shorter. A longer window averages in stale
    # slopes: the growth curve is concave in log-time, so the recent slope
    # is the one that extrapolates.
    if len(series) > 144:  # > 24h at 10-minute reads
        window = series[-144:]
    else:
        window = list(series)

    xs = [s[0] for s in window]
    lg = [logit(s[1]) for s in window]
    flat = (max(lg) - min(lg)) < 0.02 if lg else True

    # --- rate fit: logit(rate) on log-time.
    _a, b_raw, resid, _n = ols(xs, lg)
    distinct = len(set(round(x, 6) for x in xs))
    shrink = distinct / (distinct + 2.0) if distinct else 0.0
    b_rate = max(-SLOPE_CAP, min(SLOPE_CAP, b_raw * shrink))

    # --- denominator check: did N jump inside the window?
    ns = [s[2] for s in window if s[2] is not None and s[2] > 0]
    n_jumped = False
    max_step = 0.0
    if len(ns) >= 2:
        for prev, cur in zip(ns, ns[1:]):
            step = (cur - prev) / prev if prev > 0 else 0.0
            max_step = max(max_step, step)
        n_jumped = max_step >= 0.10 or ns[-1] >= 1.25 * ns[0]
    # Only a BIG single jump counts as refresh uncertainty: regular small
    # steps are already inside the rate series and need no extra spread.
    big_refresh = max_step >= 0.25

    median_logit = last_logit + max(
        -MAX_LOGIT_MOVE, min(MAX_LOGIT_MOVE, b_rate * (x_target - last_x))
    )

    # --- earner-count fit, used only when N actually moved: E cannot fall,
    # so its slope is floored at zero, and N is forecast on its own curve.
    if n_jumped and len(ns) >= 3:
        es = []
        for (x, rate, n) in window:
            if n is not None and n > 0:
                es.append((x, math.log(max(rate, 0.01) / 100.0 * n + 0.5)))
        if len(es) >= 3:
            ex = [e[0] for e in es]
            ey = [e[1] for e in es]
            _ae, b_e, _re, _ne = ols(ex, ey)
            b_e = max(0.0, min(0.35, b_e))
            e_last = math.log(max(last_rate, 0.01) / 100.0 * (ns[-1]) + 0.5)
            e_tgt = math.exp(e_last + b_e * (x_target - last_x))
            nx = [s[0] for s in window if s[2] is not None and s[2] > 0]
            ny = [math.log(s[2]) for s in window if s[2] is not None and s[2] > 0]
            _an, b_n, _rn, _nn = ols(nx, ny)
            # N growth in log-time is concave (launch rush decays), so a
            # straight-line extrapolation overstates it; cap the rate and
            # cap the total N growth over the horizon at 1.5x.
            b_n = max(0.0, min(0.20, b_n))
            n_tgt = ns[-1] * math.exp(b_n * (x_target - last_x))
            n_tgt = min(n_tgt, 1.5 * ns[-1])
            if n_tgt > 0 and e_tgt > 0:
                rate_b = clamp_rate(100.0 * e_tgt / n_tgt)
                # The count fit is the cleaner signal when N jumps, but it
                # carries its own N-forecast error, so it gets 35% weight.
                median_logit = 0.65 * median_logit + 0.35 * logit(rate_b)

    # --- flat history: no growth signal; stay near the last rate, small
    # downward refresh-risk tilt, wide spread (handled in sigma below).
    if flat and horizon_days >= 1.0:
        median_logit = last_logit - 0.25 * min(1.0, horizon_days / 4.5)

    median = clamp_rate(inv_logit(median_logit))
    median = min(median, ceil)
    if last_rate >= 99.99:
        median = 99.99

    # --- spread: random walk + fit residual + refresh term, in quadrature.
    sigma2 = (BASE_SIGMA * math.sqrt(max(horizon_days, 0.0))) ** 2
    if not flat:
        sigma2 += min(resid, 0.05)
    if big_refresh:
        sigma2 += 0.35 ** 2
    if flat:
        sigma2 += 0.45 ** 2
    sigma = math.sqrt(sigma2)
    sigma = max(SIGMA_MIN, min(SIGMA_MAX, sigma))
    return median, sigma


def forecast_case(case):
    trophies = case.get("trophies") or []
    history = case.get("history") or []
    horizon_days = max(0.0, float(case.get("horizon_minutes") or 0) / 1440.0)

    # Release origin: the game's release date at midnight UTC, else the
    # first read's slot, else the case's own checkpoint.
    origin = None
    game = case.get("game") or {}
    rel = game.get("release_date")
    if rel:
        origin = parse_time(str(rel) + "T00:00:00Z")
    if origin is None and history:
        origin = parse_time(history[0].get("slot"))
    if origin is None:
        origin = parse_time(case.get("at")) or datetime(
            2026, 11, 19, tzinfo=timezone.utc
        )
    target = parse_time(case.get("target")) or origin
    x_target = math.log1p(max(0.0, (target - origin).total_seconds() / 3600.0))

    # Per-trophy series, oldest first.
    series = {i: [] for i in range(len(trophies))}
    for read in history:
        slot = parse_time(read.get("slot"))
        if slot is None:
            continue
        x = math.log1p(max(0.0, (slot - origin).total_seconds() / 3600.0))
        rates = read.get("rates") or []
        n = read.get("players_tracked")
        if not isinstance(n, (int, float)) or n is None or n <= 0:
            n = None
        for i in range(len(trophies)):
            if i >= len(rates):
                break
            r = rates[i]
            if r is None or not isinstance(r, (int, float)):
                continue
            if not math.isfinite(r) or r < 0 or r > 100:
                continue
            series[i].append((x, float(r), n))

    # First pass: raw slopes per trophy, for the tier-median shrinkage.
    raw_slopes = {}
    for i, trophy in enumerate(trophies):
        s = series[i]
        if len(s) >= 3:
            _a, b, _r, _n = ols([p[0] for p in s], [logit(p[1]) for p in s])
            raw_slopes[i] = max(-SLOPE_CAP, min(SLOPE_CAP, b))
    tier_slopes = {}
    for i, trophy in enumerate(trophies):
        tier = str(trophy.get("tier", "")).lower()
        tier_slopes.setdefault(tier, []).append(raw_slopes.get(i, 0.0))
    tier_median = {t: median_of(v) for t, v in tier_slopes.items()}

    rows = []
    for i, trophy in enumerate(trophies):
        median, sigma = forecast_trophy(
            series[i], trophy.get("tier", ""), x_target, horizon_days
        )
        # Blend the slope toward the tier median: a trophy whose own slope
        # is an outlier gets pulled 25% of the way to its tier's typical
        # slope, in median-logit terms, when it actually has a slope.
        if i in raw_slopes:
            tier = str(trophy.get("tier", "")).lower()
            tm = tier_median.get(tier, 0.0)
            pull = 0.15 * (tm - raw_slopes[i]) * max(0.0, x_target - (series[i][-1][0] if series[i] else x_target))
            if pull:
                median = clamp_rate(
                    inv_logit(logit(median) + max(-0.25, min(0.25, pull)))
                )
        med_logit = logit(median)
        qs = []
        for z in Z:
            qs.append(round4(clamp_rate(inv_logit(med_logit + z * sigma))))
        # Strictly increasing quantiles after rounding, including at the
        # 99.99 ceiling (a trophy at 100% would otherwise tie there).
        # Work in integer units of 0.0001 so the ordering is exact.
        units = [max(1, min(999900, int(round(float(q) * 10000)))) for q in qs]
        for j in range(3, -1, -1):
            if units[j] >= units[j + 1]:
                units[j] = units[j + 1] - 1
        for j in range(1, 5):
            if units[j] <= units[j - 1]:
                units[j] = units[j - 1] + 1
        units = [max(1, min(999900, u)) for u in units]
        qs = [u / 10000.0 if (u / 10000.0) != int(u / 10000.0) else int(u / 10000.0)
              for u in units]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
