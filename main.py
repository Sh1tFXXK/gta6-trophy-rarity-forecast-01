"""Forecast GTA VI trophy earn rates on Exophase, one case at a time.

Entry for make.fun bounty gta6-trophy-rarity-forecast-01. The program reads
every ``*.in`` case file in the input directory (``$MAKEFUN_INPUT_DIR`` or
``/input``) and writes one ``*.out`` forecast per case to the output
directory (``$MAKEFUN_OUTPUT_DIR`` or ``/output``). Standard library only,
no network, fully deterministic: the same case always yields the same bytes.

What a case holds
-----------------
``history`` is every valid read the house made of Exophase's GTA VI trophy
page before the case checkpoint, oldest first. Each read lists the players
tracked and every trophy's earn rate (percent of tracked players who hold
it). The target is some hours or days after the checkpoint, and the answer
is the page as Exophase shows it at the target, as of its last refresh.

How the forecast is built
--------------------------
An earn rate is a ratio, ``rate = 100 * earned / tracked``, and both halves
move for different reasons, so the program models them separately and then
recombines them:

* ``tracked`` grows as new players sync their profiles with Exophase.
  Adoption is concave in time, so log(tracked) is fitted against
  ``log(1 + hours since release)`` and extrapolated.
* ``earned`` for a trophy only ever grows: players keep earning it. Its
  log count is fitted the same way, per trophy, anchored on the last read.
* The ratio of the two extrapolations is one median estimate.

A second, direct estimate fits the logit's own curve: the logit of the rate
against the same log-time axis, per trophy, anchored on the last read. The
final median is the average of the two logit estimates, which hedges the
case where either decomposition misbehaves (for instance a refresh that
moves the denominator faster than the numerator).

Both extrapolations are saturation-aware. A trophy's curve levels off --
the players who will ever finish a chapter mostly have -- so a straight
line in log-time would overshoot badly over a multi-day horizon. The
log-time trend is therefore damped by how far past the last observation
the target sits, and a separate fit estimates each trophy's ceiling
directly (rate = ceiling * (1 - exp(-k * hours))), which is the component
that carries the growth the damping takes away.

Three facts about the data shape the rest:

* Exophase refreshes in steps, hours apart. Between refreshes the page is
  byte-identical, so consecutive identical reads carry no information and
  are collapsed to one point before any fit; otherwise the flat stretch
  would dominate the regression.
* Rare trophies have few holders, so their per-trophy slopes are noisy.
  Each slope is shrunk toward the median slope of all trophies, with the
  weight growing in the number of distinct reads that trophy has.
* A long flat history (launch night, before the first refresh) means the
  next refresh is overdue: the median is then shifted down by the dilution
  a refresh brings -- new tracked players hold fewer trophies -- instead
  of naively repeating the last rate.

The spread is honest about what is unknown: a random-walk term growing
with the square root of the horizon, the residual scatter of the fit, the
slope's own standard error projected to the target, and one refresh step
of uncertainty, since the answering read is itself whatever the last
refresh before the target happened to show. Quantiles are the median
logit plus the standard normal quantiles times that spread, mapped back
through the inverse logit and clamped to the printable range.
"""

import json
import math
import os
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
NORMAL_Q = [-1.6448536269514722, -0.6744897501960817, 0.0,
            0.6744897501960817, 1.6448536269514722]

# Release day begins at 2026-11-19T00:00:00Z; the log-time axis is measured
# from there, so launch-night reads sit just below zero on it.
RELEASE_EPOCH = datetime(2026, 11, 19, tzinfo=timezone.utc).timestamp()

LO = 0.01
HI = 99.99


def clamp_pct(value):
    """Keep a percent inside the range the page can print (and logit can take)."""
    if value != value:  # NaN guard: treat as the floor, never propagate it.
        return LO
    return LO if value < LO else HI if value > HI else value


def logit_of(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit_of(z):
    if z >= 0:
        return 100.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return 100.0 * e / (1.0 + e)


def round4(value):
    """Round to 4 decimals, halves away from zero, returned as int when whole."""
    snapped = Decimal(repr(value)).quantize(
        Decimal("0.0001"), rounding=ROUND_HALF_UP)
    as_float = float(snapped)
    return int(as_float) if as_float == int(as_float) else as_float


def parse_moment(text):
    moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
    return moment.timestamp()


def hours_since_release(epoch_seconds):
    return (epoch_seconds - RELEASE_EPOCH) / 3600.0


def time_axis(epoch_seconds):
    """Concave time axis: log(1 + hours since release began)."""
    return math.log1p(max(-0.9, hours_since_release(epoch_seconds)))


def median_of(values):
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def weighted_line(xs, ys, ws):
    """Weighted least squares for y ~ a + b*x.

    Returns (intercept, slope, residual_std, x_bar, sxx). With fewer than
    two usable points the slope is 0 and the intercept is the weighted mean.
    """
    total_w = sum(ws)
    if total_w <= 0 or not xs:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    x_bar = sum(w * x for x, w in zip(xs, ws)) / total_w
    y_bar = sum(w * y for y, w in zip(ys, ws)) / total_w
    sxx = sum(w * (x - x_bar) ** 2 for x, w in zip(xs, ws))
    if sxx <= 1e-12 or len(xs) < 2:
        resid = math.sqrt(
            sum(w * (y - y_bar) ** 2 for y, w in zip(ys, ws)) / total_w)
        return y_bar, 0.0, resid, x_bar, sxx
    sxy = sum(w * (x - x_bar) * (y - y_bar)
              for x, y, w in zip(xs, ys, ws))
    slope = sxy / sxx
    intercept = y_bar - slope * x_bar
    dof = max(1, len(xs) - 2)
    resid = math.sqrt(
        sum(w * (y - (intercept + slope * x)) ** 2
            for x, y, w in zip(xs, ys, ws)) / (total_w * dof / len(xs)))
    return intercept, slope, resid, x_bar, sxx


def project(xs, ys, ws, x_target, x_last, y_last, slope_override=None):
    """Extrapolate a series to x_target, anchored on its last point.

    Trophy curves saturate -- a story trophy climbs fast, then levels off
    as the players who will ever finish that chapter have done so -- so a
    straight line in log-time, read far past the last observation, would
    overshoot badly over a multi-day horizon. The move is therefore damped
    by 1 / (1 + the log-time gap): a short hop trusts the trend, a long one
    mostly trusts the last observed level. The saturating fit elsewhere in
    this file carries the growth the damping takes away.

    Returns (value_at_target, resid_std, slope_se), the last two from the
    linear fit, for the spread.
    """
    _, slope, resid, x_bar, sxx = weighted_line(xs, ys, ws)
    if slope_override is not None:
        slope = slope_override
    gap = max(0.0, x_target - x_last)
    delta = slope * gap / (1.0 + gap)
    value = y_last + delta
    slope_se = 0.0
    if sxx > 1e-9 and resid > 0:
        slope_se = min(0.6, resid / math.sqrt(sxx) * abs(x_target - x_bar))
    return value, min(0.8, max(0.12, resid)), slope_se


def saturating_rate(hours, rates, weights, last_rate, h_target):
    """Fit rate(h) = cap * (1 - exp(-k*h)) and read it at the target.

    A trophy's earn rate cannot grow forever: it approaches the share of
    players who will ever earn it. This fits that ceiling directly. For
    each candidate ceiling on a grid, -ln(1 - rate/cap) should be linear
    in hours through the origin with slope k; the ceiling is chosen by the
    fit's residual in logit space, and among near-ties the smallest
    ceiling wins, which guards against runaway growth. Returns the
    forecast rate, or None when the shape does not fit (too few points,
    no time span, or a falling trend, which a ceiling cannot describe).
    """
    if len(hours) < 4 or (max(hours) - min(hours)) < 2.0:
        return None
    if last_rate <= 0:
        return None
    # Reads near zero carry no ceiling information (and the logit is
    # singular there), so they are left out of the fit entirely.
    usable = [(h, r, w) for h, r, w in zip(hours, rates, weights)
              if r >= 0.05]
    if len(usable) < 3:
        return None
    # For each candidate ceiling, -ln(1 - rate/cap) should be linear in
    # hours through the origin with slope k. The ceiling is chosen by the
    # fit's residual measured in logit space, which is stable where the
    # z-space residual is not: near saturation a hair's change in the
    # ceiling explodes z, while the rates themselves barely move.
    best = None  # (resid_logit, cap, k)
    top = 99.9 / last_rate
    if top <= 1.005:
        return None
    steps = 60
    for step in range(steps + 1):
        mult = math.exp(math.log(1.005) + (math.log(top) - math.log(1.005))
                        * step / steps)
        cap = min(99.9, last_rate * mult)
        zs = []
        ok = True
        for rate in rates:
            if rate >= cap * 0.999:
                ok = False
                break
            zs.append(-math.log(1.0 - rate / cap))
        if not ok:
            continue
        num = sum(w * h * z for h, z, w in zip(hours, zs, weights))
        den = sum(w * h * h for h, w in zip(hours, weights))
        if den <= 1e-9:
            continue
        k = num / den
        if k <= 0:
            continue
        total_w = sum(w for _, _, w in usable)
        resid = math.sqrt(sum(
            w * (logit_of(cap * (1.0 - math.exp(-k * h))) - logit_of(r)) ** 2
            for h, r, w in usable) / total_w)
        if best is None or resid < best[0]:
            best = (resid, cap, k)
    if best is None:
        return None
    resid, cap, k = best
    # Parsimony: the smallest ceiling whose fit is nearly as good.
    for step in range(steps + 1):
        mult = math.exp(math.log(1.005) + (math.log(top) - math.log(1.005))
                        * step / steps)
        cand_cap = min(99.9, last_rate * mult)
        if cand_cap >= cap:
            break
        zs = []
        ok = True
        for rate in rates:
            if rate >= cand_cap * 0.999:
                ok = False
                break
            zs.append(-math.log(1.0 - rate / cand_cap))
        if not ok:
            continue
        num = sum(w * h * z for h, z, w in zip(hours, zs, weights))
        den = sum(w * h * h for h, w in zip(hours, weights))
        if den <= 1e-9:
            continue
        cand_k = num / den
        if cand_k <= 0:
            continue
        total_w = sum(w for _, _, w in usable)
        cand_resid = math.sqrt(sum(
            w * (logit_of(cand_cap * (1.0 - math.exp(-cand_k * h)))
                 - logit_of(r)) ** 2
            for h, r, w in usable) / total_w)
        if cand_resid <= 1.10 * resid:
            cap, k = cand_cap, cand_k
            break
    forecast = cap * (1.0 - math.exp(-k * h_target))
    forecast = max(last_rate * 0.5, min(99.9, forecast))
    return forecast


def distinct_reads(history):
    """Collapse runs of identical reads (same players, same rates) to one.

    Exophase only changes the page when it refreshes, so a run of equal
    reads is one observation repeated, not many. The kept read is the first
    of each run: the rates were already true at some refresh before that
    sighting, so the first sighting is the least-stale timestamp for them.
    """
    kept = []
    for read in history:
        key = (read.get("players_tracked"),
               tuple(read.get("rates") or []))
        if kept and kept[-1][0] == key:
            continue
        kept.append((key, read))
    return [read for _, read in kept]


def recency_weights(moments, last_moment):
    """Half-weight per 12 hours of age, so stale launch-night reads fade."""
    weights = []
    for moment in moments:
        age_hours = max(0.0, (last_moment - moment) / 3600.0)
        weights.append(0.5 ** (age_hours / 12.0))
    return weights


def recent_window(moments, xs, ys, ws, span_hours=24.0):
    """Keep only the reads from the last ``span_hours`` before the last one.

    A trophy's growth regime changes as it saturates: the steep climb of
    launch night says little about how it moves on day three. Fits that
    extrapolate therefore use only the recent window (the house curve
    baseline does the same), and fall back to the whole history when the
    window holds fewer than three reads.
    """
    last = moments[-1]
    keep = [i for i, m in enumerate(moments)
            if (last - m) / 3600.0 <= span_hours]
    if len(keep) < 3:
        keep = list(range(len(moments)))
    return ([xs[i] for i in keep], [ys[i] for i in keep],
            [ws[i] for i in keep], [moments[i] for i in keep], keep)


def forecast_trophy(idx, trophies, series, target_x, horizon_days,
                    global_logit_slope, global_earn_slope):
    """Forecast one trophy's five quantiles at the target.

    ``series`` is the case's distinct reads. Returns a list of five
    percents, non-decreasing, each in [0.01, 100].
    """
    last = series[-1]
    last_rates = last.get("rates") or []
    last_rate = (last_rates[idx] if idx < len(last_rates)
                 and last_rates[idx] is not None else None)
    if last_rate is None:
        # A trophy the last read lacked: borrow the tier's median rate.
        tier = trophies[idx].get("tier")
        peers = [r for t, r in zip(trophies, last_rates)
                 if r is not None and t.get("tier") == tier]
        last_rate = median_of(peers) if peers else 1.0
    last_rate = clamp_pct(last_rate)
    last_logit = logit_of(last_rate)

    # The trophy everyone already holds stays (almost) held: its median
    # sits just under 100 with a wide lower tail, not a point mass.
    if last_rate >= HI:
        med_logit = logit_of(99.85)
        sigma = 0.85
        return quantiles_from(med_logit, sigma)

    moments = [parse_moment(r["at"]) for r in series]
    xs = [time_axis(m) for m in moments]
    weights = recency_weights(moments, moments[-1])
    x_last = xs[-1]
    n_points = len(series)

    # --- Direct fit: logit(rate) on the log-time axis, anchored at last. --
    direct_logit = None
    direct_resid = 0.25
    direct_slope_se = 0.0
    ys = []
    usable = True
    for read in series:
        rates = read.get("rates") or []
        value = rates[idx] if idx < len(rates) else None
        if value is None:
            usable = False
            break
        ys.append(logit_of(value))
    if usable and n_points >= 3 and (max(xs) - min(xs)) > 0.05:
        rx, ry, rw, _, _ = recent_window(moments, xs, ys, weights)
        _, slope, _, _, _ = weighted_line(rx, ry, rw)
        shrink = len(rx) / (len(rx) + 4.0)
        slope = shrink * max(-2.0, min(3.0, slope)) + \
            (1.0 - shrink) * global_logit_slope
        slope = max(-2.0, min(3.0, slope))
        direct_logit, direct_resid, direct_slope_se = project(
            rx, ry, rw, target_x, x_last, last_logit,
            slope_override=slope)

    # --- Count fit: earned and tracked separately, then the ratio. -------
    count_logit = None
    n_last = series[-1].get("players_tracked") or 1
    log_n_last = math.log(max(1, n_last))
    log_n_all = [math.log(max(1, r.get("players_tracked") or 1))
                 for r in series]
    _, _, _, _, keep = recent_window(moments, xs, log_n_all, weights)
    rx = [xs[i] for i in keep]
    rw = [weights[i] for i in keep]
    log_n_target, _, _ = project(
        rx, [log_n_all[i] for i in keep],
        rw, target_x, x_last, log_n_last)
    n_target = math.exp(log_n_target)
    n_target = max(n_last, min(n_last * 30.0, n_target))
    earn_ys = []
    earn_ok = True
    for read in series:
        rates = read.get("rates") or []
        value = rates[idx] if idx < len(rates) else None
        players = read.get("players_tracked") or 0
        if value is None or players <= 0:
            earn_ok = False
            break
        earn_ys.append(math.log(value * players / 100.0 + 0.5))
    if earn_ok and n_points >= 3 and (max(xs) - min(xs)) > 0.05:
        _, slope, _, _, _ = weighted_line(
            rx, [earn_ys[i] for i in keep], rw)
        shrink = len(keep) / (len(keep) + 4.0)
        slope = shrink * max(-1.0, min(4.0, slope)) + \
            (1.0 - shrink) * global_earn_slope
        slope = max(-1.0, min(4.0, slope))
        earn_last = math.log(last_rate * n_last / 100.0 + 0.5)
        log_earn_target, _, _ = project(
            rx, [earn_ys[i] for i in keep], rw, target_x, x_last, earn_last,
            slope_override=slope)
        earn_target = math.exp(log_earn_target) - 0.5
        earn_target = max(earn_last - 0.5 + 0.5, earn_target)
        earn_target = min(earn_target, n_target)
        if earn_target > 0:
            count_logit = logit_of(100.0 * earn_target / n_target)

    # --- Combine the available estimates in logit space. ---------------
    parts = []
    if direct_logit is not None:
        parts.append(direct_logit)
    if count_logit is not None:
        parts.append(count_logit)
    if parts and direct_logit is not None:
        # A saturating ceiling fit, when the trend is rising: it is the
        # one component that knows a trophy's rate levels off instead of
        # growing forever, so it anchors the fast early climbers.
        raw_rates = []
        sat_ok = True
        for read in series:
            rates = read.get("rates") or []
            value = rates[idx] if idx < len(rates) else None
            if value is None:
                sat_ok = False
                break
            raw_rates.append(value)
        if sat_ok:
            hours = [max(0.0, hours_since_release(m)) for m in moments]
            h_target = math.exp(target_x) - 1.0
            sat_rate = saturating_rate(hours, raw_rates, weights,
                                       last_rate, h_target)
            if sat_rate is not None and sat_rate > last_rate:
                parts.append(logit_of(sat_rate))
    if parts:
        med_logit = sum(parts) / len(parts)
    else:
        # No movement was ever observed: the page has been stale since the
        # first read, so a refresh is overdue, and a refresh dilutes rates
        # hard. The rules describe the first refresh: tracked players jump
        # roughly sixfold while each trophy gains only a couple of new
        # holders, because the newcomers have barely played. In count terms
        # the rate after one refresh is about (earned + 2) / (6 * tracked),
        # i.e. roughly a fifth of the stale rate for an ordinary trophy.
        # Over a longer horizon players catch up, so the shrunken rate is
        # allowed to drift back up, slowly, with time.
        n_last = series[-1].get("players_tracked") or 1
        earned_last = last_rate * n_last / 100.0
        refreshed = 100.0 * (earned_last + 2.0) / (6.0 * n_last)
        recovery = 1.0 + 0.10 * max(0.0, horizon_days - 0.25)
        med_logit = logit_of(refreshed * min(recovery, 1.6))
        direct_resid = 0.45

    # Never let an extrapolation run away from the last observed rate.
    med_logit = max(last_logit - 2.0, min(last_logit + 2.0, med_logit))

    # --- Spread: random walk + fit scatter + slope error + one refresh. --
    walk = 0.45 * math.sqrt(horizon_days)
    refresh = 0.35 * (0.5 + 0.5 * min(1.0, horizon_days))
    sigma = math.sqrt(walk * walk + direct_resid ** 2 +
                      direct_slope_se ** 2 + refresh * refresh)
    sigma = max(0.30, min(1.60, sigma))
    return quantiles_from(med_logit, sigma)


def quantiles_from(med_logit, sigma):
    out = []
    previous = LO
    for z in NORMAL_Q:
        value = inv_logit_of(med_logit + z * sigma)
        value = round4(max(LO, min(100.0, value)))
        if value < previous:
            value = previous
        out.append(value)
        previous = value
    return out


def global_slopes(series, trophy_count):
    """Median per-trophy slopes, used to shrink noisy individual fits."""
    moments = [parse_moment(r["at"]) for r in series]
    xs = [time_axis(m) for m in moments]
    weights = recency_weights(moments, moments[-1])
    logit_slopes = []
    earn_slopes = []
    for idx in range(trophy_count):
        ys, earn_ys = [], []
        ok = True
        for read in series:
            rates = read.get("rates") or []
            value = rates[idx] if idx < len(rates) else None
            players = read.get("players_tracked") or 0
            if value is None or players <= 0:
                ok = False
                break
            ys.append(logit_of(value))
            earn_ys.append(math.log(value * players / 100.0 + 0.5))
        if not ok or len(series) < 3 or (max(xs) - min(xs)) <= 0.05:
            continue
        rx, ry, rw, _, _ = recent_window(moments, xs, ys, weights)
        _, slope, _, _, _ = weighted_line(rx, ry, rw)
        logit_slopes.append(max(-2.0, min(3.0, slope)))
        _, _, _, _, keep = recent_window(moments, xs, earn_ys, weights)
        _, eslope, _, _, _ = weighted_line(
            [xs[i] for i in keep], [earn_ys[i] for i in keep],
            [weights[i] for i in keep])
        earn_slopes.append(max(-1.0, min(4.0, eslope)))
    logit_med = median_of(logit_slopes) if logit_slopes else 0.0
    earn_med = median_of(earn_slopes) if earn_slopes else 0.0
    return logit_med, earn_med


def forecast_case(case):
    trophies = case["trophies"]
    history = case.get("history") or []
    if not history:
        raise ValueError("case has no history")
    series = distinct_reads(history)
    target_x = time_axis(parse_moment(case["target"]))
    horizon_days = case["horizon_minutes"] / 1440.0
    logit_slope, earn_slope = global_slopes(series, len(trophies))
    rows = []
    for idx, trophy in enumerate(trophies):
        rows.append({
            "id": trophy["id"],
            "q": forecast_trophy(idx, trophies, series, target_x,
                                 horizon_days, logit_slope, earn_slope),
        })
    return {"trophies": rows}


def fallback_case(case):
    """Last-rate persistence with a wide spread: never an invalid output."""
    trophies = case["trophies"]
    history = case.get("history") or []
    last_rates = (history[-1].get("rates") or []) if history else []
    horizon_days = case["horizon_minutes"] / 1440.0
    sigma = max(0.5, min(1.6, 0.5 * math.sqrt(horizon_days) + 0.35))
    rows = []
    for idx, trophy in enumerate(trophies):
        rate = last_rates[idx] if idx < len(last_rates) \
            and last_rates[idx] is not None else 1.0
        rows.append({"id": trophy["id"],
                     "q": quantiles_from(logit_of(rate), sigma)})
    return {"trophies": rows}


def run_case(path):
    with open(path, encoding="utf-8") as handle:
        case = json.load(handle)
    try:
        return forecast_case(case)
    except Exception:
        return fallback_case(case)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    inputs = sorted(INPUT_DIR.glob("*.in"))
    for path in inputs:
        forecast = run_case(path)
        text = json.dumps(forecast, separators=(",", ":"),
                          ensure_ascii=False)
        if len(text.encode("utf-8")) > 65536:
            raise ValueError(f"{path.name}: forecast over 65536 bytes")
        out_path = OUTPUT_DIR / (path.stem + ".out")
        out_path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
