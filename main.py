#!/usr/bin/env python3
"""Forecast Roblox Up-and-Coming top-10 chart. Stdlib only, deterministic, no network.

Model: per-game continuous-time rank drift model.
- From history's consecutive read pairs we estimate, as a function of current rank r:
  mu(r)  = mean rank change per hour (drift, smoothed),
  sigma(r) = per-hour rank-change dispersion (smoothed),
  hazard(r) = per-hour probability the game leaves the chart.
- Optional momentum tilt from recent rank velocity.
- For horizon d hours: game at rank r ends near k with probability from a discretized
  Gaussian over rank change, times survival (1-hazard)^d. Coverage c_k(d) = probability
  place k is held by a game that was already on the chart, estimated from pairs; named
  games' probabilities at place k are rescaled to sum to c_k(d); the rest is unnamed mass
  (entrants), which the scorer distributes over unnamed chart games.
- Final probabilities mix horizons d=1..72 uniformly (scored hourly).
"""
import json, math, os, sys, bisect
from datetime import datetime, timezone

TOP = 10
HORIZONS = list(range(1, 73))
MIN_P = 1e-9
TAIL_W = 0.25
TAIL_X = 2.5
MOMENTUM_W = -0.15
MOMENTUM_LB = 2
MAX_CANDS = 12
PLACE_ROWS_MAX = 100


def parse_ts(s):
    s = s.replace("Z", "+00:00")
    return datetime.fromisoformat(s)


def load_history(indir):
    path = os.path.join(indir, "history.ndjson")
    reads = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if not r.get("ok", True):
                continue
            uc = r.get("up_and_coming") or []
            m = {}
            for row in uc:
                rk = row.get("rank")
                uid = row.get("universe_id")
                if rk is None or uid is None:
                    continue
                m[int(rk)] = int(uid)
            if not m:
                continue
            reads.append((parse_ts(r["at"]), m))
    reads.sort(key=lambda x: x[0])
    return reads


def pairs_from(reads, max_delta=30.0):
    """Consecutive (and near-consecutive) read pairs within max_delta hours."""
    out = []
    for i in range(len(reads) - 1):
        t0, m0 = reads[i]
        t1, m1 = reads[i + 1]
        d = (t1 - t0).total_seconds() / 3600.0
        if 0 < d <= max_delta:
            out.append((d, m0, m1))
    return out


def smoothed(vals, halfwidth=1, default=0.0):
    """Simple moving average over rank index (1-based list indexed by rank-1)."""
    n = len(vals)
    out = []
    for i in range(n):
        lo = max(0, i - halfwidth)
        hi = min(n, i + halfwidth + 1)
        window = [v for v in vals[lo:hi] if v is not None]
        out.append(sum(window) / len(window) if window else default)
    return out


def estimate(pairs, support=110):
    """Estimate drift, dispersion, hazard per rank; coverage per place/horizon params."""
    n = support
    sums = [[0.0, 0.0, 0] for _ in range(n)]  # per start rank: sum dr, sum dr^2, count
    hazards = [[0.0, 0] for _ in range(n)]
    for d, m0, m1 in pairs:
        base = set(m0.values())
        for r, uid in m0.items():
            if r > n:
                continue
            if uid not in m1.values():
                hazards[r - 1][0] += 1.0
                hazards[r - 1][1] += 1
                continue
            r1 = None
            for rr, uu in m1.items():
                if uu == uid:
                    r1 = rr
                    break
            if r1 is None:
                continue
            dr = r1 - r
            # wrap-around ambiguity (left chart and re-entered) is rare; ignore |dr| > 60
            if abs(dr) > 60:
                continue
            s = sums[r - 1]
            s[0] += dr / d
            s[1] += (dr / d) ** 2
            s[2] += 1
    delta_sum = sum(d for d, _, _ in pairs)
    delta_mean = delta_sum / len(pairs) if pairs else 6.0
    mu = [0.0] * n
    sig = [1.0] * n
    haz = [0.0] * n
    for r in range(n):
        c = sums[r][2]
        if c >= 3:
            mu[r] = sums[r][0] / c
            var = max(sums[r][1] / c - mu[r] ** 2, 0.0)
            sig[r] = math.sqrt(var) if var > 1e-6 else 1.0
        if hazards[r][1] >= 3:
            rate = hazards[r][0] / hazards[r][1]
            haz[r] = -math.log(max(1.0 - rate, 1e-6)) / delta_mean
        else:
            haz[r] = None
    # smoothing + fallbacks: mean of observed
    obs_mu = [m for m in mu]
    known = [m for m in obs_mu if any(sums[r][2] >= 3 for r in range(n))]
    mu_s = smoothed([m if sums[r][2] >= 3 else None for r, m in enumerate(obs_mu)],
                    halfwidth=3, default=0.0)
    sig_s = smoothed([sig[r] if sums[r][2] >= 3 else None for r in range(n)],
                     halfwidth=3, default=1.0)
    haz_known = [h for r, h in enumerate(haz) if h is not None]
    haz_default = sum(haz_known) / len(haz_known) if haz_known else 0.01
    haz_s = smoothed([h if h is not None else None for h in haz],
                     halfwidth=4, default=haz_default)
    # clamp hazards
    haz_s = [min(max(h, 0.0), 0.25) for h in haz_s]
    # coverage: P(place k held by a game on the base chart), fit log-linear in delta:
    # ln(cov) = a_k + b_k * ln(d)
    sx = [0.0] * (TOP + 1)
    sy = [0.0] * (TOP + 1)
    sxx = [0.0] * (TOP + 1)
    sxy = [0.0] * (TOP + 1)
    cnt = [0] * (TOP + 1)
    for d, m0, m1 in pairs:
        base = set(m0.values())
        x = math.log(d)
        for k in range(1, TOP + 1):
            y = 1.0 if m1.get(k) in base else 0.0
            sx[k] += x; sy[k] += y; sxx[k] += x * x; sxy[k] += x * y
            cnt[k] += 1
    cov = {}
    for k in range(1, TOP + 1):
        if cnt[k] >= 5:
            den = cnt[k] * sxx[k] - sx[k] ** 2
            if abs(den) > 1e-9:
                b = (cnt[k] * sxy[k] - sx[k] * sy[k]) / den
                a = (sy[k] - b * sx[k]) / cnt[k]
                b = min(max(b, -0.5), 0.0)
            else:
                a, b = math.log(max(sy[k] / cnt[k], 0.05)), 0.0
            cov[k] = (a, b)
        else:
            cov[k] = (math.log(0.8), 0.0)
    return {
        "mu": mu_s, "sig": sig_s, "haz": haz_s, "haz_default": haz_default,
        "cov": cov, "support": n,
    }


def apply_momentum(est, reads, base_idx, weight=0.5):
    """Tilt per-game drift by recent observed rank velocity."""
    if base_idx < 1:
        return {}
    vel = {}
    t_prev, m_prev = reads[base_idx - 1]
    t0, m0 = reads[base_idx]
    d = (t0 - t_prev).total_seconds() / 3600.0
    if d <= 0:
        return {}
    pos_prev = {u: r for r, u in m_prev.items()}
    for r, u in m0.items():
        rp = pos_prev.get(u)
        if rp is not None:
            vel[u] = (rp - r) / d  # positive = moving up
    return vel


def prob_game_at_place(mu_r, sig_r, r, k, d):
    """Two-component Gaussian mixture: bulk + wide component (heavy tails)."""
    return (1.0 - TAIL_W) * _gauss(mu_r, sig_r, r, k, d) + TAIL_W * _gauss(mu_r, TAIL_X * sig_r, r, k, d)


def _gauss(mu_r, sig_r, r, k, d):
    """Discretized Gaussian probability that rank r+d*drift lands at place k."""
    mean = r + mu_r * d
    sd = max(sig_r * math.sqrt(d), 0.25)
    # integrate normal over [k-0.5, k+0.5]
    def phi(z):
        return math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
    zlo = (k - 0.5 - mean) / sd
    zhi = (k + 0.5 - mean) / sd
    # erf-based
    def norm_cdf(z):
        return 0.5 * (1 + math.erf(z / math.sqrt(2)))
    return norm_cdf(zhi) - norm_cdf(zlo)


def forecast(reads, est, momentum_weight=MOMENTUM_W, momentum_lookback=MOMENTUM_LB, max_candidates=MAX_CANDS):
    base_idx = len(reads) - 1
    t0, m0 = reads[base_idx]
    ranks = {u: r for r, u in m0.items()}
    mu, sig, haz = est["mu"], est["sig"], est["haz"]
    n = est["support"]

    # per-game velocity from last momentum_lookback pairs
    vel = {}
    for back in range(1, momentum_lookback + 1):
        if base_idx - back < 0:
            break
        tp, mp = reads[base_idx - back]
        dt_h = (t0 - tp).total_seconds() / 3600.0
        if dt_h <= 0:
            continue
        pos_prev = {u: r for r, u in mp.items()}
        for r, u in m0.items():
            rp = pos_prev.get(u)
            if rp is not None:
                v = (rp - r) / dt_h
                vel[u] = vel.get(u, 0.0) + v / momentum_lookback
    gmu = {}
    for u, r in ranks.items():
        base_mu = mu[min(r, n) - 1] if r <= n else 0.0
        gmu[u] = base_mu + momentum_weight * vel.get(u, 0.0)

    horizon_p = []  # per horizon: {place: {uid: p}}
    for d in HORIZONS:
        per_place = {k: {} for k in range(1, TOP + 1)}
        for u, r in ranks.items():
            mr = min(r, n)
            mu_r = max(min(gmu[u], 3.0), -3.0)
            sig_r = max(sig[mr - 1], 0.35)
            surv = (1.0 - haz[mr - 1]) ** d
            if surv < 1e-6:
                continue
            for k in range(1, TOP + 1):
                p = prob_game_at_place(mu_r, sig_r, r, k, d) * surv
                if p > MIN_P:
                    per_place[k][u] = per_place[k].get(u, 0.0) + p
        # rescale each place to empirical coverage
        for k in range(1, TOP + 1):
            a, b = est["cov"][k]
            c = math.exp(min(a + b * math.log(d), 0.0))
            c = min(max(c, 0.05), 1.0)
            s = sum(per_place[k].values())
            if s > 1e-9:
                f = c / s
                for u in per_place[k]:
                    per_place[k][u] *= f
        horizon_p.append(per_place)

    # uniform mixture over horizons
    probs = {k: {} for k in range(1, TOP + 1)}
    for per_place in horizon_p:
        for k in range(1, TOP + 1):
            for u, p in per_place[k].items():
                probs[k][u] = probs[k].get(u, 0.0) + p / len(HORIZONS)

    rows = []
    top10 = []
    # top10: rank games by expected position (lowest expected rank first), survivors only
    exp_pos = {}
    for u, r in ranks.items():
        mr = min(r, n)
        mu_r = max(min(gmu[u], 3.0), -3.0)
        surv = (1.0 - haz[mr - 1]) ** 36
        exp_pos[u] = max(r + mu_r * 36, 0.5) - (0.0 if surv > 0.5 else 1000)
    ordered = sorted(ranks.keys(), key=lambda u: (exp_pos[u], u))
    top10 = ordered[:TOP]
    for k in range(1, TOP + 1):
        items = sorted(probs[k].items(), key=lambda x: (-x[1], x[0]))[:max_candidates]
        # keep mass cap: sum must be <= 1
        for u, p in items:
            rows.append({"universe_id": int(u), "place": k, "p": round(min(max(p, 0.0), 1.0), 8)})
    return {"probabilities": rows, "top10": [int(u) for u in top10]}


def main():
    indir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    outdir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    reads = load_history(indir)
    if not reads:
        raise SystemExit("no history reads")
    pairs = pairs_from(reads)
    if not pairs:
        raise SystemExit("no usable pairs")
    est = estimate(pairs)
    fc = forecast(reads, est)
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, "forecast.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(fc, f, separators=(",", ":"))
    print(f"wrote {out}: {len(fc['probabilities'])} rows")


if __name__ == "__main__":
    main()
