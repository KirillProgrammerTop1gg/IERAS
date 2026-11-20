# -*- coding: utf-8 -*-
"""
Iterative transit candidate search using BLS (version 2).

Installation:
    pip install lightkurve astropy numpy scipy matplotlib pandas

Usage:
    python ieras_transit_search.py --target Kepler-69
    python ieras_transit_search.py --selftest          # check with synthetic data
    python ieras_transit_search.py --target "TIC 261136679" --mission TESS
        --author SPOC --cadence ""

Pipeline:
    load PDCSAP flux, errors, quality flags, and segments
      -> detrend while protecting previously found transits
      -> subtract fitted planet models without discarding measurements
      -> coarse binned parallel BLS scan -> multiple peaks
      -> refine each peak on a fine grid
      -> vet each candidate (transit count, odd/even depth, single-event dominance,
         depth consistency, red noise, harmonics, and phase independence)
      -> accepted -> next iteration; no accepted peaks -> stop
      -> jointly fit all accepted planets

Not included (requires other data): centroid-shift vetting (TPF/centroid data),
a physical ingress/egress model with limb darkening.
This tool finds candidates; it does not establish discoveries.
"""

import argparse
import logging
import os
from bisect import bisect_left, insort
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.timeseries import BoxLeastSquares
from scipy.optimize import least_squares
from scipy.signal import find_peaks

log = logging.getLogger("exo")


# ============================ SETTINGS ============================
@dataclass
class Config:
    target: str = "Kepler-69"
    mission: str = "Kepler"
    author: str = "Kepler"
    cadence: Optional[str] = "long"
    flux_column: str = "pdcsap_flux"
    out_dir: str = "exoplanet_output"

    # search
    min_period: float = 1.0
    max_period: Optional[float] = None  # None -> baseline / min_transits
    durations: tuple = (0.06, 0.09, 0.13, 0.19, 0.28, 0.40, 0.60)  # days
    # In astropy, a smaller frequency_factor means a denser grid and longer runtime.
    scan_freq_factor: float = 3.0  # coarse scan
    refine_freq_factor: float = 0.25  # fine grid around the peak
    refine_halfwidth: float = 0.005  # keep refinement on the selected peak, not a nearby alias
    max_refine_grid: int = 5_000  # finer than transit timing resolution for typical baselines
    max_grid: int = 3_000_000
    n_peaks: int = 20  # independent peaks to vet per iteration
    peak_separation: float = 0.005  # keep nearby independent long-period peaks
    max_planets: int = 10
    n_jobs: int = max(1, os.cpu_count() or 1)

    # preprocessing
    window_factor: float = 2.5  # flatten window = window_factor * max(durations)
    protect_factor: float = 2.0  # transit-protection mask during flattening (times duration)
    outlier_sigma: float = 5.0  # upward outliers only

    # Empirical thresholds; --calibrate estimates them by block permutation.
    min_snr: float = 7.1
    min_sde: float = 8.0
    min_transits: int = 3
    min_points_in_transit: int = 5  # reject events defined by only a few cadences
    sparse_transit_points: int = 3
    min_sparse_transits: int = 20
    min_sparse_snr: float = 20.0
    min_coverage: float = 0.5  # fraction of expected in-transit cadences
    max_odd_even_sigma: float = 3.0
    max_single_event_fraction: float = 0.6
    max_depth_chi2: float = 5.0
    max_depth_chi2_repeated: float = 6.0
    repeated_signal_transits: int = 8
    min_half_snr: float = 5.5
    max_depth_chi2_strong: float = 15.0  # allow astrophysical depth variation for strong signals
    strong_signal_snr: float = 12.0
    strong_signal_transits: int = 8
    alias_tol: float = 0.01
    max_alias_order: int = 32
    max_alias_denominator: int = 12  # short subharmonics and simple ratios such as 2:5
    alias_overlap: float = (
        0.8  # >=80% of epochs inside another planet's transits implies a dependent signal
    )
    calibrate_n: int = 0

    plots: bool = True
    show_plots: bool = False


# ============================ HELPERS ============================
def phase_of(t, P, t0):
    return ((t - t0 + P / 2) % P) - P / 2


def in_transit(t, planets, factor=1.0):
    m = np.zeros(len(t), bool)
    for p in planets:
        m |= np.abs(phase_of(t, p["period"], p["t0"])) < factor * p["duration"] / 2
    return m


def robust_sigma(x):
    x = np.asarray(x)
    if len(x) < 5:
        return np.nan
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def p2p_sigma(t, f):
    """Estimate noise from adjacent differences, robust to slow trends."""
    if len(t) < 10:
        return np.nan
    cad = np.median(np.diff(t))
    d = np.diff(f)[np.diff(t) < 2 * cad]
    return robust_sigma(d) / np.sqrt(2) if len(d) > 5 else np.nan


def build_dy(t, f, e, seg):
    """Scale individual errors to the observed noise of each segment."""
    dy = np.empty_like(f)
    for s in np.unique(seg):
        m = seg == s
        sig = p2p_sigma(t[m], f[m])
        if not np.isfinite(sig) or sig <= 0:
            sig = np.nanstd(f[m])
        if not np.isfinite(sig) or sig <= 0:
            sig = 1e-3
        ee = np.where(np.isfinite(e[m]) & (e[m] > 0), e[m], np.nan)
        med = np.nanmedian(ee) if np.any(np.isfinite(ee)) else np.nan
        if np.isfinite(med) and med > 0:
            dy[m] = np.where(np.isfinite(ee), ee, med) * sig / med
        else:
            dy[m] = sig
    return dy


def bin_data(t, f, dy, width):
    idx = np.floor((t - t.min()) / width).astype(np.int64)
    _, inv = np.unique(idx, return_inverse=True)
    w = 1.0 / dy**2
    sw = np.bincount(inv, w)
    return (np.bincount(inv, t * w) / sw, np.bincount(inv, f * w) / sw, 1.0 / np.sqrt(sw))


# ============================ LOADING ============================
def load_segments(cfg):
    """List quarter/sector segments with time, flux, and errors."""
    import lightkurve as lk

    cache = os.path.join(cfg.out_dir, f"{cfg.target.replace(' ', '_')}_raw.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        segs = [
            dict(time=z[f"t{i}"], flux=z[f"f{i}"], err=z[f"e{i}"], id=i) for i in range(int(z["n"]))
        ]
        log.info("Loaded cached data from %s (%d segments)", cache, len(segs))
        return segs

    kw = dict(mission=cfg.mission, author=cfg.author)
    if cfg.cadence:
        kw["cadence"] = cfg.cadence
    sr = lk.search_lightcurve(cfg.target, **kw)
    if len(sr) == 0:
        raise RuntimeError(f"No data found for {cfg.target}")
    log.info("\n%s", sr)

    try:
        coll = sr.download_all(flux_column=cfg.flux_column, quality_bitmask="default")
    except Exception as ex:  # the column may have a different name
        log.warning("flux_column=%s unavailable (%s); using the default", cfg.flux_column, ex)
        coll = sr.download_all(quality_bitmask="default")
    if coll is None or len(coll) == 0:
        raise RuntimeError("Could not download data")

    segs = []
    for i, lc in enumerate(coll):
        try:
            t = np.asarray(lc.time.value, float)
            f = np.asarray(lc.flux.value, float)
            e = np.asarray(lc.flux_err.value, float)
            ok = np.isfinite(t) & np.isfinite(f)
            t, f, e = t[ok], f[ok], e[ok]
            med = np.median(f)
            if len(t) < 100 or not np.isfinite(med) or med <= 0:
                log.warning("Skipped segment %d (insufficient data)", i)
                continue
            segs.append(dict(time=t, flux=f / med, err=e / med, id=i))
        except Exception as ex:
            log.warning("Skipped segment %d: %s", i, ex)
    if not segs:
        raise RuntimeError("No usable segments")

    os.makedirs(cfg.out_dir, exist_ok=True)
    arrs = {"n": len(segs)}
    for i, s in enumerate(segs):
        arrs.update({f"t{i}": s["time"], f"f{i}": s["flux"], f"e{i}": s["err"]})
    np.savez_compressed(cache, **arrs)
    return segs


# ============================ PREPROCESSING ============================
def preprocess(segments, planets, cfg):
    """Flatten while protecting known transits, then subtract their models."""
    from lightkurve import LightCurve

    window_days = cfg.window_factor * max(cfg.durations)
    T, F, E, S = [], [], [], []
    for s in segments:
        t, f, e = s["time"], s["flux"], s["err"]
        cad = np.median(np.diff(t))
        wl = int(window_days / cad) | 1  # odd
        wl = max(wl, 11)
        if len(t) <= wl + 2:
            continue
        mask = in_transit(t, planets, cfg.protect_factor) if planets else None
        try:
            lc = LightCurve(time=t, flux=f, flux_err=np.where(np.isfinite(e), e, np.nanmedian(e)))
            flat = lc.flatten(window_length=wl, polyorder=2, break_tolerance=5, mask=mask)
            ff = np.asarray(flat.flux.value, float)
            ee = np.asarray(flat.flux_err.value, float)
        except Exception as ex:
            log.warning("Flattening failed for segment %s: %s", s["id"], ex)
            continue
        ok = np.isfinite(ff)
        T.append(t[ok])
        F.append(ff[ok])
        E.append(ee[ok])
        S.append(np.full(ok.sum(), s["id"]))
    t, f, e, seg = map(np.concatenate, (T, F, E, S))

    # Subtract planet models while retaining all measurements.
    for p in planets:
        f[np.abs(phase_of(t, p["period"], p["t0"])) < p["duration"] / 2] += p["depth"]

    # Reject upward outliers only; transits are dips.
    sig = robust_sigma(f)
    keep = f < np.median(f) + cfg.outlier_sigma * sig
    t, f, e, seg = t[keep], f[keep], e[keep], seg[keep]
    order = np.argsort(t)
    t, f, e, seg = t[order], f[order], e[order], seg[order]
    return t, f, build_dy(t, f, e, seg)


# ============================ BLS ============================
def _bls_chunk(args):
    t, f, dy, periods, durations = args
    r = BoxLeastSquares(t, f, dy).power(periods, durations, objective="likelihood")
    return r.power, r.depth, r.depth_snr, r.duration, r.transit_time


def parallel_power(t, f, dy, periods, durations, n_jobs):
    durations = np.asarray([d for d in durations if d < periods.min()])
    if n_jobs <= 1 or len(periods) < 2000:
        out = [_bls_chunk((t, f, dy, periods, durations))]
    else:
        chunks = np.array_split(periods, n_jobs * 3)
        with ProcessPoolExecutor(n_jobs) as ex:
            out = list(ex.map(_bls_chunk, [(t, f, dy, c, durations) for c in chunks if len(c)]))
    return [np.concatenate(x) for x in zip(*out, strict=True)]


def periodogram_sde(power):
    """Estimate a local BLS background without letting strong peaks raise the mean."""
    n = len(power)
    bins = min(512, max(2, n // 20))
    edges = np.unique(np.linspace(0, np.sqrt(n), bins + 1) ** 2)
    edges = np.unique(np.clip(edges.astype(int), 0, n))
    centers, background = [], []
    for start, end in zip(edges[:-1], edges[1:], strict=True):
        if end > start:
            centers.append((start + end - 1) / 2)
            background.append(np.quantile(power[start:end], 0.25))
    trend = np.interp(np.arange(n), centers, background)
    residual = power - trend
    scale = robust_sigma(residual)
    if not np.isfinite(scale) or scale <= 0:
        scale = np.std(residual)
    return residual / scale if scale > 0 else np.zeros(n)


def scan(t, f, dy, cfg):
    """Coarse binned parallel scan; return period, power, SDE array, and more."""
    base = t.max() - t.min()
    pmax = cfg.max_period or base / cfg.min_transits
    dur = np.array(sorted(set(cfg.durations[1::2]) | {cfg.durations[-1]}))
    cad = np.median(np.diff(t))
    width = dur.min() / 2
    if width > 1.5 * cad:
        tb, fb, db = bin_data(t, f, dy, width)
    else:
        tb, fb, db = t, f, dy
    df = cfg.scan_freq_factor * dur.min() / base**2
    fmin, fmax = 1 / pmax, 1 / cfg.min_period
    n = int((fmax - fmin) / df) + 1
    if n > cfg.max_grid:
        log.warning("Grid size %d exceeds %d; coarsening", n, cfg.max_grid)
        n = cfg.max_grid
    periods = 1.0 / np.linspace(fmin, fmax, n)
    log.info("Scan: %d periods, %d binned points, %d processes", n, len(tb), cfg.n_jobs)
    power, depth, snr, dd, t0 = parallel_power(tb, fb, db, periods, dur, cfg.n_jobs)

    sde_arr = periodogram_sde(power)
    return dict(period=periods, power=power, sde=sde_arr, depth=depth, snr=snr, duration=dd, t0=t0)


def observed_epochs(t, period, t0, duration):
    """Return predicted transit centers with nearby measurements."""
    cycles = np.arange(np.ceil((t[0] - t0) / period), np.floor((t[-1] - t0) / period) + 1)
    centers = t0 + cycles * period
    positions = np.searchsorted(t, centers)
    right = np.abs(t[np.clip(positions, 0, len(t) - 1)] - centers)
    left = np.abs(t[np.clip(positions - 1, 0, len(t) - 1)] - centers)
    return centers[np.minimum(left, right) < duration / 2]


def classify_alias(period, duration, epochs, accepted, cfg):
    """Identify duplicates and phase-aligned low-order period harmonics."""
    warnings = []
    for j, previous in enumerate(accepted, start=1):
        ratio = period / previous["period"]
        if abs(ratio - 1) < cfg.alias_tol:
            return j, f"duplicate period of planet {j}", warnings

        closest = None
        error = cfg.alias_tol
        for numerator in range(1, cfg.max_alias_order + 1):
            for denominator in range(1, cfg.max_alias_denominator + 1):
                if denominator > 4 and numerator > 4:
                    continue
                if numerator == denominator or np.gcd(numerator, denominator) != 1:
                    continue
                harmonic = numerator / denominator
                distance = abs(ratio / harmonic - 1)
                if distance < error:
                    closest = (numerator, denominator)
                    error = distance
        if closest is None or not len(epochs):
            continue
        aligned = np.abs(phase_of(epochs, previous["period"], previous["t0"])) < max(
            duration, previous["duration"]
        )
        overlap = np.mean(aligned)
        numerator, denominator = closest
        # A p:q subharmonic aligns once every q candidate epochs, not every epoch.
        if aligned.sum() >= 2 and overlap >= cfg.alias_overlap / denominator:
            return (
                j,
                f"harmonic of planet {j} ({numerator}:{denominator}), "
                f"epochs overlap by {overlap:.0%}",
                warnings,
            )
        warnings.append(
            f"period near {numerator}:{denominator} of planet {j}; transits are independent"
        )
    return None, None, warnings


def pick_peaks(sc, k, sep, t=None, accepted=(), cfg=None):
    """Rank separated local maxima, omitting known aliases before refinement."""
    power = sc["power"]
    period = sc["period"]
    maxima, _ = find_peaks(power)
    maxima = np.concatenate((maxima, [0, len(power) - 1]))
    maxima = maxima[np.isfinite(power[maxima]) & (power[maxima] > 0)]
    order = maxima[np.argsort(power[maxima])[::-1]]
    selected_periods = []
    selected = []
    for i in order:
        candidate_period = float(period[i])
        pos = bisect_left(selected_periods, candidate_period)
        neighbors = selected_periods[max(0, pos - 1) : pos + 1]
        if any(abs(other / candidate_period - 1) < sep for other in neighbors):
            continue
        if accepted:
            epochs = observed_epochs(
                t, candidate_period, float(sc["t0"][i]), float(sc["duration"][i])
            )
            alias_of, _, _ = classify_alias(
                candidate_period, float(sc["duration"][i]), epochs, accepted, cfg
            )
            if alias_of is not None:
                continue
        selected.append(int(i))
        insort(selected_periods, candidate_period)
        if len(selected) == k:
            break
    return selected


def best_covered_peak(t, result, min_transits):
    """Prefer a BLS maximum supported by observations at multiple epochs."""
    power = np.asarray(result.power)
    maxima, _ = find_peaks(power)
    maxima = np.unique(np.concatenate((maxima, [0, len(power) - 1, np.argmax(power)])))
    for i in maxima[np.argsort(power[maxima])[::-1]]:
        if (
            len(
                observed_epochs(
                    t,
                    float(result.period[i]),
                    float(result.transit_time[i]),
                    float(result.duration[i]),
                )
            )
            >= min_transits
        ):
            return int(i)
    return int(np.argmax(power))


def refine(t, f, dy, P, cfg):
    """Refine a peak on a fine grid across all durations without binning."""
    base = t.max() - t.min()
    dmin = min(cfg.durations)
    df = cfg.refine_freq_factor * dmin / base**2
    f1, f2 = 1 / (P * (1 + cfg.refine_halfwidth)), 1 / (P * (1 - cfg.refine_halfwidth))
    n = int(min(max((f2 - f1) / df, 50), cfg.max_refine_grid))
    periods = 1.0 / np.linspace(f1, f2, n)
    dur = np.array([d for d in cfg.durations if d < periods.min()])
    bls = BoxLeastSquares(t, f, dy)
    r = bls.power(periods, dur, objective="likelihood")
    i = best_covered_peak(t, r, cfg.min_transits)
    return dict(
        period=float(r.period[i]),
        duration=float(r.duration[i]),
        t0=float(r.transit_time[i]),
        depth=float(r.depth[i]),
        snr=float(r.depth_snr[i]),
    )


# ============================ CANDIDATE VETTING ============================
def vet(t, f, dy, c, accepted, cfg):
    P, dur, t0 = c["period"], c["duration"], c["t0"]
    out = dict(reasons=[], warnings=[], n_transits=0, snr_corr=np.nan, alias_of=None)
    cad = np.median(np.diff(t))
    ph = phase_of(t, P, t0)
    epoch = np.rint((t - t0) / P).astype(int)
    intr = np.abs(ph) < dur / 2
    expected = max(dur / cad, 1.0)

    rows = []
    for ep in np.unique(epoch[intr]):
        m = intr & (epoch == ep)
        n = int(m.sum())
        if n < cfg.sparse_transit_points or n < cfg.min_coverage * expected:
            continue
        sig = np.sqrt(np.sum(dy[m] ** 2)) / n
        rows.append((ep, 1.0 - np.mean(f[m]), sig, n))
    dense_rows = [row for row in rows if row[3] >= cfg.min_points_in_transit]
    use_sparse = (
        len(dense_rows) < cfg.min_transits
        and len(rows) >= cfg.min_sparse_transits
        and c["snr"] >= cfg.min_sparse_snr
    )
    rows = rows if use_sparse else dense_rows
    rows = np.array(rows) if rows else np.zeros((0, 4))
    out["n_transits"] = len(rows)
    if len(rows) < cfg.min_transits:
        out["reasons"].append(f"transits with data {len(rows)} < {cfg.min_transits}")
        return out

    ep, d, s = rows[:, 0], rows[:, 1], rows[:, 2]
    alias_of, reason, warnings = classify_alias(P, dur, t0 + ep * P, accepted, cfg)
    out["warnings"].extend(warnings)
    if alias_of is not None:
        out["alias_of"] = alias_of
        out["reasons"].append(reason)
        return out

    # Red noise: scatter of out-of-transit means in duration-sized windows.
    oot = ~in_transit(t, [c], 1.5)
    b = np.floor(t[oot] / dur).astype(np.int64)
    ub, inv = np.unique(b, return_inverse=True)
    cnt = np.bincount(inv)
    mean = np.bincount(inv, f[oot]) / cnt
    good = cnt >= 3
    if good.sum() > 20:
        expect = np.median(np.bincount(inv, dy[oot] ** 2)[good] ** 0.5 / cnt[good])
        beta = max(1.0, np.std(mean[good]) / expect)
    else:
        beta = 1.0
    out["beta"] = float(beta)
    out["snr_corr"] = c["snr"] / beta
    if use_sparse and out["snr_corr"] < cfg.min_sparse_snr:
        out["reasons"].append(
            f"sparse transit SNR {out['snr_corr']:.1f} < {cfg.min_sparse_snr:.1f}"
        )
    if len(rows) >= cfg.repeated_signal_transits and out["snr_corr"] < 10:
        midpoint = (t[0] + t[-1]) / 2
        for label, half in (
            ("first", t0 + ep * P < midpoint),
            ("second", t0 + ep * P >= midpoint),
        ):
            if not np.any(half):
                out["reasons"].append(f"no transits in the {label} half of observations")
                continue
            half_weight = 1 / s[half] ** 2
            half_snr = np.sum(half_weight * d[half]) / np.sqrt(np.sum(half_weight))
            out[f"{label}_half_snr"] = float(half_snr)
            if half_snr < cfg.min_half_snr:
                out["reasons"].append(
                    f"{label} half transit SNR {half_snr:.1f} < {cfg.min_half_snr:.1f}"
                )

    # Check single-event dominance and depth consistency.
    contrib = np.clip(d / s, 0, None) ** 2
    frac = contrib.max() / contrib.sum() if contrib.sum() > 0 else 1.0
    out["max_event_fraction"] = float(frac)
    if frac > cfg.max_single_event_fraction:
        out["reasons"].append(f"one transit contributes {frac:.0%} of the signal")
    w = 1 / s**2
    mu = np.sum(w * d) / np.sum(w)
    chi2 = np.sum(((d - mu) / s) ** 2) / max(len(d) - 1, 1)
    out["depth_chi2"] = float(chi2)
    depth_limit = (
        cfg.max_depth_chi2_strong
        if len(rows) >= cfg.strong_signal_transits and out["snr_corr"] >= cfg.strong_signal_snr
        else (
            cfg.max_depth_chi2_repeated
            if len(rows) >= cfg.repeated_signal_transits
            else cfg.max_depth_chi2
        )
    )
    if chi2 > depth_limit:
        out["reasons"].append(f"transit depths are inconsistent (chi2={chi2:.1f})")

    # odd/even
    odd, even = (ep % 2 == 1), (ep % 2 == 0)
    if odd.sum() >= 1 and even.sum() >= 1:

        def wm(m):
            ww = 1 / s[m] ** 2
            return np.sum(ww * d[m]) / ww.sum(), 1 / np.sqrt(ww.sum())

        (a, ea), (bb, eb) = wm(odd), wm(even)
        sg = abs(a - bb) / np.hypot(ea, eb) / max(1.0, beta)
        out["odd_even_sigma"] = float(sg)
        if sg > cfg.max_odd_even_sigma:
            out["reasons"].append(f"odd/even transit depths differ ({sg:.1f} sigma)")

    # Secondary eclipse is a warning only.
    sec = np.abs(np.abs(ph) - P / 2) < dur / 2
    if sec.sum() >= cfg.min_points_in_transit * 2:
        dsec = 1 - np.mean(f[sec])
        esec = np.sqrt(np.sum(dy[sec] ** 2)) / sec.sum()
        if dsec / esec > 3.5 and dsec > 0.3 * c["depth"]:
            out["warnings"].append("significant secondary eclipse (possible eclipsing binary)")

    return out


def clearly_invalid_coarse_candidate(v, cfg):
    """Skip peaks that cannot survive refinement without hiding borderline signals."""
    return (
        v["alias_of"] is not None
        or v["n_transits"] < cfg.min_transits
        or v.get("max_event_fraction", 0.0) > 0.8
        or v.get("depth_chi2", 0.0) > 2 * cfg.max_depth_chi2_repeated
        or v.get("odd_even_sigma", 0.0) > 2 * cfg.max_odd_even_sigma
    )


# ============================ ITERATION ============================
def plot_periodogram(sc, it, accepted_period, cfg):
    if not cfg.plots:
        return
    plt.figure(figsize=(13, 4))
    plt.plot(sc["period"], sc["sde"], lw=0.6)
    if accepted_period:
        plt.axvline(accepted_period, color="red", ls="--", label=f"P={accepted_period:.4f}")
        plt.legend()
    plt.xscale("log")
    plt.xlabel("Period (days)")
    plt.ylabel("SDE")
    plt.title(f"Iteration {it}: periodogram")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.out_dir, f"iter{it}_periodogram.png"), dpi=110)
    plt.show() if cfg.show_plots else None
    plt.close()


def plot_fold(t, f, c, n, cfg):
    if not cfg.plots:
        return
    P, dur = c["period"], c["duration"]
    ph = phase_of(t, P, c["t0"])
    w = np.abs(ph) < max(3 * dur, 0.3)
    plt.figure(figsize=(9, 4))
    plt.scatter(ph[w] * 24, f[w], s=2, alpha=0.2)
    edges = np.linspace(-max(3 * dur, 0.3), max(3 * dur, 0.3), 31)
    mids = (edges[1:] + edges[:-1]) / 2
    med = [
        np.median(f[w & (ph >= a) & (ph < b)]) if np.any(w & (ph >= a) & (ph < b)) else np.nan
        for a, b in zip(edges[:-1], edges[1:], strict=True)
    ]
    plt.plot(mids * 24, med, "r-", lw=2)
    plt.xlabel("Hours from transit center")
    plt.ylabel("Flux")
    plt.title(f"Planet {n}: P={P:.4f} d, depth {c['depth'] * 1e6:.0f} ppm")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.out_dir, f"planet{n}_fold.png"), dpi=110)
    plt.show() if cfg.show_plots else None
    plt.close()


def calibrate(t, f, dy, cfg):
    """Estimate thresholds by block permutation, retaining some correlated noise."""
    rng = np.random.default_rng(0)
    snrs, sdes = [], []
    bl = max(int(1.0 / np.median(np.diff(t))), 5)
    for k in range(cfg.calibrate_n):
        blocks = np.array_split(np.arange(len(t)), max(len(t) // bl, 2))
        idx = np.concatenate([blocks[i] for i in rng.permutation(len(blocks))])
        sc = scan(t, f[idx], dy[idx], cfg)
        i = int(np.argmax(sc["power"]))
        snrs.append(sc["snr"][i])
        sdes.append(sc["sde"][i])
        log.info("Calibration %d/%d: SNR=%.2f SDE=%.2f", k + 1, cfg.calibrate_n, snrs[-1], sdes[-1])
    return max(cfg.min_snr, 1.05 * max(snrs)), max(cfg.min_sde, 1.05 * max(sdes))


def run_search(segments, cfg):
    os.makedirs(cfg.out_dir, exist_ok=True)
    accepted, table = [], []
    thr_snr, thr_sde = cfg.min_snr, cfg.min_sde
    status = "no candidate passed current detection criteria"

    for it in range(1, cfg.max_planets + 1):
        t, f, dy = preprocess(segments, accepted, cfg)
        log.info(
            "=" * 60 + "\nITERATION %d: %d points, noise %.0f ppm", it, len(t), np.median(dy) * 1e6
        )
        if it == 1 and cfg.calibrate_n > 0:
            thr_snr, thr_sde = calibrate(t, f, dy, cfg)
            log.info("Calibrated thresholds: SNR>%.2f, SDE>%.2f", thr_snr, thr_sde)

        sc = scan(t, f, dy, cfg)
        found = None
        vetted = 0
        for i in pick_peaks(sc, cfg.n_peaks * 16, cfg.peak_separation, t, accepted, cfg):
            if sc["sde"][i] < thr_sde:
                continue
            if (
                len(
                    observed_epochs(
                        t, float(sc["period"][i]), float(sc["t0"][i]), float(sc["duration"][i])
                    )
                )
                < cfg.min_transits
            ):
                continue
            coarse = dict(
                period=float(sc["period"][i]),
                duration=float(sc["duration"][i]),
                t0=float(sc["t0"][i]),
                depth=float(sc["depth"][i]),
                snr=float(sc["snr"][i]),
            )
            if clearly_invalid_coarse_candidate(vet(t, f, dy, coarse, accepted, cfg), cfg):
                continue
            c = refine(t, f, dy, float(sc["period"][i]), cfg)
            c["sde"] = float(sc["sde"][i])
            v = vet(t, f, dy, c, accepted, cfg)
            reasons = list(v["reasons"])
            snr_eff = v["snr_corr"] if np.isfinite(v["snr_corr"]) else c["snr"]
            if not reasons:
                if snr_eff < thr_snr:
                    reasons.append(f"SNR {snr_eff:.1f} < {thr_snr:.1f}")
                if c["sde"] < thr_sde:
                    reasons.append(f"SDE {c['sde']:.1f} < {thr_sde:.1f}")
            row = dict(
                iteration=it,
                period=c["period"],
                duration=c["duration"],
                t0=c["t0"],
                depth_ppm=c["depth"] * 1e6,
                snr=c["snr"],
                snr_corr=v["snr_corr"],
                sde=c["sde"],
                n_transits=v["n_transits"],
                status="accepted"
                if not reasons
                else ("possible_alias" if v["alias_of"] else "rejected"),
                reasons="; ".join(reasons),
                warnings="; ".join(v["warnings"]),
            )
            table.append(row)
            log.info(
                "  peak P=%.4f: SNR=%.1f (adjusted %.1f) SDE=%.1f N=%d -> %s %s",
                c["period"],
                c["snr"],
                snr_eff,
                c["sde"],
                v["n_transits"],
                row["status"],
                row["reasons"],
            )
            if not reasons:
                found = c
                break
            if v["alias_of"] is None:
                vetted += 1
                if vetted >= cfg.n_peaks:
                    break

        plot_periodogram(sc, it, found["period"] if found else None, cfg)
        if found is None:
            log.info(
                "STOP: none of %d independent peaks passed vetting. "
                "This does not prove the remaining signal is only noise.",
                cfg.n_peaks,
            )
            break
        accepted.append(found)
        plot_fold(t, f, found, len(accepted), cfg)
    else:
        status = f"reached max_planets limit={cfg.max_planets}"

    t, f, dy = preprocess(segments, accepted, cfg)  # data with detected planet models removed
    if accepted:
        accepted = joint_fit(segments, accepted, cfg)
    return accepted, pd.DataFrame(table), status, (t, f, dy)


# ============================ JOINT FIT ============================
def smooth_box(t, P, t0, dur, depth):
    x = np.abs(phase_of(t, P, t0))
    return -depth * 0.5 * (1 - np.tanh((x - dur / 2) / (0.15 * dur)))


def joint_fit(segments, planets, cfg):
    """Jointly fit all planets after transit-protected detrending."""
    try:
        t, f, dy = preprocess(segments, planets, cfg)
        for p in planets:  # Restore models subtracted during preprocessing before fitting.
            f[np.abs(phase_of(t, p["period"], p["t0"])) < p["duration"] / 2] -= p["depth"]
        near = in_transit(t, planets, 6.0)
        t, f, dy = t[near], f[near], dy[near]
        tmid = 0.5 * (t.min() + t.max())
        x0, lo, hi = [], [], []
        for p in planets:
            t0m = p["t0"] + np.round((tmid - p["t0"]) / p["period"]) * p["period"]
            x0 += [p["period"], t0m, p["duration"], p["depth"]]
            lo += [p["period"] * 0.995, t0m - p["duration"], p["duration"] * 0.5, 0.0]
            hi += [p["period"] * 1.005, t0m + p["duration"], p["duration"] * 2.0, p["depth"] * 3]
        x0 = np.array(x0)

        def resid(x):
            m = np.zeros_like(t)
            for k in range(len(planets)):
                m += smooth_box(t, *x[4 * k : 4 * k + 4])
            return (f - 1.0 - m) / dy

        scale = np.array(
            [
                p["period"] * 1e-4
                if j == 0
                else (p["duration"] * 0.1 if j < 3 else p["depth"] * 0.1)
                for p in planets
                for j in range(4)
            ]
        )
        r = least_squares(resid, np.clip(x0, lo, hi), bounds=(lo, hi), x_scale=scale)
        out = []
        for k, p in enumerate(planets):
            P, t0m, d, dep = r.x[4 * k : 4 * k + 4]
            q = dict(p)
            q.update(
                period_fit=float(P), t0_fit=float(t0m), duration_fit=float(d), depth_fit=float(dep)
            )
            out.append(q)
        log.info("Joint fit complete (chi2/N = %.3f)", 2 * r.cost / len(t))
        return out
    except Exception as ex:
        log.warning("Joint fit failed: %s", ex)
        return planets


# ============================ SYNTHETIC SELF-TEST ============================
def make_synthetic(seed=1, n_quarters=6):
    rng = np.random.default_rng(seed)
    planets = [(13.7, 3.2, 0.22, 900e-6), (41.3, 10.1, 0.30, 600e-6), (5.7, 1.1, 0.15, 450e-6)]
    segs, start = [], 0.0
    for q in range(n_quarters):
        t = np.arange(start, start + 90, 29.4 / 1440)
        f = (
            np.ones_like(t)
            + 0.003 * np.sin(2 * np.pi * t / 8.3 + q)
            + 0.002 * np.sin(2 * np.pi * t / 25 + 2 * q)
        )
        for P, t0, d, dep in planets:
            f[np.abs(phase_of(t, P, t0)) < d / 2] -= dep
        f += rng.normal(0, 250e-6, len(t))
        segs.append(dict(time=t, flux=f, err=np.full_like(t, 250e-6), id=q))
        start += 93
    return segs, planets


# ============================ MAIN ============================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default=Config.target)
    ap.add_argument("--mission", default=Config.mission)
    ap.add_argument("--author", default=Config.author)
    ap.add_argument("--cadence", default=Config.cadence)
    ap.add_argument("--jobs", type=int, default=Config().n_jobs)
    ap.add_argument("--out", default=Config.out_dir)
    ap.add_argument(
        "--calibrate", type=int, default=0, help="number of block permutations for thresholds"
    )
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cfg = Config(
        target=a.target,
        mission=a.mission,
        author=a.author,
        cadence=a.cadence or None,
        n_jobs=a.jobs,
        out_dir=a.out,
        calibrate_n=a.calibrate,
        plots=not a.no_plots,
    )

    if a.selftest:
        segs, truth = make_synthetic()
        log.info("True periods: %s", [p[0] for p in truth])
    else:
        segs = load_segments(cfg)

    planets, table, status, (t, f, dy) = run_search(segs, cfg)

    log.info("\n" + "=" * 60 + "\nRESULT: %s\n" + "=" * 60, status)
    if len(table):
        table.to_csv(os.path.join(cfg.out_dir, "all_candidates.csv"), index=False)
        pd.set_option("display.width", 200)
        log.info("\n%s", table.drop(columns=["warnings"]).to_string(index=False))
    if planets:
        pd.DataFrame(planets).to_csv(os.path.join(cfg.out_dir, "planets.csv"), index=False)
    log.info(
        "Accepted planets: %d. Residual noise %.0f ppm (expected white ~%.0f ppm)",
        len(planets),
        np.std(f) * 1e6,
        np.median(dy) * 1e6,
    )


if __name__ == "__main__":
    main()
