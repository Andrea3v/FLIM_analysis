"""
flim_helpers.py
Helper functions for TCSPC-FLIM analysis (AnalyzeTCSPC.ipynb).

Functions
---------
uplowthresh        -- percentile-based display thresholds from pixel histogram
Fit_decay          -- multi-exponential decay fitting (grid search + Nelder-Mead)
fit_decays         -- Fit_decay on many decays in parallel worker processes
pooled_lifetimes   -- intensity- / amplitude-weighted mean lifetimes of one (pooled) Fit_decay
report_pooled_fit  -- printout + QC figure of a pooled_lifetimes result
prepare_irf        -- the IRF of every step: clean_irf (peak to target, echoes cut) + fit_irf_width when needed
clean_irf          -- IRF peak to the target bin, baseline 0, echoes beyond tail_ns cut
irf_session        -- is the IRF from the data's session (.fbs.xml date and laser period)?
fit_irf_width      -- rescale a stored IRF's width to a recording (IRF from another session)
plot_irf_width     -- QC figure of fit_irf_width
irf_fingerprint    -- short hash of an IRF (cache keys)
phasor_lifetimes   -- IRF-referenced phasor G/S, tau_phi, tau_mod of decays (per cell / timepoint)
Calc_Phasor_Mat   -- phasor G/S components + 2-D density matrix
phasor_ref_correction -- instrument correction via known-lifetime reference standard
read_fbs_metadata  -- parse ISS Vista .fbs.xml acquisition metadata
"""

import os
import re
import time
import functools
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.optimize import nnls, minimize
from scipy.interpolate import pchip_interpolate


# ── internal: circular FFT convolution ────────────────────────────────────────
def _convol(irf, X):
    n = len(irf)
    if X.ndim == 1:
        X = X[:, np.newaxis]
    H   = np.fft.rfft(irf, n=n)
    out = np.zeros((n, X.shape[1]))
    for j in range(X.shape[1]):
        out[:, j] = np.fft.irfft(H * np.fft.rfft(X[:, j], n=n), n=n)
    return out


def uplowthresh(img, upLim, loLim):
    """
    Find upper/lower display thresholds from pixel histogram.
    Mirrors uplowthresh.m: bins from 0.1 to max in 0.1-wide steps,
    walks cumulative sum to find upLim and loLim fractions.
    """
    flat = img.ravel()
    flat = flat[~np.isnan(flat)]
    max_val = flat.max() if flat.size > 0 else 0.0
    if max_val <= 0:
        return 0.0, 0.0
    edges        = np.arange(0.1, max_val + 0.1, 0.1)
    N, bin_edges = np.histogram(flat, bins=edges)
    valid        = N > 0
    left_edges   = np.round(bin_edges[:-1][valid], 1)
    N_valid      = N[valid].astype(float)
    tot          = N_valid.sum()
    cumsum       = np.cumsum(N_valid)
    hi_hit = np.where(cumsum >= upLim * tot)[0]
    hi_idx = hi_hit[0] if hi_hit.size > 0 else len(left_edges) - 1
    hi     = left_edges[hi_idx]
    lo_hit = np.where(cumsum >= loLim * tot)[0]
    lo_idx = lo_hit[0] if lo_hit.size > 0 else 0
    low    = left_edges[lo_idx]
    return hi, low


def Fit_decay(IRF, y, Trep, dt, max_components=3, t_stop_ns=None):
    """
    Multi-exponential TCSPC decay fit via grid search + cascade Nelder-Mead.
    Mirrors Fit_decay.m; replaces Simplex/Convol/lsqnonneg with scipy equivalents.

    Parameters
    ----------
    IRF           : raw IRF trace (1-D)
    y             : measured decay to fit (1-D)
    Trep          : laser repetition period (ns)
    dt            : time bin width (ns)
    max_components: hard cap on the number of exponential components returned
                    by the grid search (default 3). Prevents overfitting with
                    spurious slow components that inflate amplitude-weighted LT.
    t_stop_ns     : if set (ns), the chi² objective is evaluated only over bins
                    up to this time, excluding any artefact bump beyond it
                    (e.g. a reflection peak at ~5 ns).  None = use full window.

    Returns
    -------
    tau (ns), amp, off, z (fitted model), chi2, model
    """
    y  = np.asarray(y, dtype=float).ravel()
    n  = len(y)
    t  = np.arange(n)
    tp = dt * np.arange(1, int(round(Trep / dt)) + 1, dtype=float)

    irf = np.zeros(n)
    tmp = IRF.ravel().astype(float) - IRF.ravel()[-21:].mean()
    irf[:min(len(tmp), n)] = tmp[:n]

    tau_grid = (1.0 / dt) / np.exp(np.arange(126) / 100.0 * np.log(Trep / dt))

    def frac_shift(M, c):
        idx_f = np.mod(t - int(np.floor(c)), n)
        idx_c = np.mod(t - int(np.ceil(c)),  n)
        return (1 - c + np.floor(c)) * M[idx_f] + (c - np.floor(c)) * M[idx_c]

    exp_tp = np.exp(-tp[:, np.newaxis] * tau_grid[np.newaxis, :])
    if len(tp) != n:
        tmp2 = np.zeros((n, 126))
        tmp2[:min(len(tp), n)] = exp_tp[:min(len(tp), n)]
        exp_tp = tmp2
    M0  = np.hstack([np.ones((n, 1)), _convol(irf, exp_tp)])
    cs  = M0.sum(axis=0); cs[cs == 0] = 1.0
    M0 /= cs

    # objective window — applied to BOTH grid search and Nelder-Mead
    _n_stop = int(round(t_stop_ns / dt)) if t_stop_ns is not None else n
    _n_stop = max(1, min(_n_stop, n))

    c_grid   = np.arange(-5, 5.5, 0.5)
    err_grid = np.zeros(len(c_grid))
    for ki, ck in enumerate(c_grid):
        M     = frac_shift(M0, ck)
        cc    = int(np.ceil(abs(ck))) * int(np.sign(ck))
        sl    = slice(max(0, cc), min(_n_stop, n + cc))
        cx, _ = nnls(M[sl], y[sl])
        z_k   = M @ cx
        denom = np.abs(z_k[:_n_stop]); denom[denom == 0] = 1.0
        err_grid[ki] = np.sum((z_k[:_n_stop] - y[:_n_stop]) ** 2 / denom) / _n_stop

    c_fine   = np.arange(-5, 5.05, 0.1)
    err_fine = pchip_interpolate(c_grid, err_grid, c_fine)
    c_best   = float(c_fine[np.argmin(err_fine)])

    M   = frac_shift(M0, c_best)
    cc  = int(np.ceil(abs(c_best))) * int(np.sign(c_best))
    sl  = slice(max(0, cc), min(_n_stop, n + cc))
    cx, _ = nnls(M[sl], y[sl])

    cx_comp = cx[1:]
    active  = cx_comp > 0.05 * cx_comp.max()
    diff_a  = np.diff(active.astype(int))
    t1_list = list(np.where(diff_a ==  1)[0] + 1)
    t2_list = list(np.where(diff_a == -1)[0])
    if active[0]:  t1_list = [0] + t1_list
    if active[-1]: t2_list = t2_list + [len(active) - 1]
    while t1_list and t2_list and t1_list[0]  > t2_list[0]:  t2_list.pop(0)
    while t1_list and t2_list and t1_list[-1] > t2_list[-1]: t1_list.pop()
    n_grp = min(len(t1_list), len(t2_list), max_components)
    t1_list, t2_list = t1_list[:n_grp], t2_list[:n_grp]

    tau_guess = []
    for s, e in zip(t1_list, t2_list):
        w = cx_comp[s:e + 1]
        if w.sum() > 0:
            tau_guess.append(np.dot(w, tau_grid[s:e + 1]) / w.sum())

    tau_init = np.array([0.1 / dt] + sorted(1.0 / (g * dt) for g in tau_guess))
    m        = len(tau_init)
    tp2      = np.arange(1, n + 1, dtype=float)
    paramin  = np.concatenate([[-1.0 / dt], 0.25 * tau_init])
    paramax  = np.concatenate([[ 1.0 / dt],  5.0 * tau_init])

    def lsfit_obj(param):
        c_   = float(np.clip(param[0], paramin[0], paramax[0]))
        tau_ = np.abs(param[1:]).clip(1e-9)
        idx_f = np.mod(t - int(np.floor(c_)), n)
        idx_c = np.mod(t - int(np.ceil(c_)),  n)
        irs   = (1 - c_ + np.floor(c_)) * irf[idx_f] + (c_ - np.floor(c_)) * irf[idx_c]
        pc    = 1.0 / (1.0 - np.exp(-n / tau_))
        pc[~np.isfinite(pc)] = 1.0
        x_    = np.exp(-(tp2 - 1)[:, np.newaxis] / tau_) * pc
        z_    = np.hstack([np.ones((n, 1)), _convol(irs, x_)])
        cs    = z_.sum(axis=0); cs[cs == 0] = 1.0
        z_   /= cs
        A_, _ = nnls(z_[:_n_stop], y[:_n_stop])
        zfit  = z_ @ A_
        denom = np.abs(zfit[:_n_stop]); denom[denom == 0] = 1.0
        return np.sum((y[:_n_stop] - zfit[:_n_stop]) ** 2 / denom) / (_n_stop - m)

    rng0       = np.random.default_rng(0)
    best_param = np.concatenate([[0.0], tau_init])
    best_err   = np.inf

    for casc in range(1, 11):
        sub_p = np.zeros((len(best_param), 10))
        sub_e = np.full(10, np.inf)
        r0    = best_param.copy()
        for sub in range(10):
            rf = np.clip(r0 * 2.0 ** (1.2 * (rng0.random(len(r0)) - 0.5) / casc),
                         paramin, paramax)
            res = minimize(lsfit_obj, rf, method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6})
            sub_p[:, sub] = np.clip(res.x, paramin, paramax)
            sub_e[sub]    = lsfit_obj(sub_p[:, sub])
        best_sub = int(np.argmin(sub_e))
        if sub_e[best_sub] < best_err:
            best_err   = sub_e[best_sub]
            best_param = sub_p[:, best_sub]

    res_f = minimize(lsfit_obj, best_param, method='Nelder-Mead',
                     options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8})
    pf    = np.clip(res_f.x, paramin, paramax)
    c_f   = float(pf[0])
    tau_f = np.abs(pf[1:])

    idx_f  = np.mod(t - int(np.floor(c_f)), n)
    idx_c  = np.mod(t - int(np.ceil(c_f)),  n)
    irs_f  = (1 - c_f + np.floor(c_f)) * irf[idx_f] + (c_f - np.floor(c_f)) * irf[idx_c]
    pc_f   = 1.0 / (1.0 - np.exp(-n / tau_f))
    pc_f[~np.isfinite(pc_f)] = 1.0
    x_f    = np.exp(-(tp2 - 1)[:, np.newaxis] / tau_f) * pc_f
    z_f    = np.hstack([np.ones((n, 1)), _convol(irs_f, x_f)])
    cs     = z_f.sum(axis=0); cs[cs == 0] = 1.0
    z_f   /= cs
    A_f, _ = nnls(z_f[:_n_stop], y[:_n_stop])
    model  = z_f @ A_f
    chi2   = np.sum((y[:_n_stop] - model[:_n_stop]) ** 2 /
                    np.maximum(np.abs(model[:_n_stop]), 1e-12)) / (_n_stop - m)

    return dt * tau_f, A_f[1:], float(A_f[0]) / n, model, chi2, model


def pooled_lifetimes(IRF, y, Trep, dt, t_stop_ns=None, max_components=3, fast_ns=0.05, fit=None):
    """
    Lifetime summary of one Fit_decay on a pooled decay (e.g. all cells of a z-stack or of one timepoint).

    Fit_decay scales every component curve to unit area, so its `amp` is the number of photons per component:
      tau_int = sum(n_i tau_i) / sum(n_i)                      intensity-weighted mean lifetime
      tau_amp = sum(a_i tau_i) / sum(a_i),  a_i = n_i (1 - exp(-dt / tau_i))   amplitude-weighted mean lifetime
    (a_i is exact for Fit_decay's periodic exponentials: a unit-amplitude one has area IRF area / (1 - exp(-dt/tau)).)
    Components shorter than fast_ns are IRF-shaped (on the 2026-11 brain-slice data, with an IRF broader than the
    recording's, Fit_decay's fast slot went to its 0.025 ns bound); they are left out of both means and their photon
    share is returned as fast_frac. fast_ns = 0.05 ns lies between that bound and the slot's start value (0.1 ns), so
    a component pinned at the bound is always flagged and one that stays near its start is never split by the cutoff.

    fit : an existing Fit_decay result (tau, amp, off, model, chi2, ...) to summarise instead of fitting y.
    Returns a flat dict: tau_int, tau_amp, fast_frac, chi2, tau1.. / frac1.. (slow components, sorted by lifetime;
    frac = share of all photons in decay components, background excluded: frac1 + frac2 + ... + fast_frac = 1)
    and model (the fitted curve).
    """
    if fit is None:
        fit = Fit_decay(IRF, y, Trep, dt, max_components=max_components, t_stop_ns=t_stop_ns)
    tau, amp, _, model, chi2 = fit[:5]
    tau = np.asarray(tau, dtype=float).ravel()
    amp = np.clip(np.asarray(amp, dtype=float).ravel(), 0.0, None)
    slow = (tau >= fast_ns) & (amp > 0)
    total = amp.sum()
    out = dict(tau_int=np.nan, tau_amp=np.nan,
               fast_frac=float(amp[~slow].sum() / total) if total > 0 else np.nan,
               chi2=float(chi2))
    if slow.any():
        n, t = amp[slow], tau[slow]
        a = n * (1.0 - np.exp(-dt / t))
        out.update(tau_int=float(n @ t / n.sum()), tau_amp=float(a @ t / a.sum()))
        for i, k in enumerate(np.argsort(t), 1):
            out[f'tau{i}'] = float(t[k])
            out[f'frac{i}'] = float(n[k] / total)
    out['model'] = np.asarray(model, dtype=float)
    return out


def report_pooled_fit(pf, tau, decay, irf, Tcycle, t_stop_ns=None, label='', png_path=None, fast_note=0.02):
    """
    Print the summary of a pooled_lifetimes result and draw its QC figure (fit, IRF; normalised residuals).
    tau: time axis (ns); decay: the pooled decay; irf: the IRF of the fit; png_path: where the figure is saved.
    """
    comp = [(pf[f'tau{i}'], pf[f'frac{i}']) for i in range(1, 6) if np.isfinite(pf.get(f'tau{i}', np.nan))]
    print(f'{label}: tau_int {pf["tau_int"]:.3f} ns   tau_amp {pf["tau_amp"]:.3f} ns   chi2 {pf["chi2"]:.1f}')
    print('  components (share of photons): ' + ',  '.join(f'{t:.3f} ns ({100 * f:.0f}%)' for t, f in comp)
          + (f',  IRF-shaped {100 * pf["fast_frac"]:.0f}%' if pf['fast_frac'] > 0.005 else ''))
    if pf['fast_frac'] > fast_note:
        print(f'  NOTE: {100 * pf["fast_frac"]:.0f}% of the photons are in an IRF-shaped component (< 0.05 ns, left out '
              f'of tau_int / tau_amp): the IRF is probably broader than this recording\'s (Step 5: irf_width_mode).')
    decay = np.asarray(decay, dtype=float)
    pm = np.asarray(pf['model'], dtype=float)
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
    ax[0].semilogy(tau, decay / decay.max(), '.', ms=3, color='steelblue', label='pooled decay')
    ax[0].semilogy(tau, pm / decay.max(), color='tomato', lw=1.0, label='Fit_decay')
    ax[0].semilogy(tau, irf / irf.max(), '--k', lw=0.6, label='IRF')
    ax[0].set(xlabel='time / ns', ylabel='intensity (norm.)', xlim=(0, Tcycle), ylim=(5e-3, 1.3))
    ax[1].plot(tau, (decay - pm) / np.sqrt(np.maximum(pm, 1.0)), color='0.3', lw=0.8)
    ax[1].axhline(0, color='tomato', lw=0.6)
    ax[1].set(xlabel='time / ns', ylabel='(data − fit) / √fit', xlim=(0, Tcycle))
    if t_stop_ns:
        for a in ax:
            a.axvline(t_stop_ns, color='red', ls='--', lw=0.8)
    ax[0].legend(fontsize=8)
    fig.suptitle(f'{label}   tau_int {pf["tau_int"]:.3f} ns,  tau_amp {pf["tau_amp"]:.3f} ns,  '
                 f'fast {100 * pf["fast_frac"]:.1f}%,  chi2 {pf["chi2"]:.1f}', fontsize=10)
    fig.tight_layout()
    if png_path:
        fig.savefig(png_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.show()
    return fig


# ── IRF: one preparation for every step (Step 5) ──────────────────────────────
def irf_fingerprint(irf):
    """Short hash of an IRF array. Stored with cached results so they are recomputed when the IRF changes."""
    import hashlib
    return hashlib.sha1(np.ascontiguousarray(irf, dtype=np.float64).tobytes()).hexdigest()[:12]


def clean_irf(IRF, target, dt, tail_ns=2.5, n_base=25):
    """
    1-D IRF for the analysis: peak moved to bin `target` (circular shift), baseline subtracted, negatives set to 0,
    and everything further than tail_ns from the peak (circular distance) set to 0.

    The cut removes the echoes of the urea-SHG IRF recordings (~3.4 and ~4.9 ns after the peak; the slice decays
    have none) and the noise of the IRF's baseline. Baseline: median of the bins beyond tail_ns (robust to the
    echoes). tail_ns=None keeps the IRF as measured, baseline = median of the last n_base bins after the shift.
    """
    ir = np.asarray(IRF, dtype=float).ravel()
    n = ir.size
    ir = np.roll(ir, int(target) - int(np.argmax(ir)))
    if tail_ns is None:
        return np.clip(ir - np.median(ir[-n_base:]), 0.0, None)
    far = np.abs((np.arange(n) - int(target) + n // 2) % n - n // 2) > tail_ns / dt
    ir = ir - np.median(ir[far])
    ir[far] = 0.0
    return np.clip(ir, 0.0, None)


def irf_session(meta_irf, meta, tol_ns=0.01):
    """(same_session, reason): the IRF is from the data's session when both .fbs.xml exist with the same date and
    laser period (within tol_ns)."""
    d_irf = ((meta_irf or {}).get('datetime') or '')[:10]
    d_dat = ((meta or {}).get('datetime') or '')[:10]
    same = bool(d_irf and d_irf == d_dat
                and abs(((meta_irf or {}).get('Tcycle') or 0) - ((meta or {}).get('Tcycle') or 0)) < tol_ns)
    why = ('IRF from the same session (.fbs.xml dates and laser period match)' if same else
           'IRF has no .fbs.xml' if not meta_irf else f'IRF of {d_irf or "?"}, data of {d_dat or "?"}')
    return same, why


def fit_irf_width(IRF, y, dt, win_ns=3.0, n_exp=2, s_bounds=(0.3, 2.0), min_gain=0.2):
    """
    Rescale the width of a stored IRF to the instrument response of a recording.

    A stored IRF (other day, other settings, or without its .fbs.xml) can be broader or narrower than the response
    in the data. On the 2026-11-10 brain slices the urea-SHG IRFs of 2 and 6 Nov were too broad (fitted factor
    0.54-0.66): the data rose faster than any decay convolved with them, and Fit_decay compensated with a ~0.025 ns
    component holding ~9 % of the photons.

    Model, on the bins up to `win_ns` after the peak of y (a reflection bump further out does not enter):
        y = background + sum_k A_k * (IRF_s,c  (x) periodic exp(-t / tau_k)),   k = 1..n_exp
    with IRF_s,c the IRF stretched in time by s about its (sub-bin) peak and shifted by c bins. The lifetimes, A_k
    and the background are nuisance parameters (A by NNLS); s, c and tau_k by Nelder-Mead from several starts.
    The same fit with s = 1 gives the chi2 of the IRF as given.

    IRF : 1-D, prepared by clean_irf (baseline 0, echoes cut); y : 1-D, same bins (dt ns), peaks near each other.
    Returns dict:
        s, c                       : width factor (< 1: narrower) and shift (bins) of the best fit
        chi2_measured, chi2_fitted : reduced Pearson chi2 with s = 1 and with the fitted s
        apply                      : chi2_fitted < (1 - min_gain) * chi2_measured
        irf                        : the IRF rescaled by s about its own peak (shift NOT applied), same area
        irf_measured               : the IRF as given (s = 1)
        stop                       : fit window = bins [0, stop)
        model_measured, model_fitted : fitted curves (full length) for a QC plot
        taus                       : nuisance lifetimes (ns) of the fitted model
    """
    y = np.asarray(y, dtype=float).ravel()
    n = y.size
    T = np.arange(n, dtype=float)
    ir = np.asarray(IRF, dtype=float).ravel()
    if ir.size != n:
        raise ValueError(f'IRF has {ir.size} bins, the data {n}')
    pk = int(np.argmax(ir))
    a, b, c3 = ir[pk - 1], ir[pk], ir[(pk + 1) % n]
    den = a - 2 * b + c3
    tpk = pk + (0.5 * (a - c3) / den if den != 0 else 0.0)        # sub-bin peak (parabola through the top 3 bins)
    area = ir.sum()

    def irf_sc(s, c):
        out = np.interp((T - tpk - c) / s + tpk, T, ir, left=0.0, right=0.0)
        return out * (area / out.sum()) if out.sum() > 0 else out

    stop = min(n, int(np.argmax(y)) + int(round(win_ns / dt)) + 1)

    def model(s, c, taus):
        h = np.fft.rfft(irf_sc(s, c))
        cols = [np.ones(n)]
        for tau in taus:
            x = np.exp(-T * dt / tau) / (1.0 - np.exp(-n * dt / tau))
            cols.append(np.fft.irfft(h * np.fft.rfft(x), n=n))
        M = np.array(cols).T
        M /= np.maximum(M[:stop].sum(axis=0), 1e-12)
        A, _ = nnls(M[:stop], y[:stop])
        return M @ A

    def chi2(p, s_fixed):
        c = p[0]
        s = s_fixed if s_fixed is not None else p[1]
        taus = 0.05 + np.abs(p[-n_exp:])
        if not (s_bounds[0] < s < s_bounds[1]) or abs(c) > 10:
            return 1e12
        z = model(s, c, taus)
        return float(np.sum((y[:stop] - z[:stop]) ** 2 / np.maximum(z[:stop], 1.0)) / max(1, stop - n_exp - 3))

    tau0 = list(np.linspace(0.5, 2.5, n_exp))
    res = {}
    for key, s_fixed in (('measured', 1.0), ('fitted', None)):
        s_starts = (1.0,) if s_fixed is not None else (0.6, 0.8, 1.0, 1.3)
        best = None
        for c0 in (-1.0, 0.0, 1.0):
            for s0 in s_starts:
                x0 = [c0] + ([s0] if s_fixed is None else []) + tau0
                r = minimize(chi2, x0, args=(s_fixed,), method='Nelder-Mead',
                             options={'maxiter': 3000, 'xatol': 1e-5, 'fatol': 1e-7})
                if best is None or r.fun < best.fun:
                    best = r
        s = s_fixed if s_fixed is not None else float(best.x[1])
        taus = 0.05 + np.abs(best.x[-n_exp:])
        res[key] = dict(chi2=float(best.fun), s=float(s), c=float(best.x[0]), taus=taus,
                        model=model(s, best.x[0], taus))
    f, m = res['fitted'], res['measured']
    return dict(s=f['s'], c=f['c'], chi2_measured=m['chi2'], chi2_fitted=f['chi2'],
                apply=bool(f['chi2'] < (1.0 - min_gain) * m['chi2']),
                irf=irf_sc(f['s'], 0.0), irf_measured=ir.copy(), stop=stop,
                model_measured=m['model'], model_fitted=f['model'], taus=f['taus'])


def prepare_irf(IRF, y, dt, target, tail_ns=2.5, width_mode='auto', same_session=False, win_ns=3.0, min_gain=0.2):
    """
    The IRF used by every later step (tav reference H, phasor reference, all fits), built in one place from the IRF
    as loaded. The input is never modified, so re-running gives the same result until another IRF is loaded.
      1. clean_irf: peak -> bin target, baseline 0, 0 beyond tail_ns from the peak (the IRF recording's echoes)
      2. width: fit_irf_width when width_mode == 'always', or 'auto' and not same_session; the rescaled IRF is used
         when chi2 drops by more than min_gain (20 %). 'off': never.
    y : data decay with its peak at bin `target`.
    Returns dict: irf (to use), irf_clean, width (fit_irf_width result, or None when not fitted),
    key (irf_fingerprint of irf).
    """
    irf_c = clean_irf(IRF, target, dt, tail_ns=tail_ns)
    width = None
    if width_mode == 'always' or (width_mode == 'auto' and not same_session):
        width = fit_irf_width(irf_c, y, dt, win_ns=win_ns, min_gain=min_gain)
    irf = width['irf'] if width is not None and width['apply'] else irf_c
    return dict(irf=irf, irf_clean=irf_c, width=width, key=irf_fingerprint(irf))


def plot_irf_width(width, tau, y, target, win_ns, title=''):
    """QC figure of fit_irf_width: rising edge (IRF as given, rescaled IRF, data); residuals of both fits."""
    w = width
    nbin = len(tau)
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.6))
    ax[0].plot(tau, w['irf_measured'] / w['irf_measured'].max(), '--k', lw=1, label='IRF (echoes cut)')
    ax[0].plot(tau, w['irf'] / w['irf'].max(), color='C3', lw=1.2, label=f'IRF x {w["s"]:.2f} (width)')
    ax[0].plot(tau, y / y.max(), '.', color='steelblue', ms=4, label='data (middle plane)')
    ax[0].set(xlim=(tau[max(target - 6, 0)], tau[min(target + 25, nbin - 1)]), xlabel='time / ns',
              ylabel='intensity (norm.)', title='rising edge')
    st = w['stop']
    for m, col, lbl in ((w['model_measured'], 'k', 'IRF as measured'), (w['model_fitted'], 'C3', 'rescaled IRF')):
        ax[1].plot(tau[:st], ((y - m) / np.sqrt(np.maximum(m, 1.0)))[:st], color=col, lw=1, label=lbl)
    ax[1].axhline(0, color='0.6', lw=0.5)
    ax[1].set(xlabel='time / ns', ylabel='(data − model) / √model', title=f'fit window: to {win_ns} ns after the peak')
    for a in ax:
        a.legend(fontsize=8)
    fig.suptitle(f'IRF width  {title}   factor {w["s"]:.3f},  chi2 {w["chi2_measured"]:.0f} → {w["chi2_fitted"]:.0f}'
                 + ('' if w['apply'] else '   (not applied)'), fontsize=10)
    fig.tight_layout()
    plt.show()
    return fig


def phasor_lifetimes(decays, IRF, dt, Trep, min_photons=1):
    """
    Phasor of decays (one per row), referenced to the IRF as in Step 11: phase 0 at the IRF's mean arrival time,
    the IRF's own phasor mapped to (1, 0) (phasor_ref_correction with a 0 ns reference).
    Returns G, S, tau_phi (phase lifetime S / (ωG): needs only the IRF timing) and tau_mod (modulation lifetime:
    depends on the IRF width), in ns; NaN for rows with fewer than min_photons.
    """
    D = np.atleast_2d(np.asarray(decays, dtype=float))
    irf = np.asarray(IRF, dtype=float).ravel()
    tp = (np.arange(1, D.shape[1] + 1) - 0.5) * dt
    T = 2.0 * np.pi * (tp - tp @ irf / irf.sum()) / Trep
    nph = D.sum(axis=1)
    ok = nph >= max(min_photons, 1)
    w = D / np.maximum(nph, 1.0)[:, np.newaxis]
    g = np.where(ok, w @ np.cos(T), np.nan)
    s = np.where(ok, w @ np.sin(T), np.nan)
    G, S, tau_mod, tau_phi = phasor_ref_correction(
        g, s, np.array([[np.cos(T) @ irf / irf.sum()]]), np.array([[np.sin(T) @ irf / irf.sum()]]),
        tau_ref_ns=0.0, freq_mhz=1000.0 / Trep)
    return G, S, tau_phi, tau_mod


def _fit_row(IRF, Trep, dt, max_components, t_stop_ns, y):
    # worker task of fit_decays (module level, so the spawned processes can import it)
    return Fit_decay(IRF, y, Trep, dt, max_components=max_components, t_stop_ns=t_stop_ns)[:5]


def fit_decays(IRF, decays, Trep, dt, max_components=3, t_stop_ns=None, n_workers=None, verbose=True):
    """
    Fit_decay on every row of `decays` (n, nbin), spread over worker processes.

    Results are identical to calling Fit_decay row by row (its random restarts use a fixed seed),
    and come back in row order. n_workers: None = all logical cores but one; 1 = sequential, in this process.
    Prints progress about every 10 % when verbose.

    Returns
    -------
    list of (tau (ns), amp, off, model, chi2) per row
    """
    decays = np.asarray(decays, dtype=float)
    if decays.size == 0:
        return []
    decays = decays.reshape(-1, decays.shape[-1])
    n = len(decays)
    n_workers = max(1, min(n, n_workers or (os.cpu_count() or 2) - 1))
    task = functools.partial(_fit_row, np.asarray(IRF, dtype=float), Trep, dt, max_components, t_stop_ns)
    out, t0, step = [None] * n, time.perf_counter(), max(1, n // 10)

    def _progress(done):
        if verbose and (done % step == 0 or done == n):
            el = time.perf_counter() - t0
            left = f'  ~{el / done * (n - done) / 60:.1f} min left' if done < n else ''
            print(f'  fitted {done}/{n}  ({el:.0f} s{left})', end='\n' if done == n else '\r', flush=True)

    if n_workers == 1:
        for i, y in enumerate(decays):
            out[i] = task(y)
            _progress(i + 1)
        return out
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(task, y): i for i, y in enumerate(decays)}
        for done, f in enumerate(as_completed(futs), 1):
            out[futs[f]] = f.result()
            _progress(done)
    return out


def Calc_Phasor_Mat(IRF2, Data, Trep, dt, wdth, fname, min_count=10, spatial_bin=1):
    """
    Compute phasor components G (cos) and S (sin) for each pixel and
    build the 2-D density matrix M. Mirrors Calc_Phasor_Mat.m.

    Parameters
    ----------
    IRF2        : preprocessed IRF (background-subtracted, 1-D)
    Data        : TCSPC image  (nx, ny, nbin)  or  (nx, ny, nbin, nch)
    Trep        : repetition period (ns)
    dt          : time bin width (ns)
    wdth        : phasor histogram bin width (e.g. 0.005 -> 201x201 grid)
    fname       : output figure path (None to skip saving)
    min_count   : pixels with fewer photons are excluded (pass threshold)
    spatial_bin : NxN pixel binning before phasor computation (1 = none).
                  Sums NxN blocks -> N^2 more photons per point, lower spatial res.

    Returns
    -------
    S, G  -- phasor components (nx, ny),  M -- density matrix
    """
    if Data.ndim == 4:
        Data = Data.squeeze(axis=3) if Data.shape[3] == 1 else Data.sum(axis=3)
    Data = Data.astype(float)
    nx, ny, n = Data.shape

    if spatial_bin > 1:
        nx_b  = (nx // spatial_bin) * spatial_bin
        ny_b  = (ny // spatial_bin) * spatial_bin
        Data  = Data[:nx_b, :ny_b, :]
        Data  = (Data.reshape(nx_b // spatial_bin, spatial_bin,
                              ny_b // spatial_bin, spatial_bin, n)
                     .sum(axis=(1, 3)))
        nx, ny, n = Data.shape

    tag   = Data.sum(axis=2)
    tau_p = (np.arange(1, n + 1) - 0.5) * dt
    ph    = np.dot(tau_p, IRF2) / IRF2.sum() if IRF2.sum() != 0 else 0.0
    T     = 2.0 * np.pi * (tau_p - ph) / Trep
    T3    = T[np.newaxis, np.newaxis, :]

    tag_safe = np.where(tag > 0, tag, np.nan)
    G = (np.cos(T3) * Data).sum(axis=2) / tag_safe
    S = (np.sin(T3) * Data).sum(axis=2) / tag_safe

    ind    = (tag < min_count) | (G < 0) | (S < 0)
    G[ind] = np.nan
    S[ind] = np.nan

    Xv, Yv = G.ravel(), S.ravel()
    valid   = ~(np.isnan(Xv) | np.isnan(Yv))
    Xv, Yv = Xv[valid], Yv[valid]

    nbin_p = int(round(1.0 / wdth)) + 1
    X_idx  = np.round(Xv / wdth).astype(int)
    Y_idx  = np.round(Yv / wdth).astype(int)
    M      = np.zeros((nbin_p, nbin_p), dtype=float)
    ok     = (X_idx >= 0) & (X_idx < nbin_p) & (Y_idx >= 0) & (Y_idx < nbin_p)
    np.add.at(M, (Y_idx[ok], X_idx[ok]), 1)

    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    if M.max() > 0:
        vmax = np.percentile(M[M > 0], 99)
        ax.imshow(np.flipud(M), extent=[0, 1, 0, 1], aspect='equal',
                  cmap='hot', origin='upper', vmax=vmax)
    theta = np.linspace(0, np.pi, 300)
    ax.plot(0.5 + 0.5 * np.cos(theta), 0.5 * np.sin(theta),
            '-', color=(0.6, 0, 0), linewidth=1.5)
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.0)
    ax.set_aspect('equal')
    ax.set_xlabel('phasor G', fontweight='bold', fontsize=16,
                  fontstyle='italic', color='white')
    ax.set_ylabel('phasor S', fontweight='bold', fontsize=16,
                  fontstyle='italic', color='white')
    ax.tick_params(colors='white', labelsize=12)
    bin_label = f'{spatial_bin}x{spatial_bin} px' if spatial_bin > 1 else 'no binning'
    ax.set_title(f'{bin_label}  |  {valid.sum()} points',
                 color='white', fontsize=10)
    for spine in ax.spines.values():
        spine.set_edgecolor('white')
    plt.tight_layout()
    if fname:
        plt.savefig(fname, facecolor='black', bbox_inches='tight')
    plt.show()

    return S, G, M


def phasor_ref_correction(g_sample, s_sample, g_ref, s_ref, tau_ref_ns, freq_mhz):
    """
    Correct sample phasor coordinates using a fluorescence reference standard.
    Ports the reference correction from FLIMPA (github.com/SofiaKapsiani/FLIMPA).

    Corrects for modulation error (detector/optics frequency response) and
    phase offset (timing delay in electronics/optics).

    Parameters
    ----------
    g_sample, s_sample : ndarray (ny, nx)
        Sample phasor G/S coordinates (NaN for invalid pixels).
    g_ref, s_ref       : ndarray
        Reference standard phasor G/S from same instrument session.
    tau_ref_ns         : float
        Known lifetime of the reference standard (ns).
        Common: Rhodamine 6G ~4.08 ns, Fluorescein ~4.0 ns, POPOP ~1.35 ns.
    freq_mhz           : float
        Laser repetition frequency (MHz), e.g. 1000 / Tcycle.

    Returns
    -------
    g_corr, s_corr : ndarray  -- corrected phasor coordinates
    tau_mod        : ndarray  -- modulation lifetime map (ns)
    tau_phi        : ndarray  -- phase lifetime map (ns)
    """
    omega   = 2.0 * np.pi * freq_mhz * 1e6
    tau_ref = tau_ref_ns * 1e-9

    g_ref_m  = float(np.nanmean(g_ref))
    s_ref_m  = float(np.nanmean(s_ref))
    mod_exp  = 1.0 / np.sqrt(1.0 + (omega * tau_ref) ** 2)
    phi_exp  = np.arctan(omega * tau_ref)
    mod_meas = np.sqrt(g_ref_m ** 2 + s_ref_m ** 2)
    phi_meas = np.arctan2(s_ref_m, g_ref_m)
    M_cor    = mod_exp / mod_meas if mod_meas > 0 else 1.0
    phi_cor  = phi_exp - phi_meas

    print(f'Ref correction:  M_cor = {M_cor:.4f},  phi_cor = {np.degrees(phi_cor):.2f} deg')

    mod_s  = np.sqrt(g_sample ** 2 + s_sample ** 2) * M_cor
    phi_s  = np.arctan2(s_sample, g_sample) + phi_cor
    g_corr = mod_s * np.cos(phi_s)
    s_corr = mod_s * np.sin(phi_s)

    with np.errstate(invalid='ignore', divide='ignore'):
        tau_mod = np.sqrt(np.maximum(1.0 / mod_s ** 2 - 1.0, 0.0)) / omega * 1e9

    with np.errstate(invalid='ignore'):
        tau_phi = np.tan(phi_s) / omega * 1e9
        tau_phi = np.where(tau_phi > 0, tau_phi, np.nan)

    return g_corr, s_corr, tau_mod, tau_phi


def read_fbs_metadata(xml_path):
    """
    Parse an ISS Vista .fbs.xml file and return relevant acquisition parameters.

    Parameters
    ----------
    xml_path : str  path to the .fbs.xml file

    Returns
    -------
    dict with keys:
        datetime       : str    acquisition timestamp
        nx, ny, nz     : int    image dimensions (pixels)
        channels       : int    number of detection channels
        nbin           : int    TCSPC time bins  (AdcResolution)
        Tcycle         : float  laser repetition period (ns)  — use as Tcycle
        freq_mhz       : float  laser repetition frequency (MHz)
        dt             : float  time bin width (ns)  = Tcycle / nbin
        dwell_time_ms  : float  pixel dwell time (ms)
        pixel_size_um  : float  lateral pixel size (µm/px)
        z_step_um      : float  axial step between slices (µm); 0 if single plane
        scan_plane     : str    e.g. 'XY'
        excitation_nm  : int or None   laser wavelength (nm)
        detector_gain  : int or None   detector gain (%)
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    def _get(path, cast=str, default=None):
        el = root.find(path)
        if el is None or el.text is None:
            return default
        try:
            return cast(el.text.strip())
        except (ValueError, TypeError):
            return default

    # ── spatial & scan params ────────────────────────────────────────────────
    nx = _get('ScanParams/XPixels', int)
    ny = _get('ScanParams/YPixels', int)
    nz = _get('ScanParams/ZPixels', int, default=1)

    x_start = _get('ScanParams/ImageRegion/XStart', float, 0.0)
    x_end   = _get('ScanParams/ImageRegion/XEnd',   float, 0.0)
    y_start = _get('ScanParams/ImageRegion/YStart', float, 0.0)
    y_end   = _get('ScanParams/ImageRegion/YEnd',   float, 0.0)
    z_start = _get('ScanParams/ImageRegion/ZStart', float, 0.0)
    z_end   = _get('ScanParams/ImageRegion/ZEnd',   float, 0.0)

    pixel_size_um = (x_end - x_start) / nx if nx else None
    z_step_um     = (z_end - z_start) / (nz - 1) if nz and nz > 1 else 0.0

    # ── TCSPC params ─────────────────────────────────────────────────────────
    nbin   = _get('PhotonCountingSettings/AdcResolution',       int)
    Tcycle = _get('PhotonCountingSettings/TacTimeRange',        float)   # ns
    macro_freq = _get('PhotonCountingSettings/MacroTimeClockFrequency', float)

    freq_mhz = macro_freq / 1e6 if macro_freq else (1000.0 / Tcycle if Tcycle else None)
    dt       = Tcycle / nbin if (Tcycle and nbin) else None

    # ── other acquisition params ─────────────────────────────────────────────
    dwell_time_ms = _get('ScanParams/PixelDwellTime', float)
    channels      = _get('ScanParams/Channels',       int)
    scan_plane    = _get('ScanParams/ScannerInfo/FrameScanPlane', str)
    datetime_str  = _get('DateTimeStamp', str)

    # ── optional: parse laser wavelength and gain from free-text comments ────
    excitation_nm = None
    detector_gain = None
    for el in root.findall('ItemizedSystemSettings/fromComments'):
        text = (el.text or '').strip()
        if excitation_nm is None:
            m = re.search(r'Wavelength.*?:\s*(\d+)\s*nm', text)
            if m:
                excitation_nm = int(m.group(1))
        if detector_gain is None:
            m = re.search(r'Gain\s*-.*?:\s*(\d+)\s*%', text)
            if m:
                detector_gain = int(m.group(1))

    return {
        'datetime':      datetime_str,
        'nx':            nx,
        'ny':            ny,
        'nz':            nz,
        'channels':      channels,
        'nbin':          nbin,
        'Tcycle':        Tcycle,
        'freq_mhz':      freq_mhz,
        'dt':            dt,
        'dwell_time_ms': dwell_time_ms,
        'pixel_size_um': pixel_size_um,
        'z_step_um':     z_step_um,
        'scan_plane':    scan_plane,
        'excitation_nm': excitation_nm,
        'detector_gain': detector_gain,
    }


