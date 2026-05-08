"""
flim_helpers.py
Helper functions for TCSPC-FLIM analysis (AnalyzeTCSPC.ipynb).

Functions
---------
uplowthresh        -- percentile-based display thresholds from pixel histogram
Fit_decay          -- multi-exponential decay fitting (grid search + Nelder-Mead)
Calc_Phasor_Mat    -- phasor G/S components + 2-D density matrix
phasor_ref_correction -- instrument correction via known-lifetime reference standard
read_fbs_metadata  -- parse ISS Vista .fbs.xml acquisition metadata
"""

import re
import xml.etree.ElementTree as ET
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


def Fit_decay(IRF, y, Trep, dt):
    """
    Multi-exponential TCSPC decay fit via grid search + cascade Nelder-Mead.
    Mirrors Fit_decay.m; replaces Simplex/Convol/lsqnonneg with scipy equivalents.

    Parameters
    ----------
    IRF  : raw IRF trace (1-D)
    y    : measured decay to fit (1-D)
    Trep : laser repetition period (ns)
    dt   : time bin width (ns)

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

    exp_tp = np.exp(-tp[:, np.newaxis] / tau_grid[np.newaxis, :])
    if len(tp) != n:
        tmp2 = np.zeros((n, 126))
        tmp2[:min(len(tp), n)] = exp_tp[:min(len(tp), n)]
        exp_tp = tmp2
    M0  = np.hstack([np.ones((n, 1)), _convol(irf, exp_tp)])
    cs  = M0.sum(axis=0); cs[cs == 0] = 1.0
    M0 /= cs

    c_grid   = np.arange(-5, 5.5, 0.5)
    err_grid = np.zeros(len(c_grid))
    for ki, ck in enumerate(c_grid):
        M     = frac_shift(M0, ck)
        cc    = int(np.ceil(abs(ck))) * int(np.sign(ck))
        sl    = slice(max(0, cc), min(n, n + cc))
        cx, _ = nnls(M[sl], y[sl])
        z_k   = M @ cx
        denom = np.abs(z_k); denom[denom == 0] = 1.0
        err_grid[ki] = np.sum((z_k - y) ** 2 / denom) / n

    c_fine   = np.arange(-5, 5.05, 0.1)
    err_fine = pchip_interpolate(c_grid, err_grid, c_fine)
    c_best   = float(c_fine[np.argmin(err_fine)])

    M   = frac_shift(M0, c_best)
    cc  = int(np.ceil(abs(c_best))) * int(np.sign(c_best))
    sl  = slice(max(0, cc), min(n, n + cc))
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
    n_grp = min(len(t1_list), len(t2_list))
    t1_list, t2_list = t1_list[:n_grp], t2_list[:n_grp]

    tau_guess = []
    for s, e in zip(t1_list, t2_list):
        w = cx_comp[s:e + 1]
        if w.sum() > 0:
            tau_guess.append(np.dot(w, tau_grid[s:e + 1]) / w.sum())

    tau_init = np.array([0.1] + sorted(g / dt for g in tau_guess)) / dt
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
        A_, _ = nnls(z_, y)
        zfit  = z_ @ A_
        denom = np.abs(zfit); denom[denom == 0] = 1.0
        return np.sum((y - zfit) ** 2 / denom) / (n - m)

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
    A_f, _ = nnls(z_f, y)
    model  = z_f @ A_f
    chi2   = np.sum((y - model) ** 2 / np.maximum(np.abs(model), 1e-12)) / (n - m)

    return dt * tau_f, A_f[1:], float(A_f[0]) / n, model, chi2, model


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


