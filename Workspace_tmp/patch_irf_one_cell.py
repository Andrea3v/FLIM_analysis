"""One IRF cell, code-review fixes, acquisition-folder picker + raw deletion (2026-11-24).

All three AnalyzeTCSPC notebooks:
  - Step 4 (IRF preprocessing) merged into Step 5: one cell builds the data alignment and the IRF of every later step
    from what Step 1 loaded (data_raw, IRF), via flim_helpers.prepare_irf (clean_irf: peak to target, baseline 0,
    echoes beyond irf_tail_ns cut; fit_irf_width when the IRF is not from the same session). Re-running it gives the
    same IRF2 (review: re-running the old cell fitted the already-rescaled IRF and overwrote the record).
  - Step 14 no longer re-shifts or flattens the IRF (IRF2_fit gone: Steps 8, 11, 13, 14, 15, 17 all use IRF2; review:
    the echoes were kept by some steps and cut by others, 0.03-0.045 ns offset in H). Its caches store irf_key and are
    recomputed when the IRF changes (review). Phasor per cell via flim_helpers.phasor_lifetimes; pooled-fit printout +
    figure via flim_helpers.report_pooled_fit (component shares now of all photons: review).
  - Step 15 plots the Step 14 phasor columns instead of recomputing them (review: duplicate).
  - Step 17 (Time): IRF2, phasor via phasor_lifetimes (no Step 11 G_ref needed).
XYTZTime / XYTZTile notebooks, Step 0:
  - the picker selects the acquisition folder; vt.find_raw finds the raw Vista TIFF in it, or its name from the uint8
    conversion report once the raw file was deleted;
  - delete_raw_after_convert (Step 0a, default True): Step 0b deletes the float32 raw TIFF via
    vt.delete_raw_if_lossless (no clipped counts, tiles read back and verified);
  - Step 2b session date from the .fbs.xml (the folder name of a download, '...-20261124T025404Z-...', was read as the date).

Idempotent (marker: 'prepare_irf'). Usage: python Workspace_tmp/patch_irf_one_cell.py [notebook ...]
"""
import json, os, re, sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NBS = [os.path.join(HERE, n) for n in ('AnalyzeTCSPC.ipynb', 'AnalyzeTCSPC_XYTZTile.ipynb', 'AnalyzeTCSPC_XYTZTime.ipynb')]


def cell(nb, cid):
    return next(c for c in nb['cells'] if c.get('id') == cid)


def src(nb, cid):
    return ''.join(cell(nb, cid)['source'])


def put(nb, cid, s):
    cell(nb, cid)['source'] = s.splitlines(keepends=True)


def sub(s, old, new, where, count=1):
    n = s.count(old)
    if n != count:
        sys.exit(f'{where}: expected {count} x {old[:70]!r}, found {n}; nothing written')
    return s.replace(old, new)


# ════════════════════════════════════════════════════════════════════════════
# Step 1 parameters
# ════════════════════════════════════════════════════════════════════════════
P1_OLD = "# ── IRF width (Step 5) ────────────────────────────────────────────────────────\n"
P1_NEW = """# ── IRF (Step 5: one cell prepares it for every later step) ─────────────────
irf_tail_ns      = 2.5         # IRF kept within ±irf_tail_ns of its peak, 0 beyond: removes the echoes of the urea-SHG
                               # IRF recordings (~3.4 and ~4.9 ns after the peak; the slice decays have none). None = as measured
"""

# ════════════════════════════════════════════════════════════════════════════
# Step 5: markdown + code
# ════════════════════════════════════════════════════════════════════════════
MD5 = """## Step 5: IRF and peak realignment
One cell prepares the data alignment and the IRF for every later step (Step 4, *IRF preprocessing*, merged in on
2026-11-24). It starts from what Step 1 loaded (`data_raw`, `IRF`), so re-running it gives the same result until Step 1
loads other files.

- **Data**: circular shift so the fluorescence peak of the middle plane lands at 8 % of `nbin` (`target`) →
  `shift_data`, applied by every step that reads planes from disk (11, 13, 14, 17).
- **IRF** → `IRF2` (`flim_helpers.prepare_irf`):
  1. `clean_irf`: peak → `target`, baseline 0, and 0 further than `irf_tail_ns` (2.5 ns, Step 1) from the peak. The
     urea-SHG IRFs have echoes ~3.4 and ~4.9 ns after the peak that the slice decays do not have. Until 2026-11-24 the
     tav maps (Step 8), the phasor reference (Steps 11, 15) and Step 13 kept them, while Step 14 cut them when it found
     no bump and the width rescale cut them when applied: the IRF mean arrival time then differed by 0.03–0.045 ns
     between those paths (Slice3: `tau_phi` ~0.02 ns, `tau_mod` ~0.05 ns). Every step now uses the same IRF.
  2. Width (`irf_width_mode`, below).
- `IRF2` is used unchanged by every later step: tav reference `H` (Steps 8, 13, 14, 17), phasor reference (Steps 11,
  14, 15, 17) and all fits (Steps 14, 17). Its fingerprint `irf_key` is stored with the Step 14 caches, which are
  recomputed when the IRF changes.
- Output: printout, QC plots (IRF width: rising edge and residuals; alignment: IRF as loaded, `IRF2` and data) and
  `*_irf.json` (IRF file, `irf_key`, settings, width factor and chi2).

**IRF width (since 2026-11-23; `irf_width_mode` in Step 1).** An IRF recorded in another session can be wider or narrower
than the instrument response of the recording. On the 2026-11-10 brain slices, with the urea-SHG IRFs of 2 and 6 Nov:
- the data rose faster than any decay convolved with the measured IRF (impossible for a convolution), and `Fit_decay`
  compensated with a ~0.025 ns component holding ~9 % of the photons;
- a fitted width factor of 0.54–0.60 (2 Nov IRF) / 0.58–0.66 (6 Nov IRF) on Slice3 baseline, Slice3 sucrose and Slice1,
  the same with a 3 ns or the full window, brought the pooled fit's chi2 from 286 to 34 and the fast component from
  9 % to 0 %, and gave the same mean lifetime (1.91 ns) with either IRF;
- not the IRF's laser period (all November data and IRFs at 80 MHz; a ~53 MHz IRF would need broadening, not
  narrowing) and hardly count rate (2.7 → 3.4 bins from the dimmest to the brightest IRF pixels).

`fit_irf_width` (flim_helpers) fits, on the middle plane's decay up to `irf_width_win_ns` (3 ns) after the peak, the IRF
(echoes cut) stretched by a factor *s* about its peak and shifted by *c* bins, convolved with 2 free exponentials +
background (nuisance parameters); *s* = 1 gives the chi2 of the measured width. When chi2 drops by more than 20 %, the
rescaled IRF (*s* applied, *c* not: IRF and data stay peak-aligned and `Fit_decay` fits its own shift) becomes `IRF2`.
`'auto'` fits when the IRF has no `.fbs.xml` or its date or laser period differs from the data's, `'always'` fits
anyway, `'off'` keeps the measured width. The fit takes < 1 s. An older z-stack (XYZ_288, its own IRF): chi2 94 → 90,
not applied. An IRF recorded in the same session, with its `.fbs.xml`, is still the better reference.
"""

CODE5 = """## Step 5: IRF and peak realignment: the one cell that prepares the data alignment and the IRF of every later step
# Starts from what Step 1 loaded (data_raw, IRF): re-running it gives the same result until Step 1 loads other files.
#   data: circular shift so the fluorescence peak of the middle plane lands at 8 % of nbin (target) -> shift_data,
#         applied by every step that reads planes from disk (11, 13, 14, 17)
#   IRF:  flim_helpers.prepare_irf (see the Step 5 notes)
#         1. peak -> target, baseline 0, 0 beyond irf_tail_ns from the peak (echoes of the IRF recording)  [clean_irf]
#         2. width rescaled to this recording when the IRF is not from the same session (irf_width_mode)  [fit_irf_width]
#         -> IRF2, used unchanged by every later step: tav reference H (Steps 8, 13, 14, 17), phasor reference
#            (Steps 11, 14, 15, 17), all fits (Steps 14, 17). irf_key, its fingerprint, is stored with the Step 14
#            caches, which are recomputed when the IRF changes.
import json as _json
import flim_helpers as _fh_mod; importlib.reload(_fh_mod)
from flim_helpers import prepare_irf, irf_session, plot_irf_width

# ── data and IRF as loaded (Step 1); bin count check ─────────────────────────
DataTCSPC = _to_yxt(data_raw, dim_data)              # middle plane
_irf_in   = np.asarray(IRF, dtype=float).ravel()     # IRF summed to 1-D
Ix        = _irf_in.size
nbin      = DataTCSPC.shape[2]
if nbin != Ix:
    raise ValueError(f'Bin count mismatch: data {nbin} bins, IRF {Ix} bins. Check dim_data / dim_irf (Step 1); the '
                     'retired bin-alignment cell at the end of the notebook handles a true mismatch.')
dt     = Tcycle / nbin
tau    = np.arange(1, nbin + 1) * dt
target = int(round(0.08 * nbin))
print(f'nbin = {nbin},  dt = {dt:.4f} ns,  target peak bin {target} / {nbin} ({100 * target / nbin:.1f}%)')

# ── data: fluorescence peak (middle plane, pixel mask) → target ──────────────
maskPixel    = mask3d[mid_idx]
decay_masked = DataTCSPC[maskPixel].sum(axis=0).astype(float)
peak_data    = int(np.argmax(decay_masked))
shift_data   = target - peak_data
DataTCSPC    = np.roll(DataTCSPC, shift_data, axis=2)
print(f'Data peak was bin {peak_data:3d},  shifted by {shift_data:+d}')

# ── IRF: peak → target, echoes cut, width (irf_tail_ns, irf_width_mode, irf_width_win_ns: Step 1) ──
_same_session, _why = irf_session(meta_irf, meta)
_y_w = DataTCSPC.sum(axis=(0, 1)).astype(float)      # whole middle plane: the decay the IRF width is fitted to
_irf = prepare_irf(_irf_in, _y_w, dt, target, tail_ns=irf_tail_ns, width_mode=irf_width_mode,
                   same_session=_same_session, win_ns=irf_width_win_ns)
IRF2, irf_key, irf_width = _irf['irf'], _irf['key'], _irf['width']
print(f'IRF  peak was bin {int(np.argmax(_irf_in)):3d},  moved to {target}'
      + (f';  baseline 0 beyond ±{irf_tail_ns} ns of the peak (echoes cut)' if irf_tail_ns is not None
         else ';  tail as measured (irf_tail_ns = None)'))
if irf_width is None:
    print(f'IRF width fit skipped ({_why}; irf_width_mode = {irf_width_mode!r}).')
else:
    print(f'IRF width fit ({_why}): factor {irf_width["s"]:.3f} (shift {irf_width["c"]:+.2f} bins, not applied);  '
          f'chi2 {irf_width["chi2_measured"]:.1f} with the measured width → {irf_width["chi2_fitted"]:.1f}'
          + ('  → IRF2 rescaled' if irf_width['apply'] else '  → < 20 % better: width kept as measured'))
    plot_irf_width(irf_width, tau, _y_w, target, irf_width_win_ns, title=f'[{FName}]')
with open(os.path.join(newFolder, f'{FName}_irf.json'), 'w') as _f:    # record of the IRF used by Steps 6-17
    _json.dump(dict(irf=os.path.basename(irf_path), irf_key=irf_key, tail_ns=irf_tail_ns, reason=_why,
                    width_mode=irf_width_mode, win_ns=irf_width_win_ns,
                    **({} if irf_width is None else
                       dict(factor=irf_width['s'], shift_bins=irf_width['c'], chi2_measured=irf_width['chi2_measured'],
                            chi2_fitted=irf_width['chi2_fitted'], applied=irf_width['apply']))), _f, indent=1)
print(f'IRF2 (Steps 6-17): key {irf_key},  mean arrival time {np.dot(tau, IRF2) / IRF2.sum():.4f} ns  '
      f'→ {FName}_irf.json')

# ── sanity-check plot ─────────────────────────────────────────────────────────
_decay_aligned = DataTCSPC[maskPixel].sum(axis=0).astype(float)
_irf_loaded    = np.roll(_irf_in, target - int(np.argmax(_irf_in)))

fig_align, ax_align = plt.subplots(figsize=(7, 4))
ax_align.semilogy(tau, _irf_loaded / _irf_loaded.max(), color='0.7', linewidth=0.8, label='IRF as loaded')
ax_align.semilogy(tau, IRF2 / IRF2.max(), '--k', linewidth=0.8, label='IRF2 (used by Steps 6-17)')
ax_align.semilogy(tau, _decay_aligned / _decay_aligned.max(), color='steelblue', linewidth=1.2,
                  label=f'Data  ({maskPixel.sum()} px)')
ax_align.axvline(tau[target], color='orange', linewidth=1, linestyle=':', label=f'target  bin {target}')
ax_align.set_xlabel('time / ns')
ax_align.set_ylabel('intensity (norm.)')
ax_align.set_title(f'After realignment  [{FName}]' + (f'\\n{meta_str}' if meta_str else ''), fontsize=10)
ax_align.set_xlim(0, Tcycle)
ax_align.set_ylim(5e-3, 1.3)
ax_align.legend()
fig_align.tight_layout()
plt.show()
"""

# ════════════════════════════════════════════════════════════════════════════
# Step 14
# ════════════════════════════════════════════════════════════════════════════
S14 = [
    ("from flim_helpers import Fit_decay, fit_decays, pooled_lifetimes\n",
     "from flim_helpers import Fit_decay, fit_decays, pooled_lifetimes, report_pooled_fit, phasor_lifetimes\n"),
    ("_peak_irf        = int(np.argmax(IRF2))\n", ""),
    ("IRF2_fit   = np.roll(IRF2, target - _peak_irf)  # IRF also goes to target\n"
     "print(f'Bulk peak bin {_peak_bulk}  →  shifted {_shift_to_target:+d} bins  →  target bin {target}  |  IRF shifted to target')\n",
     "print(f'Bulk peak bin {_peak_bulk}  →  shifted {_shift_to_target:+d} bins  →  target bin {target}  '\n"
     "      f'(IRF2: at the target since Step 5)')\n"),
    ("# itself has two, ~3.4 and ~4.9 ns after its peak).\n"
     "#   bump found -> the fit stops bump_safety_bins before it (t_stop_ns).\n"
     "#   no bump    -> the whole decay is fitted (t_stop_ns = None), and the IRF after irf_tail_ns is set to its\n"
     "#                 baseline so the echoes of the IRF recording don't put a bump into the model.\n",
     "# itself has two, ~3.4 and ~4.9 ns after its peak; Step 5 cuts them from IRF2, irf_tail_ns).\n"
     "#   bump found -> the fit stops bump_safety_bins before it (t_stop_ns).\n"
     "#   no bump    -> the whole decay is fitted (t_stop_ns = None).\n"),
    ("irf_tail_ns        = 2.5   # ← no bump: IRF kept up to this long after its peak, baseline after (None = as measured)\n", ""),
    ("if t_stop_ns is None and irf_tail_ns is not None and target + int(round(irf_tail_ns / dt)) < nbin:\n"
     "    _irf_cut = target + int(round(irf_tail_ns / dt))\n"
     "    IRF2_fit = IRF2_fit.copy()\n"
     "    IRF2_fit[_irf_cut:] = np.median(IRF2_fit[_irf_cut:])\n"
     "    print(f'IRF after {tau[_irf_cut]:.2f} ns set to its baseline (removes the echoes of the IRF recording).')\n", ""),
    ("if _have_fits:   # cached results made with another t_stop or fit_per_cell, or without the pooled fit, are re-made\n"
     "    _fz        = np.load(_fits_path)\n"
     "    _ts_cached = float(_fz['t_stop_ns'][0])\n"
     "    _pc_cached = bool(_fz['per_cell_fit'][0]) if 'per_cell_fit' in _fz.files else True\n"
     "    if not (",
     "if _have_fits:   # cached results made with another IRF, t_stop or fit_per_cell, or without the pooled fit, are re-made\n"
     "    _fz         = np.load(_fits_path)\n"
     "    _ts_cached  = float(_fz['t_stop_ns'][0])\n"
     "    _pc_cached  = bool(_fz['per_cell_fit'][0]) if 'per_cell_fit' in _fz.files else True\n"
     "    _irf_cached = str(_fz['irf_key'][0]) if 'irf_key' in _fz.files else None\n"
     "    if _irf_cached != irf_key:\n"
     "        _have_fits = False\n"
     "        print(f'Cached results were made with another IRF (key {_irf_cached}, now {irf_key}: Step 5) → recomputing.')\n"
     "    elif not ("),
    ("_irf_n  = IRF2_fit  / IRF2_fit.max()\n", "_irf_n  = IRF2 / IRF2.max()\n"),
    ("    _H_irf      = float(np.dot(tau, IRF2_fit) / IRF2_fit.sum())   # IRF mean arrival time: tav_pooled reference (as Step 17)\n",
     "    _H_irf      = float(np.dot(tau, IRF2) / IRF2.sum())   # IRF mean arrival time = H of Step 8: tav_pooled reference\n"),
    ("fit_decays(IRF2_fit, _cell_mat[_fit_ids]", "fit_decays(IRF2, _cell_mat[_fit_ids]"),
    ("""    # ── phasor per cell (the Step 15 definition): G, S, phase lifetime tau_phi, modulation lifetime tau_mod_c ──
    _ok_ph = df_cells['n_photons'].values >= min_photons
    _tp_ph = (np.arange(1, nbin + 1) - 0.5) * dt
    _T_ph  = 2.0 * np.pi * (_tp_ph - np.dot(_tp_ph, IRF2) / IRF2.sum()) / Tcycle   # phase 0 = IRF centroid (Step 13)
    _w_ph  = _cell_mat[1:] / np.maximum(_cell_mat[1:].sum(axis=1, keepdims=True), 1.0)
    _gc, _sc, _tau_mod_c, _tau_phi_c = phasor_ref_correction(
        np.where(_ok_ph, _w_ph @ np.cos(_T_ph), np.nan), np.where(_ok_ph, _w_ph @ np.sin(_T_ph), np.nan),
        np.array([[np.dot(np.cos(_T_ph), IRF2) / IRF2.sum()]]), np.array([[np.dot(np.sin(_T_ph), IRF2) / IRF2.sum()]]),
        tau_ref_ns=0.0, freq_mhz=1000.0 / Tcycle)
""", """    # ── phasor per cell: G, S, phase lifetime tau_phi, modulation lifetime tau_mod (IRF reference as Step 11) ──
    _gc, _sc, _tau_phi_c, _tau_mod_c = phasor_lifetimes(_cell_mat[1:], IRF2, dt, Tcycle, min_photons=min_photons)
"""),
    ("pooled_fit = pooled_lifetimes(IRF2_fit, _pool_dec", "pooled_fit = pooled_lifetimes(IRF2, _pool_dec"),
    ("                        irf_aligned  = IRF2_fit,\n",
     "                        irf_aligned  = IRF2,\n"
     "                        irf_key      = np.array([irf_key]),       # Step 5 IRF: the cache is re-made when it changes\n"),
    ("(keys: tau_axis, irf_aligned, t_stop_ns,", "(keys: tau_axis, irf_aligned, irf_key, t_stop_ns,"),
]
S14_TAIL_OLD = "# ── pooled fit: summary + QC plot (fit and residuals) ─────────────────────────\n"
S14_TAIL_NEW = """# ── pooled fit: summary + QC plot (fit and residuals) ─────────────────────────
report_pooled_fit(pooled_fit, tau, _pool_dec, IRF2, Tcycle, t_stop_ns=t_stop_ns,
                  label=f'Pooled fit  [{FName}{cell_tag}]  all {n_cells} cells, {_pool_dec.sum():,.0f} photons, {_t_stop_str}',
                  png_path=_pool_csv.replace('.csv', '.png'))
"""

MD14 = [("an IRF-shaped component (< 0.1 ns), left out of both means",
         "an IRF-shaped component (< 0.05 ns), left out of both means"),
        ("IRF is not from the same session (`irf_width_mode`); an IRF recorded on the imaging day is still best.\n",
         "IRF is not from the same session (`irf_width_mode`); an IRF recorded on the imaging day is still best.\n"
         "  The component shares (`frac1..`) are of all photons in decay components: with `fast_frac` they add up to 1.\n\n"
         "The IRF (`IRF2`) comes unchanged from Step 5 (no re-shift, no tail handling here since 2026-11-24). The caches\n"
         "store its fingerprint `irf_key` and are recomputed when the IRF changes (another file, `irf_tail_ns`,\n"
         "`irf_width_mode`).\n")]

# ════════════════════════════════════════════════════════════════════════════
# Step 15
# ════════════════════════════════════════════════════════════════════════════
S15_END = "          f'(median tau_phi {np.nanmedian(_tphi_c):.3f} ns, tau_mod {np.nanmedian(_tmod_c):.3f} ns)')\n"
S15_HEAD = """if 'G_phasor' not in df_cells:
    print('df_cells has no phasor columns: run Step 14 first.')
else:
    # per-cell G / S / tau_phi / tau_mod: computed in Step 14 (flim_helpers.phasor_lifetimes, IRF reference as Step 11)
    _G_corr_c = df_cells['G_phasor'].values.astype(float)
    _S_corr_c = df_cells['S_phasor'].values.astype(float)
    print(f'Per-cell phasor of Step 14 ({os.path.basename(csv_out)}): median tau_phi {df_cells["tau_phi"].median():.3f} ns, '
          f'tau_mod {df_cells["tau_mod"].median():.3f} ns')
"""
MD15 = [("Computes a single corrected (G, S) phasor coordinate per Cellpose cell from its pooled 3D decay.\n"
         "Requires `G_ref` / `S_ref` from Step 11 and `_cell_decays` from Step 14.\n",
         "Plots the corrected (G, S) phasor coordinate per Cellpose cell, computed from its pooled 3D decay in Step 14\n"
         "(`flim_helpers.phasor_lifetimes`, the IRF reference of Step 11; columns of the per-cell CSV).\n"),
        ("`G_phasor` / `S_phasor` are appended to the per-cell CSV, with the lifetimes read from the phasor:\n",
         "`G_phasor` / `S_phasor` are in the per-cell CSV, with the lifetimes read from the phasor:\n")]

# ════════════════════════════════════════════════════════════════════════════
# Step 17 (Time)
# ════════════════════════════════════════════════════════════════════════════
S17 = [
    ("    _H     = float(np.dot(tau, IRF2_fit) / IRF2_fit.sum())            # IRF mean arrival time\n"
     "    _tau_p = (np.arange(1, nbin + 1) - 0.5) * dt\n"
     "    _T_c   = 2.0 * np.pi * (_tau_p - np.dot(_tau_p, IRF2) / IRF2.sum()) / Tcycle\n",
     "    _H     = float(np.dot(tau, IRF2) / IRF2.sum())                    # IRF mean arrival time (H of Step 8)\n"),
    ("fit_decays(IRF2_fit, _dec_t[:, 1:][_ok]", "fit_decays(IRF2, _dec_t[:, 1:][_ok]"),
    ("aw_LT=np.nan, chi2=np.nan, tav_pooled=np.nan, G_raw=np.nan, S_raw=np.nan)",
     "aw_LT=np.nan, chi2=np.nan, tav_pooled=np.nan)"),
    ("                rec.update(tav_pooled=round(float(np.dot(tau, _dec) / _nph - _H), 4),\n"
     "                           G_raw=float(np.dot(np.cos(_T_c), _dec) / _nph), S_raw=float(np.dot(np.sin(_T_c), _dec) / _nph))\n",
     "                rec.update(tav_pooled=round(float(np.dot(tau, _dec) / _nph - _H), 4))\n"),
    ("""    if G_ref is not None and S_ref is not None:
        _gc, _sc, _tm, _tp = phasor_ref_correction(df_tc['G_raw'].values, df_tc['S_raw'].values, G_ref, S_ref,
                                                   tau_ref_ns=0.0, freq_mhz=freq_mhz)
        df_tc['G_phasor'], df_tc['S_phasor'] = np.round(_gc, 5), np.round(_sc, 5)
        df_tc['tau_phi'],  df_tc['tau_mod']  = np.round(_tp, 4), np.round(_tm, 4)
    else:
        print('G_ref / S_ref not available (Step 11): no phasor columns')
    df_tc = df_tc.drop(columns=['G_raw', 'S_raw'])
""", """    # phasor per cell and timepoint (rows: timepoint-major, as _recs), IRF reference as Steps 11 / 14
    _gc, _sc, _tp, _tm = phasor_lifetimes(_dec_t[:, 1:].reshape(-1, nbin), IRF2, dt, Tcycle, min_photons=min_photons)
    df_tc['G_phasor'], df_tc['S_phasor'] = np.round(_gc, 5), np.round(_sc, 5)
    df_tc['tau_phi'],  df_tc['tau_mod']  = np.round(_tp, 4), np.round(_tm, 4)
"""),
    ("fit_decays(IRF2_fit, _pool_t[_pool_ok]", "fit_decays(IRF2, _pool_t[_pool_ok]"),
    ("pooled_lifetimes(IRF2_fit, _pool_t[_t]", "pooled_lifetimes(IRF2, _pool_t[_t]"),
]
MD17 = [("- `G_phasor`, `S_phasor`, `tau_phi`, `tau_mod`: corrected phasor of the pooled decay and its lifetimes (as Step 15);\n",
         "- `G_phasor`, `S_phasor`, `tau_phi`, `tau_mod`: corrected phasor of the pooled decay and its lifetimes (as Step 14,\n"
         "  `flim_helpers.phasor_lifetimes`);\n")]

# ════════════════════════════════════════════════════════════════════════════
# Step 0 (Time / Tile): acquisition folder, raw deletion; Step 2b session date
# ════════════════════════════════════════════════════════════════════════════
PICK_OLD = """def _pick_raw():
    root = tk.Tk(); root.withdraw(); root.attributes('-topmost', True)
    p = filedialog.askopenfilename(title='Select ISS Vista XYTZ TIFF',
                                   filetypes=[('TIFF files', '*.tif *.tiff'), ('All files', '*.*')])
    root.destroy()
    return p or None

# ── raw file — swap comments to switch between hardcoded and interactive ──────
"""
PICK_NEW = """def _pick_acq_dir():
    root = tk.Tk(); root.withdraw(); root.attributes('-topmost', True)
    p = filedialog.askdirectory(title='Select the acquisition folder (raw Vista TIFF, .fbs.xml, Olympus/)')
    root.destroy()
    return p or None

# ── acquisition folder (or the raw .tif itself) — swap comments for hardcoded / interactive ──
"""
PICK_TIME_OLD = """# raw_path = r"D:\\Yu_Brain_slices\\Slice3_DG_CA3_baseline1_Cycle\\Slice3_DG_CA3_baseline1_Decay128_512x512_Z72_Ch1_T1_P1_S1.tif"
raw_path = _pick_raw()
if not raw_path:
    raise RuntimeError('No raw file selected.')
"""
PICK_TIME_NEW = """# acq_path = r"D:\\Yu_Brain_slices\\Slice3_DG_CA3_baseline1_Cycle"
acq_path = _pick_acq_dir()
if not acq_path:
    raise RuntimeError('No acquisition folder selected.')
"""
PICK_TILE_OLD = """raw_path = r"D:\\Peredox_FLIM\\XYZTTile1_336_X0.4Y-2.8_Decay128_512x512_Z210_Ch1_T1_P1_S1\\XYZTTile1_336_X0.4Y-2.8_Decay128_512x512_Z210_Ch1_T1_P1_S1.tif"
# raw_path = _pick_raw()
if not raw_path:
    raise RuntimeError('No raw file selected.')
"""
PICK_TILE_NEW = """acq_path = r"D:\\Peredox_FLIM\\XYZTTile1_336_X0.4Y-2.8_Decay128_512x512_Z210_Ch1_T1_P1_S1"
# acq_path = _pick_acq_dir()
if not acq_path:
    raise RuntimeError('No acquisition folder selected.')
"""
PICK_RESOLVE = """# the raw float32 Vista TIFF in that folder; once it was deleted after its uint8 conversion (delete_raw_after_convert),
# its name comes from the conversion report in tiles_uint8/ and Step 0b uses the uint8 tiles
raw_path, raw_exists = vt.find_raw(acq_path)
"""
RERUN_OLD = "rerun_fuse     = False\n"
RERUN_NEW = """rerun_fuse     = False
delete_raw_after_convert = True   # Step 0b: delete the float32 raw TIFF (~4x the size of its uint8 tiles) once the tiles
                                  # are verified to hold all of its photons (no count clipped at 255; every tile read back)
"""
SIZE_OLD = "print(f'Raw file: {raw_path}  ({os.path.getsize(raw_path) / 1e9:.1f} GB)')\n"
SIZE_NEW = ("print(f'Raw file: {raw_path}  ' + (f'({os.path.getsize(raw_path) / 1e9:.1f} GB)' if raw_exists else\n"
            "      '(deleted after its uint8 conversion: Step 0b uses the uint8 tiles)'))\n")
DEL_0B = """
# ── raw float32 file: deleted once the uint8 tiles hold all of its photons (delete_raw_after_convert, Step 0a) ──
if raw_exists and delete_raw_after_convert:
    _deleted, _msg = vt.delete_raw_if_lossless(raw_path, conv)
    raw_exists = not _deleted
    print(('Raw file deleted: ' if _deleted else 'Raw file kept: ') + _msg)
elif raw_exists:
    print(f'Raw file kept (delete_raw_after_convert = False): {os.path.getsize(raw_path) / 1e9:.1f} GB')
"""
DATE_OLD = "date=md.session_date(raw_path), raw_path=raw_path,"
DATE_NEW = "date=((fbs_meta or {}).get('datetime') or '')[:10] or md.session_date(raw_path), raw_path=raw_path,"
MD0_OLD = "- **0a** parameters & metadata."
MD0_NEW = """- **Acquisition folder** (`acq_path`, folder picker): the raw float32 Vista TIFF in it is the `.tif` named
  `<name>_Decay<n>_<nx>x<ny>_Z<n>_Ch1_T1_P1_S1.tif` (`vt.find_raw`). With `delete_raw_after_convert = True` (default),
  0b deletes it once the uint8 tiles hold all of its photons (`vt.delete_raw_if_lossless`): no count clipped at 255,
  at most a few bins that are not photon counts (2 in every file so far), and every tile file, read back, gives the
  photon counts per z-block and bin recorded during the conversion. The uint8 tiles are ~1/4 of the raw size
  (Slice3: 9.7 GB → 2.4 GB). Without the raw file its name comes from the conversion report, and Steps 0–17 run from the
  uint8 tiles; only `rerun_convert = True` then fails. The `.fbs.xml` and the Olympus files stay.
- **0a** parameters & metadata."""


def patch(nb_path):
    nb = json.load(open(nb_path, encoding='utf-8'))
    name = os.path.basename(nb_path)
    ids = [c.get('id') for c in nb['cells']]
    if 'prepare_irf' in src(nb, '5622bfa1'):
        print(f'{name}: already patched')
        return

    # Step 1 parameters
    put(nb, 'code-03', sub(src(nb, 'code-03'), P1_OLD, P1_NEW, f'{name} code-03'))

    # Step 4 cells out, Step 5 markdown + code
    if 'code-05' in ids:
        nb['cells'] = [c for c in nb['cells'] if c.get('id') not in ('md-05', 'code-05')]
    put(nb, '1a7938ca', MD5)
    put(nb, '5622bfa1', CODE5)

    # Step 14
    s = src(nb, 'b3fb5433')
    for old, new in S14:
        s = sub(s, old, new, f'{name} Step 14')
    i = s.find(S14_TAIL_OLD)
    if i < 0:
        sys.exit(f'{name} Step 14: pooled-fit summary block not found')
    s = s[:i] + S14_TAIL_NEW
    if 'IRF2_fit' in s:
        sys.exit(f'{name} Step 14: IRF2_fit left')
    put(nb, 'b3fb5433', s)
    m = src(nb, 'ad5b1d96')
    for old, new in MD14:
        m = sub(m, old, new, f'{name} Step 14 notes')
    put(nb, 'ad5b1d96', m)

    # Step 15
    s = src(nb, '6e4891ec')
    i = s.find(S15_END)
    if i < 0 or not s.startswith('if G_ref is None or S_ref is None:'):
        sys.exit(f'{name} Step 15: head not found')
    s = S15_HEAD + s[i + len(S15_END):]
    s = sub(s, "    _omega  = 2.0 * np.pi * freq_mhz * 1e6\n", "    _omega  = 2.0 * np.pi * 1e9 / Tcycle\n", f'{name} Step 15')
    put(nb, '6e4891ec', s)
    m = src(nb, 'dc21f008')
    for old, new in MD15:
        m = sub(m, old, new, f'{name} Step 15 notes')
    put(nb, 'dc21f008', m)

    # Step 17
    if 'code-17' in ids:
        s = src(nb, 'code-17')
        for old, new in S17:
            s = sub(s, old, new, f'{name} Step 17')
        if 'IRF2_fit' in s or 'G_raw' in s:
            sys.exit(f'{name} Step 17: IRF2_fit / G_raw left')
        put(nb, 'code-17', s)
        m = src(nb, 'md-17')
        for old, new in MD17:
            m = sub(m, old, new, f'{name} Step 17 notes')
        put(nb, 'md-17', m)

    # Step 0 (raw-file notebooks)
    if 'code-tile0a' in ids:
        s = src(nb, 'code-tile0a')
        s = sub(s, PICK_OLD, PICK_NEW, f'{name} 0a picker')
        if PICK_TIME_OLD in s:
            s = sub(s, PICK_TIME_OLD, PICK_TIME_NEW + PICK_RESOLVE, f'{name} 0a path')
        else:
            s = sub(s, PICK_TILE_OLD, PICK_TILE_NEW + PICK_RESOLVE, f'{name} 0a path')
        s = sub(s, RERUN_OLD, RERUN_NEW, f'{name} 0a rerun')
        s = sub(s, SIZE_OLD, SIZE_NEW, f'{name} 0a size')
        put(nb, 'code-tile0a', s)
        s = src(nb, 'code-tile0b')
        put(nb, 'code-tile0b', s.rstrip('\n') + '\n' + DEL_0B)
        put(nb, 'md-tile0', sub(src(nb, 'md-tile0'), MD0_OLD, MD0_NEW, f'{name} Step 0 notes'))
        put(nb, 'code-2b', sub(src(nb, 'code-2b'), DATE_OLD, DATE_NEW, f'{name} Step 2b'))

    with open(nb_path, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(nb, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(f'{name}: patched')


if __name__ == '__main__':
    for p in (sys.argv[1:] or NBS):
        patch(p)
