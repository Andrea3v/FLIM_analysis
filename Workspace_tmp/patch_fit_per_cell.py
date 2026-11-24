"""fit_per_cell switch (2026-11-23), AnalyzeTCSPC_XYTZTime.ipynb only.

fit_per_cell = False (default, Step 1): no Fit_decay per cell (Step 14) or per cell x timepoint (Step 17); per cell only
model-free values (tav_pooled, phasor G/S, tau_phi, tau_mod). Fit_decay runs once on the pooled decay of all cells
(Step 14, + QC plot) and once per timepoint (Step 17), summarised by flim_helpers.pooled_lifetimes.
fit_per_cell = True: the per-cell fits run as before (nothing removed).

Idempotent (skips a notebook that already has `fit_per_cell`); every replacement must match exactly once.
Usage: python Workspace_tmp/patch_fit_per_cell.py [notebook]
"""
import json, os, sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB = os.path.join(HERE, 'AnalyzeTCSPC_XYTZTime.ipynb')

R = {}   # cell id -> [(old, new), ...]

R['code-03'] = [(
"""# before 2026-11-20: 3D NLM h 0.8 sigma, no smoothing, 'ball' radius 30 px, thresh_norm 0.55
""",
"""# before 2026-11-20: 3D NLM h 0.8 sigma, no smoothing, 'ball' radius 30 px, thresh_norm 0.55

# ── lifetime per cell (Steps 14, 15, 17) ──────────────────────────────────────
fit_per_cell   = False         # True: multi-exponential Fit_decay per cell (Step 14) and per cell x timepoint (Step 17).
                               # False: per cell only model-free values (tav_pooled, phasor G/S, tau_phi, tau_mod);
                               # Fit_decay runs once on the pooled decay of all cells (Step 14) and of each timepoint (17).
                               # Slice3, 700 photons/cell: SD of a per-cell fit 0.34-0.68 ns, tav / tau_phi 0.08 ns.
""")]

R['ad5b1d96'] = [(
"""## Step 14: Per-cell 3D decay fitting
Fits a multi-exponential decay to each Cellpose cell's pooled 3D TCSPC.
""",
"""## Step 14: Per-cell 3D decay fitting
Fits a multi-exponential decay to each Cellpose cell's pooled 3D TCSPC.

**`fit_per_cell` (Step 1, default `False` since 2026-11-23).** At a few hundred to a few thousand photons the per-cell
fits are unstable (Slice3, 700 photons: SD 0.34–0.68 ns between two halves of the same cell, larger than the spread
between cells). With `False`:
- per cell only `tav_pooled` (mean photon arrival time of the cell's decay minus the IRF's, as Step 17); the phasor,
  `tau_phi` and `tau_mod` come in Step 15 (SD at 700 photons: 0.08 ns);
- `Fit_decay` runs **once on the pooled decay of all cells** → `*_pooled_fit.csv` (`tau_int` intensity-weighted,
  `tau_amp` amplitude-weighted mean lifetime, components, `fast_frac`, `chi2`) + `*_pooled_fit.png` (fit and
  residuals). `fast_frac` is the photon share of an IRF-shaped component (< 0.1 ns), left out of both means: on the
  2026-11 slices it is ~9% because the stored IRF is broader than the recording's (data rise faster than the IRF
  allows; a fitted IRF width of 0.61–0.67× removes it, chi2 286 → 34). Record an IRF on the imaging day.

`aw_LT` of the per-cell fits (`True`) is the intensity-weighted mean lifetime: `Fit_decay` returns photons per
component, not amplitudes (see `flim_helpers.pooled_lifetimes`).
""")]

R['b3fb5433'] = [
("from flim_helpers import Fit_decay, fit_decays\n",
 "from flim_helpers import Fit_decay, fit_decays, pooled_lifetimes\n"),
("""_fits_path  = os.path.join(newFolder, f'{FName}{cell_tag}_cell_fits.npz')   # decay + model per cell
""",
"""_fits_path  = os.path.join(newFolder, f'{FName}{cell_tag}_cell_fits.npz')   # decay + model per cell
_pool_csv   = os.path.join(newFolder, f'{FName}{cell_tag}_pooled_fit.csv')  # Fit_decay of the pooled decay of all cells
"""),
("""if _have_fits:   # cached fits made with another t_stop are re-fitted
    _ts_cached = float(np.load(_fits_path)['t_stop_ns'][0])
    if not (np.isnan(_ts_cached) if t_stop_ns is None else np.isclose(_ts_cached, t_stop_ns)):
        _have_fits = False
        print(f'Cached fits used t_stop = {_ts_cached:.3f} ns → re-fitting.')
""",
"""if _have_fits:   # cached results made with another t_stop or fit_per_cell, or without the pooled fit, are re-made
    _fz        = np.load(_fits_path)
    _ts_cached = float(_fz['t_stop_ns'][0])
    _pc_cached = bool(_fz['per_cell_fit'][0]) if 'per_cell_fit' in _fz.files else True
    if not (np.isnan(_ts_cached) if t_stop_ns is None else np.isclose(_ts_cached, t_stop_ns)):
        _have_fits = False
        print(f'Cached fits used t_stop = {_ts_cached:.3f} ns → re-fitting.')
    elif _pc_cached != fit_per_cell or 'pooled_model' not in _fz.files or not os.path.exists(_pool_csv):
        _have_fits = False
        print(f'Cached results: fit_per_cell = {_pc_cached}, pooled fit {"present" if "pooled_model" in _fz.files else "missing"}'
              f' → recomputing with fit_per_cell = {fit_per_cell}.')
"""),
("""_vol_cx = (cp_masks.shape[2] - 1) / 2.0

if _have_acc and _have_csv and _have_fits:
    print(f'Reloading fitting results: {os.path.basename(csv_out)}')
    df_cells = pd.read_csv(csv_out)
    _fits     = np.load(_fits_path)
    _cell_models = _fits['models']   # (n_cells, nbin)
""",
"""_vol_cx = (cp_masks.shape[2] - 1) / 2.0
_t_stop_str = f't_stop={t_stop_ns:.3f} ns' if t_stop_ns else 't_stop=None'
_show_cols  = ['cell_id', 'n_voxels', 'n_photons'] + (['aw_LT', 'chi2'] if fit_per_cell else ['tav_pooled'])
_pool_dec   = _cell_mat[1:].sum(axis=0)        # pooled decay of all cells

if _have_acc and _have_csv and _have_fits:
    print(f'Reloading fitting results: {os.path.basename(csv_out)}')
    df_cells = pd.read_csv(csv_out)
    _fits     = np.load(_fits_path)
    _cell_models = _fits['models']   # (n_cells, nbin)
    pooled_fit = dict(pd.read_csv(_pool_csv).iloc[0], model=_fits['pooled_model'])
"""),
("""    print(df_cells[['cell_id', 'n_voxels', 'n_photons', 'aw_LT', 'chi2']].to_string(index=False))
else:
    records      = []
    _cell_models = np.full((n_cells, nbin), np.nan)   # row i = cell i+1

    _t_stop_str = f't_stop={t_stop_ns:.3f} ns' if t_stop_ns else 't_stop=None'
    _fit_ids    = [cid for cid in range(1, n_cells + 1) if int(_cell_mat[cid].sum()) >= min_photons]
    print(f'Fitting {len(_fit_ids)} of {n_cells} cells  (min_photons={min_photons},  {_t_stop_str})...')
    _t_fit = _time.perf_counter()
    _fit_res = dict(zip(_fit_ids, fit_decays(IRF2_fit, _cell_mat[_fit_ids], Tcycle, dt, t_stop_ns=t_stop_ns,
                                             n_workers=fit_workers)))
""",
"""    print(df_cells[[c for c in _show_cols if c in df_cells]].to_string(index=False))
else:
    records      = []
    _cell_models = np.full((n_cells, nbin), np.nan)   # row i = cell i+1

    _fit_ids    = [cid for cid in range(1, n_cells + 1) if int(_cell_mat[cid].sum()) >= min_photons]
    _H_irf      = float(np.dot(tau, IRF2_fit) / IRF2_fit.sum())   # IRF mean arrival time: tav_pooled reference (as Step 17)
    _t_fit = _time.perf_counter()
    if fit_per_cell:
        print(f'Fitting {len(_fit_ids)} of {n_cells} cells  (min_photons={min_photons},  {_t_stop_str})...')
        _fit_res = dict(zip(_fit_ids, fit_decays(IRF2_fit, _cell_mat[_fit_ids], Tcycle, dt, t_stop_ns=t_stop_ns,
                                                 n_workers=fit_workers)))
    else:
        _fit_res = {}
        print(f'Per-cell fits off (fit_per_cell = False, Step 1): tav_pooled for {len(_fit_ids)} of {n_cells} cells '
              f'(>= {min_photons} photons); phasor and tau_phi in Step 15.')
"""),
("""                   centroid_x=round(cx, 2), dist_from_center_um=_dist_um)
        if cid not in _fit_res:
            rec.update(aw_LT=np.nan, chi2=np.nan)
            for i in range(1, 6):
                rec[f'tau{i}'] = np.nan;  rec[f'amp{i}'] = np.nan
            print(f'  cell {cid:3d}  SKIP  (n_ph={n_ph})')
""",
"""                   centroid_x=round(cx, 2), dist_from_center_um=_dist_um,
                   tav_pooled=round(float(np.dot(tau, decay) / n_ph - _H_irf), 4) if n_ph >= min_photons else np.nan)
        if cid not in _fit_res:
            rec.update(aw_LT=np.nan, chi2=np.nan)
            for i in range(1, 6):
                rec[f'tau{i}'] = np.nan;  rec[f'amp{i}'] = np.nan
            if fit_per_cell:
                print(f'  cell {cid:3d}  SKIP  (n_ph={n_ph})')
"""),
("""    print(f'Fitting done in {_time.perf_counter() - _t_fit:.1f} s.')
    df_cells = pd.DataFrame(records)
    df_cells.to_csv(csv_out, index=False)
    np.savez_compressed(_fits_path,
                        tau_axis    = tau,
                        irf_aligned = IRF2_fit,
                        t_stop_ns   = np.array([t_stop_ns if t_stop_ns else np.nan]),
                        cell_ids    = np.arange(1, n_cells + 1),
                        decays      = _cell_mat[1:],    # (n_cells, nbin) raw TCSPC
                        models      = _cell_models)     # (n_cells, nbin) fitted model
    print(f'Saved: {csv_out}')
    print(f'Saved: {os.path.basename(_fits_path)}  '
          f'(keys: tau_axis, irf_aligned, t_stop_ns, cell_ids, decays, models)')
    print(df_cells[['cell_id', 'n_voxels', 'n_photons', 'aw_LT', 'chi2']].to_string(index=False))""",
"""    if fit_per_cell:
        print(f'Fitting done in {_time.perf_counter() - _t_fit:.1f} s.')
    df_cells = pd.DataFrame(records)
    df_cells.to_csv(csv_out, index=False)

    # ── pooled fit: one Fit_decay on the summed decay of all cells ────────────
    pooled_fit = pooled_lifetimes(IRF2_fit, _pool_dec, Tcycle, dt, t_stop_ns=t_stop_ns)
    pd.DataFrame([dict(n_cells=n_cells, n_photons=int(_pool_dec.sum()), t_stop_ns=t_stop_ns if t_stop_ns else np.nan,
                       **{k: v for k, v in pooled_fit.items() if k != 'model'})]).to_csv(_pool_csv, index=False)

    np.savez_compressed(_fits_path,
                        tau_axis     = tau,
                        irf_aligned  = IRF2_fit,
                        t_stop_ns    = np.array([t_stop_ns if t_stop_ns else np.nan]),
                        per_cell_fit = np.array([fit_per_cell]),
                        cell_ids     = np.arange(1, n_cells + 1),
                        decays       = _cell_mat[1:],    # (n_cells, nbin) raw TCSPC
                        models       = _cell_models,     # (n_cells, nbin) fitted model (NaN: fit_per_cell = False)
                        pooled_decay = _pool_dec,
                        pooled_model = pooled_fit['model'])
    print(f'Saved: {csv_out}')
    print(f'Saved: {os.path.basename(_pool_csv)}, {os.path.basename(_fits_path)}  '
          f'(keys: tau_axis, irf_aligned, t_stop_ns, per_cell_fit, cell_ids, decays, models, pooled_decay, pooled_model)')
    print(df_cells[_show_cols].to_string(index=False))

# ── pooled fit: summary + QC plot (fit and residuals) ─────────────────────────
_comp = [(pooled_fit[f'tau{i}'], pooled_fit[f'frac{i}']) for i in range(1, 6)
         if np.isfinite(pooled_fit.get(f'tau{i}', np.nan))]
print(f'\\nPooled fit, all {n_cells} cells ({_pool_dec.sum():,.0f} photons, {_t_stop_str}):  '
      f'tau_int {pooled_fit["tau_int"]:.3f} ns   tau_amp {pooled_fit["tau_amp"]:.3f} ns   chi2 {pooled_fit["chi2"]:.1f}')
print('  components (share of photons): ' + ',  '.join(f'{t:.3f} ns ({100 * f:.0f}%)' for t, f in _comp))
if pooled_fit['fast_frac'] > 0.02:
    print(f'  NOTE: {100 * pooled_fit["fast_frac"]:.0f}% of the photons are in an IRF-shaped component (< 0.1 ns, left out '
          f'of tau_int / tau_amp): the IRF is probably broader than this recording\\'s (Step 14 notes).')
_pm = np.asarray(pooled_fit['model'], dtype=float)
fig_pool, ax_pool = plt.subplots(1, 2, figsize=(12, 3.8))
ax_pool[0].semilogy(tau, _pool_dec / _pool_dec.max(), '.', ms=3, color='steelblue', label='pooled decay (all cells)')
ax_pool[0].semilogy(tau, _pm / _pool_dec.max(), color='tomato', lw=1.0, label='Fit_decay')
ax_pool[0].semilogy(tau, IRF2_fit / IRF2_fit.max(), '--k', lw=0.6, label='IRF')
ax_pool[0].set(xlabel='time / ns', ylabel='intensity (norm.)', xlim=(0, Tcycle), ylim=(5e-3, 1.3))
ax_pool[1].plot(tau, (_pool_dec - _pm) / np.sqrt(np.maximum(_pm, 1.0)), color='0.3', lw=0.8)
ax_pool[1].axhline(0, color='tomato', lw=0.6)
ax_pool[1].set(xlabel='time / ns', ylabel='(data − fit) / √fit', xlim=(0, Tcycle))
if t_stop_ns:
    for _a in ax_pool:
        _a.axvline(t_stop_ns, color='red', ls='--', lw=0.8)
ax_pool[0].legend(fontsize=8)
fig_pool.suptitle(f'Pooled fit  [{FName}{cell_tag}]   tau_int {pooled_fit["tau_int"]:.3f} ns,  tau_amp '
                  f'{pooled_fit["tau_amp"]:.3f} ns,  fast {100 * pooled_fit["fast_frac"]:.1f}%,  chi2 {pooled_fit["chi2"]:.1f}',
                  fontsize=10)
fig_pool.tight_layout()
fig_pool.savefig(_pool_csv.replace('.csv', '.png'), dpi=150, bbox_inches='tight', facecolor='white')
plt.show()"""),
]

R['decay-inspect'] = [
("""_row = df_cells[df_cells["cell_id"] == CELL_NUM].iloc[0]
""",
"""_row = df_cells[df_cells["cell_id"] == CELL_NUM].iloc[0]
_mod_lbl = "model fit"
if not np.isfinite(_mod).any() and "pooled_model" in _fits.files:   # fit_per_cell = False: pooled fit, scaled
    _mod = _fits["pooled_model"].astype(float) * _raw.sum() / _fits["pooled_model"].sum()
    _mod_lbl = "pooled fit (scaled to this cell)"
"""),
("""ax.plot   (_t, _mod_n, "-",  color="tomato",    linewidth=1.8, zorder=3, label="model fit")""",
 """ax.plot   (_t, _mod_n, "-",  color="tomato",    linewidth=1.8, zorder=3, label=_mod_lbl)"""),
("""    f"aw_LT = {_row['aw_LT']:.3f} ns   "
    f"chi2 = {_row['chi2']:.2f}   "
    f"n_photons = {int(_row['n_photons']):,}"
""",
"""    + (f"aw_LT = {_row['aw_LT']:.3f} ns   chi2 = {_row['chi2']:.2f}   " if np.isfinite(_row['aw_LT'])
       else f"tav = {_row.get('tav_pooled', np.nan):.3f} ns   ")
    + f"n_photons = {int(_row['n_photons']):,}"
"""),
]

R['dc21f008'] = [(
"""- **Left** — dots colored by amplitude-weighted lifetime (`aw_LT`)
""",
"""- **Left** — dots colored by `aw_LT` of the per-cell fits (`fit_per_cell = True`), else by `tav_pooled`
"""), (
"""`G_phasor` / `S_phasor` are appended to the per-cell CSV.""",
"""`G_phasor` / `S_phasor` are appended to the per-cell CSV, with the lifetimes read from the phasor:
`tau_phi` = S / (ω G) (phase lifetime: needs only the phase, i.e. the IRF timing) and `tau_mod` =
√(1 / (G² + S²) − 1) / ω (modulation lifetime: needs the IRF's modulation, i.e. its width). Both equal τ for a
single exponential; for a mixture `tau_phi` < `tau_mod`.""")]

R['6e4891ec'] = [
("""    _G_corr_c, _S_corr_c, _, _ = phasor_ref_correction(
        _G_raw, _S_raw, G_ref, S_ref, tau_ref_ns=0.0, freq_mhz=freq_mhz)
""",
"""    _G_corr_c, _S_corr_c, _tmod_c, _tphi_c = phasor_ref_correction(
        _G_raw, _S_raw, G_ref, S_ref, tau_ref_ns=0.0, freq_mhz=freq_mhz)
"""),
("""                            for s in _S_corr_c]
    df_cells.to_csv(csv_out, index=False)
    print(f'Added G_phasor / S_phasor  →  {os.path.basename(csv_out)}')
""",
"""                            for s in _S_corr_c]
    df_cells['tau_phi'] = np.round(_tphi_c, 4)   # phase lifetime S / (ω G), ns
    df_cells['tau_mod'] = np.round(_tmod_c, 4)   # modulation lifetime, ns
    df_cells.to_csv(csv_out, index=False)
    print(f'Added G_phasor / S_phasor / tau_phi / tau_mod  →  {os.path.basename(csv_out)}  '
          f'(median tau_phi {np.nanmedian(_tphi_c):.3f} ns, tau_mod {np.nanmedian(_tmod_c):.3f} ns)')
"""),
("""    _aw     = df_cells['aw_LT'].values[_ok_c].astype(float)
""",
"""    _lt_col = ('aw_LT' if fit_per_cell and np.isfinite(df_cells['aw_LT'].values.astype(float)).any()
               else 'tav_pooled')   # per-cell fit lifetime, or the model-free mean arrival time
    _aw     = df_cells[_lt_col].values[_ok_c].astype(float)
"""),
("""            ['aw_LT (ns)', f'centroid z  (0 – {n_planes - 1})'],""",
 """            [f'{_lt_col} (ns)', f'centroid z  (0 – {n_planes - 1})'],"""),
("""    axes_cp[0].set_title(f'Per-cell phasor — color: aw_LT  [{FName}]', fontsize=10)""",
 """    axes_cp[0].set_title(f'Per-cell phasor — color: {_lt_col}  [{FName}]', fontsize=10)"""),
]

R['md-17'] = [(
"""- pooled 3D decay → `Fit_decay` (`aw_LT`, `chi2`, `tau1-3`, `amp1-3`), with the IRF, peak alignment, `t_stop`,
  `min_photons` and parallel `fit_workers` of Step 14;
- `tav_pooled`: mean arrival time of the pooled decay minus the IRF's (the Step 8 definition, per cell);
- `G_phasor`, `S_phasor`: corrected phasor of the pooled decay (as Step 15);
""",
"""- only with `fit_per_cell = True` (Step 1): pooled 3D decay → `Fit_decay` (`aw_LT`, `chi2`, `tau1-4`, `amp1-4`), with
  the IRF, peak alignment, `t_stop`, `min_photons` and parallel `fit_workers` of Step 14;
- `tav_pooled`: mean arrival time of the pooled decay minus the IRF's (the Step 8 definition, per cell);
- `G_phasor`, `S_phasor`, `tau_phi`, `tau_mod`: corrected phasor of the pooled decay and its lifetimes (as Step 15);
"""), (
"""→ `*_per_cell_timecourse.csv` (one row per cell × timepoint) + `*_per_cell_timecourse.png`.""",
"""Per timepoint, `Fit_decay` also runs once on the summed decay of the cells used for the summary (below) →
`*_pooled_fit_timecourse.csv` (`tau_int`, `tau_amp`, components, `fast_frac`; see Step 14) and the first panel of the plot.

→ `*_per_cell_timecourse.csv` (one row per cell × timepoint) + `*_per_cell_timecourse.png`.""")]

R['code-17'] = [
("""    _ok    = _dec_t[:, 1:].sum(axis=2) >= min_photons               # (timepoint, cell): decays to fit
    print(f'Fitting {int(_ok.sum())} of {_ok.size} decays ({n_cells} cells x {n_timepoints} timepoints, min_photons={min_photons}, '
          + (f't_stop={t_stop_ns:.3f} ns)...' if t_stop_ns else 't_stop=None)...'))
    _t0 = _time.perf_counter()
    _fit_tc = {(int(t), int(c) + 1): r for (t, c), r in
               zip(np.argwhere(_ok), fit_decays(IRF2_fit, _dec_t[:, 1:][_ok], Tcycle, dt, t_stop_ns=t_stop_ns,
                                                n_workers=fit_workers))}
""",
"""    _ok    = _dec_t[:, 1:].sum(axis=2) >= min_photons               # (timepoint, cell): enough photons
    _t0 = _time.perf_counter()
    if fit_per_cell:
        print(f'Fitting {int(_ok.sum())} of {_ok.size} decays ({n_cells} cells x {n_timepoints} timepoints, min_photons={min_photons}, '
              + (f't_stop={t_stop_ns:.3f} ns)...' if t_stop_ns else 't_stop=None)...'))
        _fit_tc = {(int(t), int(c) + 1): r for (t, c), r in
                   zip(np.argwhere(_ok), fit_decays(IRF2_fit, _dec_t[:, 1:][_ok], Tcycle, dt, t_stop_ns=t_stop_ns,
                                                    n_workers=fit_workers))}
    else:
        _fit_tc = {}
        print(f'Per-cell fits off (fit_per_cell = False, Step 1): tav_pooled, phasor and tau_phi for {int(_ok.sum())} of '
              f'{_ok.size} decays (>= {min_photons} photons); Fit_decay on the pooled decay of each timepoint below.')
"""),
("""            if (_t, cid) in _fit_tc:
                _t2, _a2, _, _, _chi2 = _fit_tc[(_t, cid)]
                _t2a, _a2a = np.asarray(_t2).ravel(), np.asarray(_a2).ravel()
                _v = _a2a > 0
                rec.update(aw_LT=round(float(np.dot(_t2a[_v], _a2a[_v]) / _a2a[_v].sum()), 4) if _v.any() else np.nan,
                           chi2=round(float(_chi2), 4), tav_pooled=round(float(np.dot(tau, _dec) / _nph - _H), 4),
                           G_raw=float(np.dot(np.cos(_T_c), _dec) / _nph), S_raw=float(np.dot(np.sin(_T_c), _dec) / _nph))
                for i in range(3):
                    rec[f'tau{i + 1}'] = round(float(_t2a[i]), 4) if i < _t2a.size else np.nan
                    rec[f'amp{i + 1}'] = round(float(_a2a[i]), 4) if i < _a2a.size else np.nan
            _recs.append(rec)
        print(f'  t{_t:02d}  {sum(r["timepoint"] == _t and np.isfinite(r["aw_LT"]) for r in _recs)} cells fitted')
""",
"""            if _ok[_t, cid - 1]:
                rec.update(tav_pooled=round(float(np.dot(tau, _dec) / _nph - _H), 4),
                           G_raw=float(np.dot(np.cos(_T_c), _dec) / _nph), S_raw=float(np.dot(np.sin(_T_c), _dec) / _nph))
            if (_t, cid) in _fit_tc:
                _t2, _a2, _, _, _chi2 = _fit_tc[(_t, cid)]
                _t2a, _a2a = np.asarray(_t2).ravel(), np.asarray(_a2).ravel()
                _v = _a2a > 0
                rec.update(aw_LT=round(float(np.dot(_t2a[_v], _a2a[_v]) / _a2a[_v].sum()), 4) if _v.any() else np.nan,
                           chi2=round(float(_chi2), 4))
                for i in range(4):   # Fit_decay returns up to max_components + 1 (the IRF-shaped one)
                    rec[f'tau{i + 1}'] = round(float(_t2a[i]), 4) if i < _t2a.size else np.nan
                    rec[f'amp{i + 1}'] = round(float(_a2a[i]), 4) if i < _a2a.size else np.nan
            _recs.append(rec)
        print(f'  t{_t:02d}  {sum(r["timepoint"] == _t and np.isfinite(r["tav_pooled"]) for r in _recs)} cells with '
              f'>= {min_photons} photons'
              + (f', {sum(r["timepoint"] == _t and np.isfinite(r["aw_LT"]) for r in _recs)} fitted' if fit_per_cell else ''))
"""),
("""        _gc, _sc, _, _ = phasor_ref_correction(df_tc['G_raw'].values, df_tc['S_raw'].values, G_ref, S_ref,
                                               tau_ref_ns=0.0, freq_mhz=freq_mhz)
        df_tc['G_phasor'], df_tc['S_phasor'] = np.round(_gc, 5), np.round(_sc, 5)
""",
"""        _gc, _sc, _tm, _tp = phasor_ref_correction(df_tc['G_raw'].values, df_tc['S_raw'].values, G_ref, S_ref,
                                                   tau_ref_ns=0.0, freq_mhz=freq_mhz)
        df_tc['G_phasor'], df_tc['S_phasor'] = np.round(_gc, 5), np.round(_sc, 5)
        df_tc['tau_phi'],  df_tc['tau_mod']  = np.round(_tp, 4), np.round(_tm, 4)
"""),
("""    _mcols  = ['aw_LT', 'tav_pooled'] + (['G_phasor', 'S_phasor'] if 'G_phasor' in df_tc else [])
    _summ   = _in_fov.groupby('timepoint').agg(cells_fitted=('aw_LT', 'count'), photons_median=('n_photons', 'median'),
                                               **{f'{c}_median': (c, 'median') for c in _mcols})
    print(f'\\nPer timepoint  ({_sel_txt}):')
    print(_summ.round(4).to_string())

    # ── plot: single cells (grey) and median + IQR per timepoint ──────────────
    _x = np.arange(n_timepoints)
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4))
    for _a, (_c, _ylab) in zip(axs, [('aw_LT', 'aw_LT (ns)'), ('tav_pooled', 'tav pooled (ns)'),
                                     ('n_photons', 'photons per cell')]):
""",
"""    _mcols  = (['aw_LT'] if fit_per_cell else []) + ['tav_pooled'] \\
              + (['tau_phi', 'G_phasor', 'S_phasor'] if 'G_phasor' in df_tc else [])
    _summ   = _in_fov.groupby('timepoint').agg(cells=('tav_pooled', 'count'), photons_median=('n_photons', 'median'),
                                               **{f'{c}_median': (c, 'median') for c in _mcols})
    print(f'\\nPer timepoint  ({_sel_txt}):')
    print(_summ.round(4).to_string())

    # ── pooled fit per timepoint: one Fit_decay on the summed decay of the same cells ──
    _pool_t  = np.stack([_dec_t[_t, _in_fov.loc[_in_fov['timepoint'] == _t, 'cell_id'].values.astype(int)].sum(axis=0)
                         for _t in range(n_timepoints)])
    _pool_ok = np.flatnonzero(_pool_t.sum(axis=1) > 0)
    _pool_r  = dict(zip(_pool_ok, fit_decays(IRF2_fit, _pool_t[_pool_ok], Tcycle, dt, t_stop_ns=t_stop_ns,
                                             n_workers=fit_workers, verbose=False)))
    df_pool_t = pd.DataFrame([dict(timepoint=_t, n_cells=int((_in_fov['timepoint'] == _t).sum()),
                                   n_photons=int(_pool_t[_t].sum()),
                                   **({k: v for k, v in pooled_lifetimes(IRF2_fit, _pool_t[_t], Tcycle, dt,
                                                                          fit=_pool_r[_t]).items() if k != 'model'}
                                      if _t in _pool_r else {}))
                              for _t in range(n_timepoints)])
    _pool_tc_csv = os.path.join(newFolder, f'{FName}{cell_tag}_pooled_fit_timecourse.csv')
    df_pool_t.to_csv(_pool_tc_csv, index=False)
    print(f'\\nPooled fit per timepoint ({_sel_txt}) → {os.path.basename(_pool_tc_csv)}:')
    print(df_pool_t.round(4).to_string(index=False))

    # ── plot: pooled fit, then single cells (grey) and median + IQR per timepoint ──
    _x = np.arange(n_timepoints)
    _panels = (([('aw_LT', 'aw_LT (ns)')] if fit_per_cell else []) + [('tav_pooled', 'tav pooled (ns)')]
               + ([('tau_phi', 'phase lifetime tau_phi (ns)')] if 'tau_phi' in df_tc else [])
               + [('n_photons', 'photons per cell')])
    fig, axs = plt.subplots(1, len(_panels) + 1, figsize=(4.6 * (len(_panels) + 1), 4.4))
    axs[0].plot(_x, df_pool_t.get('tau_int', np.full(n_timepoints, np.nan)), 'o-', color='C3', lw=2, ms=6,
                label='intensity-weighted')
    axs[0].plot(_x, df_pool_t.get('tau_amp', np.full(n_timepoints, np.nan)), 's--', color='C1', lw=1.5, ms=5,
                label='amplitude-weighted')
    axs[0].set(xlabel='timepoint', ylabel='pooled-fit mean lifetime (ns)', xticks=_x, title='Fit_decay, pooled decay')
    axs[0].legend(fontsize=8, frameon=False)
    axs[0].spines[['top', 'right']].set_visible(False)
    for _a, (_c, _ylab) in zip(axs[1:], _panels):
"""),
("""    axs[2].set_yscale('log')
    axs[0].legend(fontsize=8, frameon=False)
""",
"""    axs[-1].set_yscale('log')
    axs[1].legend(fontsize=8, frameon=False)
"""),
]


def _src(c):
    return ''.join(c['source'])


def main(nb_path=NB):
    nb = json.load(open(nb_path, encoding='utf-8'))
    cells = {c.get('id'): c for c in nb['cells']}
    if 'fit_per_cell' in _src(cells['code-03']):
        print(f'{os.path.basename(nb_path)}: already patched')
        return
    new = {}
    for cid, reps in R.items():
        s = _src(cells[cid])
        for k, (old, rep) in enumerate(reps):
            if s.count(old) != 1:
                sys.exit(f'{cid} replacement {k}: found {s.count(old)} times, nothing written')
            s = s.replace(old, rep)
        new[cid] = s
    for cid, s in new.items():
        cells[cid]['source'] = s.splitlines(keepends=True)
    with open(nb_path, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(nb, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(f'{os.path.basename(nb_path)}: patched ' + ', '.join(new))


if __name__ == '__main__':
    main(*sys.argv[1:])
