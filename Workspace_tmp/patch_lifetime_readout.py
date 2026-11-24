"""Per-cell lifetime readout + IRF width (2026-11-23), all three TCSPC notebooks.

1. Tile + base notebooks: the fit_per_cell switch of patch_fit_per_cell.py (default False: no Fit_decay per cell,
   one Fit_decay on the pooled decay of all cells). The base notebook's sequential per-cell loop becomes the
   parallel fit_decays of the XYTZ notebooks (results identical; it gets cell_tag = '').
2. All three: Step 14 also stores the per-cell phasor (G_phasor, S_phasor) and its phase / modulation lifetimes
   (tau_phi, tau_mod) next to tav_pooled; the decay inspection shows them; Step 17 (Time) gets a tau_mod panel.
3. All three: Step 5 rescales the IRF width to the recording (flim_helpers.fit_irf_width) when the IRF is not from
   the same session (irf_width_mode = 'auto' | 'always' | 'off', Step 1), documented in the Step 5 notes.

Idempotent per part (markers: 'fit_per_cell', 'tau_mod_c', 'irf_width_mode'); every replacement must match once.
Usage: python Workspace_tmp/patch_lifetime_readout.py [notebook ...]   (default: the three notebooks)
"""
import json, os, sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, 'Workspace_tmp'))
import patch_fit_per_cell as PF                     # noqa: E402

NOTEBOOKS = ['AnalyzeTCSPC_XYTZTime.ipynb', 'AnalyzeTCSPC_XYTZTile.ipynb', 'AnalyzeTCSPC.ipynb']

FPC_PARAMS = PF.R['code-03'][0][1].split('\n\n', 1)[1]          # the fit_per_cell block of patch_fit_per_cell

IRF_PARAMS = """
# ── IRF width (Step 5) ────────────────────────────────────────────────────────
irf_width_mode   = 'auto'      # 'auto': rescale the IRF width to this recording when the IRF is not from the same session
                               # (no IRF .fbs.xml, or another date / laser period); 'always'; 'off' (IRF as measured)
irf_width_win_ns = 3.0         # fit window after the peak (ns): rising edge + early decay, before any reflection bump
"""

# ── base notebook: Step 14 parts that differ from the XYTZ notebooks ─────────
BASE_14 = [
("""from flim_helpers import Fit_decay
from scipy.signal import find_peaks
""",
"""from flim_helpers import Fit_decay, fit_decays, pooled_lifetimes
from scipy.signal import find_peaks
"""),
("""reRunPerCellDecay  = True   # set True to force re-accumulation (overwrites cache)
""",
"""reRunPerCellDecay  = True   # set True to force re-accumulation (overwrites cache)
fit_workers        = None   # fitting processes: None = all cores but one, 1 = sequential
cell_tag           = ''     # output-name tag (the XYTZ notebooks use it for subject-reference cells)
"""),
("""_fits_path  = os.path.join(newFolder, f'{FName}_cell_fits.npz')   # decay + model per cell
""",
"""_fits_path  = os.path.join(newFolder, f'{FName}_cell_fits.npz')   # decay + model per cell
_pool_csv   = os.path.join(newFolder, f'{FName}_pooled_fit.csv')  # Fit_decay of the pooled decay of all cells
"""),
("""# STAGE 2 — per-cell Fit_decay  (sequential; cached in csv_out + _fits_path)""",
 """# STAGE 2 — per-cell Fit_decay  (parallel processes, fit_workers; cached in csv_out + _fits_path)"""),
PF.R['b3fb5433'][2],
PF.R['b3fb5433'][3],
("""    print(df_cells[['cell_id', 'n_voxels', 'n_photons', 'aw_LT', 'chi2']].to_string(index=False))
else:
    records      = []
    _cell_models = np.full((n_cells, nbin), np.nan)   # row i = cell i+1

    _t_stop_str = f't_stop={t_stop_ns:.3f} ns' if t_stop_ns else 't_stop=None'
    print(f'Fitting {n_cells} cells  (min_photons={min_photons},  {_t_stop_str})...')
    _t_fit = _time.perf_counter()
""", PF.R['b3fb5433'][4][1]),
("""                   centroid_x=round(cx, 2), dist_from_center_um=_dist_um)
        if n_ph < min_photons:
            rec.update(aw_LT=np.nan, chi2=np.nan)
            for i in range(1, 6):
                rec[f'tau{i}'] = np.nan;  rec[f'amp{i}'] = np.nan
            print(f'  cell {cid:3d}  SKIP  (n_ph={n_ph})')
        else:
            print(f'  cell {cid:3d}  fitting...', end='\\r', flush=True)
            _t2, _a2, _, _model_curve, _chi2, _ = Fit_decay(
                IRF2_fit, decay, Tcycle, dt, t_stop_ns=t_stop_ns)
""", PF.R['b3fb5433'][5][1] + """        else:
            _t2, _a2, _, _model_curve, _chi2 = _fit_res[cid]
"""),
PF.R['b3fb5433'][6],
]
BASE_14MD = [("""- **Parallel fitting**: `ThreadPoolExecutor` runs `Fit_decay` concurrently across cells.""",
              """- **Parallel fitting**: `fit_decays` runs `Fit_decay` in worker processes (`fit_workers`: `None` = all cores but one, `1` = sequential); the results are identical to sequential fits.""")]

# ── part 2: per-cell phasor lifetimes ─────────────────────────────────────────
PH_14 = [
("""_show_cols  = ['cell_id', 'n_voxels', 'n_photons'] + (['aw_LT', 'chi2'] if fit_per_cell else ['tav_pooled'])
""",
"""_show_cols  = (['cell_id', 'n_voxels', 'n_photons'] + (['aw_LT', 'chi2'] if fit_per_cell else [])
               + ['tav_pooled', 'tau_phi', 'tau_mod'])
"""),
("""        print(f'Per-cell fits off (fit_per_cell = False, Step 1): tav_pooled for {len(_fit_ids)} of {n_cells} cells '
              f'(>= {min_photons} photons); phasor and tau_phi in Step 15.')
""",
"""        print(f'Per-cell fits off (fit_per_cell = False, Step 1): tav_pooled, phasor, tau_phi and tau_mod for '
              f'{len(_fit_ids)} of {n_cells} cells (>= {min_photons} photons).')
"""),
("""    df_cells = pd.DataFrame(records)
""",
"""    df_cells = pd.DataFrame(records)

    # ── phasor per cell (the Step 15 definition): G, S, phase lifetime tau_phi, modulation lifetime tau_mod_c ──
    _ok_ph = df_cells['n_photons'].values >= min_photons
    _tp_ph = (np.arange(1, nbin + 1) - 0.5) * dt
    _T_ph  = 2.0 * np.pi * (_tp_ph - np.dot(_tp_ph, IRF2) / IRF2.sum()) / Tcycle   # phase 0 = IRF centroid (Step 13)
    _w_ph  = _cell_mat[1:] / np.maximum(_cell_mat[1:].sum(axis=1, keepdims=True), 1.0)
    _gc, _sc, _tau_mod_c, _tau_phi_c = phasor_ref_correction(
        np.where(_ok_ph, _w_ph @ np.cos(_T_ph), np.nan), np.where(_ok_ph, _w_ph @ np.sin(_T_ph), np.nan),
        np.array([[np.dot(np.cos(_T_ph), IRF2) / IRF2.sum()]]), np.array([[np.dot(np.sin(_T_ph), IRF2) / IRF2.sum()]]),
        tau_ref_ns=0.0, freq_mhz=1000.0 / Tcycle)
    df_cells['G_phasor'], df_cells['S_phasor'] = np.round(_gc, 5), np.round(_sc, 5)
    df_cells['tau_phi'],  df_cells['tau_mod']  = np.round(_tau_phi_c, 4), np.round(_tau_mod_c, 4)
"""),
("""of tau_int / tau_amp): the IRF is probably broader than this recording\\'s (Step 14 notes).')""",
 """of tau_int / tau_amp): the IRF is probably broader than this recording\\'s (Step 5: irf_width_mode).')"""),
]
PH_INSPECT = [(
"""       else f"tav = {_row.get('tav_pooled', np.nan):.3f} ns   ")""",
"""       else f"tav {_row.get('tav_pooled', np.nan):.3f}  tau_phi {_row.get('tau_phi', np.nan):.3f}  "
            f"tau_mod {_row.get('tau_mod', np.nan):.3f} ns   ")""")]
PH_14MD = [(
"""- per cell only `tav_pooled` (mean photon arrival time of the cell's decay minus the IRF's, as Step 17); the phasor,
  `tau_phi` and `tau_mod` come in Step 15 (SD at 700 photons: 0.08 ns);
""",
"""- per cell only model-free values: `tav_pooled` (mean photon arrival time of the cell's decay minus the IRF's, as
  Step 17) and the phasor `G_phasor`, `S_phasor` with its phase lifetime `tau_phi` = S / (ω G) and modulation lifetime
  `tau_mod` = √(1/(G² + S²) − 1) / ω (Step 15 plots them). SD at 700 photons: `tav` and `tau_phi` 0.08 ns;
  `tau_mod` depends on the IRF width (Step 5);
"""), (
"""  allows; a fitted IRF width of 0.61–0.67× removes it, chi2 286 → 34). Record an IRF on the imaging day.
""",
"""  allows; a fitted IRF width of 0.61–0.67× removes it, chi2 286 → 34). Step 5 now rescales the IRF width when the
  IRF is not from the same session (`irf_width_mode`); an IRF recorded on the imaging day is still best.
""")]
PH_17 = [
("""        print(f'Per-cell fits off (fit_per_cell = False, Step 1): tav_pooled, phasor and tau_phi for {int(_ok.sum())} of '""",
 """        print(f'Per-cell fits off (fit_per_cell = False, Step 1): tav_pooled, phasor, tau_phi, tau_mod for {int(_ok.sum())} of '"""),
("""                  + (['tau_phi', 'G_phasor', 'S_phasor'] if 'G_phasor' in df_tc else [])""",
 """                  + (['tau_phi', 'tau_mod', 'G_phasor', 'S_phasor'] if 'G_phasor' in df_tc else [])"""),
("""                   + ([('tau_phi', 'phase lifetime tau_phi (ns)')] if 'tau_phi' in df_tc else [])""",
 """                   + ([('tau_phi', 'phase lifetime tau_phi (ns)'), ('tau_mod', 'modulation lifetime tau_mod (ns)')]
                      if 'tau_phi' in df_tc else [])"""),
]

# ── part 3: IRF width in Step 5 ───────────────────────────────────────────────
IRFW_CODE = [(
"""print(f'IRF   peak was frame {peak_irf:3d},  shifted by {shift_irf:+d}')
""",
"""print(f'IRF   peak was frame {peak_irf:3d},  shifted by {shift_irf:+d}')

# ── IRF width: rescale a stored IRF to this recording (irf_width_mode, Step 1; see the Step 5 notes) ──
# fit_irf_width: IRF stretched by a factor s about its peak (and shifted by c), convolved with 2 free exponentials
# + background, fitted to this plane's decay up to irf_width_win_ns after the peak; s = 1 gives the measured IRF's chi2.
# The rescaled IRF replaces IRF2 for every later step when chi2 drops by > 20 %. The shift c is only a nuisance
# parameter: IRF and data stay peak-aligned (above) and Fit_decay fits its own shift.
import flim_helpers as _fh_mod; importlib.reload(_fh_mod)
from flim_helpers import fit_irf_width
_irf_date  = (meta_irf or {}).get('datetime') or ''
_data_date = (meta or {}).get('datetime') or ''
_same_session = bool(_irf_date[:10] and _irf_date[:10] == _data_date[:10]
                     and abs((meta_irf.get('Tcycle') or 0) - (meta.get('Tcycle') or 0)) < 0.01)
_why = ('IRF from the same session (.fbs.xml dates and laser period match)' if _same_session else
        'IRF has no .fbs.xml' if not meta_irf else f'IRF of {_irf_date[:10]}, data of {_data_date[:10] or "?"}')
irf_width = None
if irf_width_mode == 'always' or (irf_width_mode == 'auto' and not _same_session):
    _y_w = DataTCSPC.sum(axis=(0, 1)).astype(float)          # whole middle plane, peak-aligned above
    irf_width = fit_irf_width(IRF2, _y_w, dt, win_ns=irf_width_win_ns)
    print(f'IRF width fit ({_why}): factor {irf_width["s"]:.3f} (shift {irf_width["c"]:+.2f} bins, not applied);  '
          f'chi2 {irf_width["chi2_measured"]:.1f} with the measured IRF → {irf_width["chi2_fitted"]:.1f}'
          + ('  → IRF2 rescaled' if irf_width['apply'] else '  → < 20 % better: IRF kept as measured'))
    import json as _json
    with open(os.path.join(newFolder, f'{FName}_irf_width.json'), 'w') as _f:
        _json.dump(dict(mode=irf_width_mode, reason=_why, irf=os.path.basename(irf_path), factor=irf_width['s'],
                        shift_bins=irf_width['c'], chi2_measured=irf_width['chi2_measured'],
                        chi2_fitted=irf_width['chi2_fitted'], applied=irf_width['apply'], win_ns=irf_width_win_ns),
                   _f, indent=1)
    fig_w, ax_w = plt.subplots(1, 2, figsize=(12, 3.6))
    ax_w[0].plot(tau, irf_width['irf_measured'] / irf_width['irf_measured'].max(), '--k', lw=1, label='IRF as measured')
    ax_w[0].plot(tau, irf_width['irf'] / irf_width['irf'].max(), color='C3', lw=1.2,
                 label=f'IRF x {irf_width["s"]:.2f} (width)')
    ax_w[0].plot(tau, _y_w / _y_w.max(), '.', color='steelblue', ms=4, label='data (middle plane)')
    ax_w[0].set(xlim=(tau[max(target - 6, 0)], tau[min(target + 25, nbin - 1)]), xlabel='time / ns',
                ylabel='intensity (norm.)', title='rising edge')
    for _m, _c, _l in ((irf_width['model_measured'], 'k', 'IRF as measured'), (irf_width['model_fitted'], 'C3', 'rescaled IRF')):
        _st = irf_width['stop']
        ax_w[1].plot(tau[:_st], ((_y_w - _m) / np.sqrt(np.maximum(_m, 1.0)))[:_st], color=_c, lw=1, label=_l)
    ax_w[1].axhline(0, color='0.6', lw=0.5)
    ax_w[1].set(xlabel='time / ns', ylabel='(data − model) / √model',
                title=f'fit window: to {irf_width_win_ns} ns after the peak')
    for _a in ax_w:
        _a.legend(fontsize=8)
    fig_w.suptitle(f'IRF width  [{FName}]   factor {irf_width["s"]:.3f},  chi2 {irf_width["chi2_measured"]:.0f} → '
                   f'{irf_width["chi2_fitted"]:.0f}' + ('' if irf_width['apply'] else '   (not applied)'), fontsize=10)
    fig_w.tight_layout()
    plt.show()
    if irf_width['apply']:
        IRF2 = irf_width['irf']
else:
    print(f'IRF width fit skipped ({_why}; irf_width_mode = {irf_width_mode!r}).')
""")]
IRFW_MD = [(
"""Circularly shifts `DataTCSPC` and `IRF2` so both peaks land at 8 % of `nbin` (the same target frame). This single global alignment replaces the old per-session manual shift and ensures `F - H` is unbiased.""",
"""Circularly shifts `DataTCSPC` and `IRF2` so both peaks land at 8 % of `nbin` (the same target frame). This single global alignment replaces the old per-session manual shift and ensures `F - H` is unbiased.

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
stretched by a factor *s* about its peak and shifted by *c* bins, convolved with 2 free exponentials + background
(nuisance parameters); *s* = 1 gives the measured IRF's chi2. When chi2 drops by more than 20 %, the rescaled IRF
(*s* applied, *c* not: IRF and data stay peak-aligned and `Fit_decay` fits its own shift; the IRF beyond ±2.5 ns of its
peak, with its echoes, is cut) replaces `IRF2`, so the tav maps (Step 8), the phasor reference (Steps 13, 15, 17) and
all fits use it. `'auto'` fits when the IRF has no `.fbs.xml` or its date or laser period differs from the data's,
`'always'` fits anyway, `'off'` keeps the measured IRF. Output: printout, QC plot (rising edge; residuals with both
IRFs) and `*_irf_width.json`. An older z-stack (XYZ_288, its own IRF): chi2 94 → 90, not applied. An IRF recorded in
the same session, with its `.fbs.xml`, is still the better reference.""")]


def _src(c):
    return ''.join(c['source'])


def _apply(s, reps, where):
    for k, (old, new) in enumerate(reps):
        if s.count(old) != 1:
            sys.exit(f'{where} replacement {k}: found {s.count(old)} times, nothing written')
        s = s.replace(old, new)
    return s


def patch(nb_path):
    name = os.path.basename(nb_path)
    nb = json.load(open(nb_path, encoding='utf-8'))
    cells = {c.get('id'): c for c in nb['cells']}
    is_base = 'code-17' not in cells and 'from flim_helpers import Fit_decay\nfrom scipy' in _src(cells['b3fb5433'])
    new, done = {cid: _src(c) for cid, c in cells.items() if cid}, []

    # 1. fit_per_cell switch (Tile, base)
    if 'fit_per_cell' not in new['code-03']:
        new['code-03'] = _apply(new['code-03'], [('szBin     = 3\n', 'szBin     = 3\n\n' + FPC_PARAMS)], f'{name} code-03')
        for cid, reps in PF.R.items():
            if cid in ('code-03', 'md-17', 'code-17'):
                continue
            if cid == 'b3fb5433' and is_base:
                reps = BASE_14
            if cid == 'ad5b1d96' and is_base:
                reps = BASE_14MD + reps
            new[cid] = _apply(new[cid], reps, f'{name} {cid}')
        done.append('fit_per_cell')

    # 2. per-cell phasor lifetimes
    if '_tau_mod_c' not in new['b3fb5433']:
        new['b3fb5433'] = _apply(new['b3fb5433'], PH_14, f'{name} b3fb5433 (phasor)')
        new['decay-inspect'] = _apply(new['decay-inspect'], PH_INSPECT, f'{name} decay-inspect')
        new['ad5b1d96'] = _apply(new['ad5b1d96'], PH_14MD, f'{name} ad5b1d96')
        if 'code-17' in new:
            new['code-17'] = _apply(new['code-17'], PH_17, f'{name} code-17')
        done.append('phasor lifetimes per cell')

    # 3. IRF width
    if 'irf_width_mode' not in new['code-03']:
        anchor = FPC_PARAMS.rstrip('\n').splitlines()[-1] + '\n'
        new['code-03'] = _apply(new['code-03'], [(anchor, anchor + IRF_PARAMS)], f'{name} code-03 (IRF)')
        new['5622bfa1'] = _apply(new['5622bfa1'], IRFW_CODE, f'{name} 5622bfa1')
        new['1a7938ca'] = _apply(new['1a7938ca'], IRFW_MD, f'{name} 1a7938ca')
        done.append('IRF width')

    if not done:
        print(f'{name}: already patched')
        return
    for cid, c in cells.items():
        if cid and new[cid] != _src(c):
            c['source'] = new[cid].splitlines(keepends=True)
    with open(nb_path, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(nb, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(f'{name}: ' + ', '.join(done) + (' (base notebook)' if is_base else ''))


if __name__ == '__main__':
    for p in (sys.argv[1:] or [os.path.join(HERE, n) for n in NOTEBOOKS]):
        patch(p)
