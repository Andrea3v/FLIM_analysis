"""Step 14 t_stop + analysis-folder name (2026-11-23), all three TCSPC notebooks.

1. Step 14 (cell b3fb5433): if no bump is found, fit the whole decay (t_stop_ns = None) instead of stopping at
   the search-window start, and set the IRF tail (> irf_tail_ns after its peak) to baseline so the echoes of the
   IRF recording don't enter the model. Cached fits made with a different t_stop are re-fitted.
2. Step 1 (cell code-03): analysis folder named <file name>_<YYYYMMDD_HHMM>_FLIM_Analysis; existing folders
   (with or without the name in front) are found and sorted by their time stamp.

Idempotent: a notebook that already has the change is left alone, so it can be re-run after a VS Code save.
Usage: python Workspace_tmp/patch_tstop_folder.py
"""
import json, os, sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTEBOOKS = ['AnalyzeTCSPC_XYTZTime.ipynb', 'AnalyzeTCSPC_XYTZTile.ipynb', 'AnalyzeTCSPC.ipynb']

BUMP_START, BUMP_END = '# ── Auto-detect coherent artifact bump', 'ax_bulk.set_xlabel'

BUMP_OLD = """# ── Auto-detect coherent artifact bump ────────────────────────────────────────
# Tweak these if the detected bump is wrong (check the shaded region on the plot).
bump_srch_lo_ns    = 3.5   # ← start of search window, ns after peak; raise if still in decay tail
bump_srch_hi_ns    = 7.0   # ← end of search window
bump_safety_bins   = 4     # ← bins to cut before bump centre → t_stop
bump_min_prominence = 0.02  # ← min prominence as fraction of window peak (lower = more sensitive)
_srch_lo = min(nbin - 1, target + max(5, int(round(bump_srch_lo_ns / dt))))
_srch_hi = min(nbin,     target + int(round(bump_srch_hi_ns / dt)))
if _srch_hi > _srch_lo + 3:
    _smooth  = np.convolve(bulk_decay, np.ones(5) / 5, mode='same')
    _win     = _smooth[_srch_lo:_srch_hi]
    _win_n   = _win / _win.max() if _win.max() > 0 else _win
    _peaks, _props = find_peaks(_win_n, prominence=bump_min_prominence)
    print(f'Search window: {tau[_srch_lo]:.2f} – {tau[min(_srch_hi-1,nbin-1)]:.2f} ns  '
          f'(bins {_srch_lo}–{_srch_hi})  |  {len(_peaks)} peak(s) found')
    if _peaks.size:
        _best_rel   = _peaks[np.argmax(_win_n[_peaks])]
        _bump_bin   = _srch_lo + int(_best_rel)
        _bump_ns    = float(tau[_bump_bin])
        _t_stop_bin = max(0, _bump_bin - bump_safety_bins)
        t_stop_ns   = float(tau[_t_stop_bin])
        print(f'Coherent bump: bin {_bump_bin}  ({_bump_ns:.3f} ns)  '
              f'→  t_stop = {t_stop_ns:.3f} ns  (bin {_t_stop_bin})')
    else:
        _bump_bin   = _srch_lo
        _bump_ns    = float(tau[_bump_bin])
        _t_stop_bin = max(0, _bump_bin - bump_safety_bins)
        t_stop_ns   = float(tau[_t_stop_bin])
        print(f'WARNING: no peak found (prominence >= {bump_min_prominence}) in bump search window — '
              f'defaulting bump to window start ({_bump_ns:.3f} ns). '
              f'Lower bump_min_prominence or adjust bump_srch_lo/hi_ns.')
        print(f'  t_stop = {t_stop_ns:.3f} ns  (bin {_t_stop_bin})')
else:
    t_stop_ns = None;  _bump_ns = None
    print('Bump search window too narrow — fitting full decay.')

# ── QC: bulk decay plot ───────────────────────────────────────────────────────
_bulk_n = bulk_decay / bulk_decay.max()
_irf_n  = IRF2_fit  / IRF2_fit.max()
fig_bulk, ax_bulk = plt.subplots(figsize=(7, 4))
ax_bulk.semilogy(tau, _irf_n,  '--k', lw=0.8, label='IRF (aligned)')
ax_bulk.semilogy(tau, _bulk_n, color='steelblue', lw=1.2, label='Bulk 3D decay')
ax_bulk.axvline(tau[target], color='orange', lw=0.8, ls=':', label=f'peak  bin {target}  (8% target)')
if t_stop_ns is not None:
    ax_bulk.axvspan(tau[_srch_lo], tau[min(_srch_hi - 1, nbin - 1)],
                    alpha=0.08, color='red', label='search window')
    ax_bulk.axvline(_bump_ns,  color='orangered', lw=0.8, ls=':', label=f'bump  {_bump_ns:.2f} ns')
    ax_bulk.axvline(t_stop_ns, color='red',       lw=1.0, ls='--', label=f't_stop  {t_stop_ns:.2f} ns')
"""

BUMP_NEW = """# ── Auto-detect coherent artifact bump ────────────────────────────────────────
# Some recordings show a small bump a few ns after the peak (an echo of the excitation; the IRF recording
# itself has two, ~3.4 and ~4.9 ns after its peak).
#   bump found -> the fit stops bump_safety_bins before it (t_stop_ns).
#   no bump    -> the whole decay is fitted (t_stop_ns = None), and the IRF after irf_tail_ns is set to its
#                 baseline so the echoes of the IRF recording don't put a bump into the model.
# (Until 2026-11-23 the no-bump case stopped at the search-window start, ~3.1 ns after the peak. Slice3, 80 cells
#  split into 700-photon halves: per-cell SD of the fitted lifetime 0.51 ns with that stop, 0.34 ns whole decay.)
# Tweak these if the detected bump is wrong (check the shaded region on the plot).
bump_srch_lo_ns    = 3.5   # ← start of search window, ns after peak; raise if still in decay tail
bump_srch_hi_ns    = 7.0   # ← end of search window
bump_safety_bins   = 4     # ← bins to cut before bump centre → t_stop
bump_min_prominence = 0.02  # ← min prominence as fraction of window peak (lower = more sensitive)
irf_tail_ns        = 2.5   # ← no bump: IRF kept up to this long after its peak, baseline after (None = as measured)
_srch_lo = min(nbin - 1, target + max(5, int(round(bump_srch_lo_ns / dt))))
_srch_hi = min(nbin,     target + int(round(bump_srch_hi_ns / dt)))
t_stop_ns = None;  _bump_ns = None
if _srch_hi > _srch_lo + 3:
    _smooth  = np.convolve(bulk_decay, np.ones(5) / 5, mode='same')
    _win     = _smooth[_srch_lo:_srch_hi]
    _win_n   = _win / _win.max() if _win.max() > 0 else _win
    _peaks, _props = find_peaks(_win_n, prominence=bump_min_prominence)
    print(f'Search window: {tau[_srch_lo]:.2f} – {tau[min(_srch_hi-1,nbin-1)]:.2f} ns  '
          f'(bins {_srch_lo}–{_srch_hi})  |  {len(_peaks)} peak(s) found')
    if _peaks.size:
        _best_rel   = _peaks[np.argmax(_win_n[_peaks])]
        _bump_bin   = _srch_lo + int(_best_rel)
        _bump_ns    = float(tau[_bump_bin])
        _t_stop_bin = max(0, _bump_bin - bump_safety_bins)
        t_stop_ns   = float(tau[_t_stop_bin])
        print(f'Coherent bump: bin {_bump_bin}  ({_bump_ns:.3f} ns)  '
              f'→  t_stop = {t_stop_ns:.3f} ns  (bin {_t_stop_bin})')
    else:
        print(f'No bump found (prominence >= {bump_min_prominence}) → fitting the whole decay (t_stop = None). '
              f'If the plot shows a bump, lower bump_min_prominence or adjust bump_srch_lo/hi_ns.')
else:
    print('Bump search window too narrow — fitting the whole decay.')
if t_stop_ns is None and irf_tail_ns is not None and target + int(round(irf_tail_ns / dt)) < nbin:
    _irf_cut = target + int(round(irf_tail_ns / dt))
    IRF2_fit = IRF2_fit.copy()
    IRF2_fit[_irf_cut:] = np.median(IRF2_fit[_irf_cut:])
    print(f'IRF after {tau[_irf_cut]:.2f} ns set to its baseline (removes the echoes of the IRF recording).')
if _have_fits:   # cached fits made with another t_stop are re-fitted
    _ts_cached = float(np.load(_fits_path)['t_stop_ns'][0])
    if not (np.isnan(_ts_cached) if t_stop_ns is None else np.isclose(_ts_cached, t_stop_ns)):
        _have_fits = False
        print(f'Cached fits used t_stop = {_ts_cached:.3f} ns → re-fitting.')

# ── QC: bulk decay plot ───────────────────────────────────────────────────────
_bulk_n = bulk_decay / bulk_decay.max()
_irf_n  = IRF2_fit  / IRF2_fit.max()
fig_bulk, ax_bulk = plt.subplots(figsize=(7, 4))
ax_bulk.semilogy(tau, _irf_n,  '--k', lw=0.8, label='IRF (aligned)')
ax_bulk.semilogy(tau, _bulk_n, color='steelblue', lw=1.2, label='Bulk 3D decay')
ax_bulk.axvline(tau[target], color='orange', lw=0.8, ls=':', label=f'peak  bin {target}  (8% target)')
if _srch_hi > _srch_lo + 3:
    ax_bulk.axvspan(tau[_srch_lo], tau[min(_srch_hi - 1, nbin - 1)],
                    alpha=0.08, color='red', label='bump search window')
if _bump_ns is not None:
    ax_bulk.axvline(_bump_ns,  color='orangered', lw=0.8, ls=':', label=f'bump  {_bump_ns:.2f} ns')
if t_stop_ns is not None:
    ax_bulk.axvline(t_stop_ns, color='red',       lw=1.0, ls='--', label=f't_stop  {t_stop_ns:.2f} ns')
else:
    ax_bulk.plot([], [], ' ', label='no bump: whole decay fitted')
"""

FOLDER = [
    ("""_existing = sorted(
    [d for d in glob.glob(os.path.join(data_folder, '*_FLIM_Analysis'))
     if os.path.isdir(d)]
)   # lexicographic sort == chronological (YYYYMMDD_HHMM prefix)
""",
     """_name_tag = re.sub(r'_Decay\\d+_.*$', '', FName)   # file name without the Vista suffix (_Decay128_512x512_...)
_existing = sorted(
    [d for d in glob.glob(os.path.join(data_folder, '*_FLIM_Analysis'))
     if os.path.isdir(d)],
    key=lambda d: (re.findall(r'(\\d{8}_\\d{4})_FLIM_Analysis$', d) or [''])[0]
)   # chronological by the YYYYMMDD_HHMM stamp (folders made before 2026-11-23 have no name in front)
"""),
    ("newFolder = os.path.join(data_folder, f'{_ts}_FLIM_Analysis')",
     "newFolder = os.path.join(data_folder, f'{_name_tag}_{_ts}_FLIM_Analysis')"),
]
MD_TREE = ('<YYYYMMDD_HHMM>_FLIM_Analysis/   ', '<name>_<YYYYMMDD_HHMM>_FLIM_Analysis/')


def _src(c):
    return ''.join(c['source'])


def _set(c, s):
    c['source'] = s.splitlines(keepends=True)


def patch(nb_path):
    nb = json.load(open(nb_path, encoding='utf-8'))
    cells = {c.get('id'): c for c in nb['cells']}
    done = []

    s = _src(cells['b3fb5433'])
    if 'irf_tail_ns' not in s:
        i, j = s.index(BUMP_START), s.index(BUMP_END)
        if s[i:j] != BUMP_OLD:
            sys.exit(f'{nb_path}: Step 14 bump block differs from the expected text, nothing written')
        _set(cells['b3fb5433'], s[:i] + BUMP_NEW + s[j:])
        done.append('Step 14 t_stop')

    s = _src(cells['code-03'])
    if '_name_tag' not in s:
        for old, new in FOLDER:
            if s.count(old) != 1:
                sys.exit(f'{nb_path}: Step 1 folder text not found once, nothing written')
            s = s.replace(old, new)
        _set(cells['code-03'], s)
        done.append('Step 1 folder name')

    for c in nb['cells']:
        if c['cell_type'] == 'markdown' and MD_TREE[0] in _src(c):
            # keep the tree's column alignment: the longer name replaces the padding after it
            s = _src(c).replace(MD_TREE[0], MD_TREE[1] + ' ' * max(1, len(MD_TREE[0]) - len(MD_TREE[1])))
            _set(c, s)
            done.append('folder tree (markdown)')

    if done:
        with open(nb_path, 'w', encoding='utf-8', newline='\n') as f:
            json.dump(nb, f, ensure_ascii=False, indent=2)
            f.write('\n')
    print(f'{os.path.basename(nb_path)}: ' + (', '.join(done) if done else 'already patched'))


if __name__ == '__main__':
    for name in NOTEBOOKS:
        patch(os.path.join(HERE, name))
