"""Branch the time-series logic out of AnalyzeTCSPC_XYTZTile.ipynb.

AnalyzeTCSPC_XYTZTime.ipynb (new): the current notebook (time series in Step 0, Step 17), outputs cleared,
  slice parameters with n_timepoints = 4 (manual, no .oir), warns when n_timepoints = 1.
AnalyzeTCSPC_XYTZTile.ipynb: Step 0 cells back to their state before the time-series patch (single acquisition),
  Step 17 removed, tiled 336 dataset as raw_path, error on a time-lapse .oir (-> Time notebook).
Both: Step 14 fits in parallel worker processes (flim_helpers.fit_decays, fit_workers); the Time notebook's Step 17
  uses the same."""
import copy
import json
import os
import re
import shutil

ROOT = r'X:\FLIM_2026'
TILE = os.path.join(ROOT, 'AnalyzeTCSPC_XYTZTile.ipynb')
TIME = os.path.join(ROOT, 'AnalyzeTCSPC_XYTZTime.ipynb')
SCR = r'C:\Users\Andrea\AppData\Local\Temp\claude\X--FLIM-2026\c8d93f44-2eb1-4357-84ee-b7563a8b7110\scratchpad'
BEFORE_TS = os.path.join(SCR, 'AnalyzeTCSPC_XYTZTile.before_time_series.ipynb')
assert not os.path.exists(TIME), 'Time notebook exists already'

raw = open(TILE, encoding='utf-8', newline='').read()
indent = len(raw.splitlines()[1]) - len(raw.splitlines()[1].lstrip(' '))
nb = json.loads(raw)
assert any(c.get('id') == 'code-17' for c in nb['cells']), 'expected the time-series version'
shutil.copy2(TILE, os.path.join(SCR, 'AnalyzeTCSPC_XYTZTile.before_branch.ipynb'))
before_ts = {c.get('id'): c for c in json.load(open(BEFORE_TS, encoding='utf-8'))['cells']}

TILED_RAW = (r'D:\Peredox_FLIM\XYZTTile1_336_X0.4Y-2.8_Decay128_512x512_Z210_Ch1_T1_P1_S1'
             r'\XYZTTile1_336_X0.4Y-2.8_Decay128_512x512_Z210_Ch1_T1_P1_S1.tif')
SLICE_RAW = (r'D:\Yu_Brain_slices\Slice1_entor_baseline1_2_Decay128_512x512_Z32_Ch1_T1_P1_S1'
             r'\Slice1_entor_baseline1_2_Decay128_512x512_Z32_Ch1_T1_P1_S1.tif')


def cells_of(book):
    return {c.get('id'): c for c in book['cells']}


def src(c):
    return ''.join(c['source'])


def put(c, s):
    c['source'] = s.splitlines(keepends=True)
    if c['cell_type'] == 'code':
        c['outputs'], c['execution_count'] = [], None


def sub(cells, cid, old, new):
    s = src(cells[cid])
    assert s.count(old) == 1, (cid, old[:80], s.count(old))
    put(cells[cid], s.replace(old, new))


def save(book, path):
    with open(path, 'w', encoding='utf-8', newline='') as f:
        f.write(json.dumps(book, indent=indent, ensure_ascii=False) + ('\n' if raw.endswith('\n') else ''))


# ════════════════════════════════════════════════════════════════════════════
# Step 14 (both notebooks): parallel fits
# ════════════════════════════════════════════════════════════════════════════
def parallel_step14(cells):
    sub(cells, 'ad5b1d96', '- **Parallel fitting**: `ThreadPoolExecutor` runs `Fit_decay` concurrently across cells.',
        '- **Parallel fitting**: `fit_decays` runs `Fit_decay` in worker processes (`fit_workers`: `None` = all cores '
        'but one, `1` = sequential); the results are identical to sequential fits.')
    sub(cells, 'b3fb5433', 'from flim_helpers import Fit_decay\n', 'from flim_helpers import Fit_decay, fit_decays\n')
    sub(cells, 'b3fb5433', '''reRunPerCellDecay  = True   # set True to force re-accumulation (overwrites cache)
''', '''reRunPerCellDecay  = True   # set True to force re-accumulation (overwrites cache)
fit_workers        = None   # fitting processes: None = all cores but one, 1 = sequential
''')
    sub(cells, 'b3fb5433', '# STAGE 2 — per-cell Fit_decay  (sequential; cached in csv_out + _fits_path)',
        '# STAGE 2 — per-cell Fit_decay  (parallel processes, fit_workers; cached in csv_out + _fits_path)')
    sub(cells, 'b3fb5433', '''    print(f'Fitting {n_cells} cells  (min_photons={min_photons},  {_t_stop_str})...')
    _t_fit = _time.perf_counter()
''', '''    _fit_ids    = [cid for cid in range(1, n_cells + 1) if int(_cell_mat[cid].sum()) >= min_photons]
    print(f'Fitting {len(_fit_ids)} of {n_cells} cells  (min_photons={min_photons},  {_t_stop_str})...')
    _t_fit = _time.perf_counter()
    _fit_res = dict(zip(_fit_ids, fit_decays(IRF2_fit, _cell_mat[_fit_ids], Tcycle, dt, t_stop_ns=t_stop_ns,
                                             n_workers=fit_workers)))
''')
    sub(cells, 'b3fb5433', '        if n_ph < min_photons:\n', '        if cid not in _fit_res:\n')
    sub(cells, 'b3fb5433', '''            print(f'  cell {cid:3d}  fitting...', end='\\r', flush=True)
            _t2, _a2, _, _model_curve, _chi2, _ = Fit_decay(
                IRF2_fit, decay, Tcycle, dt, t_stop_ns=t_stop_ns)
''', '''            _t2, _a2, _, _model_curve, _chi2 = _fit_res[cid]
''')


# ════════════════════════════════════════════════════════════════════════════
# Time notebook
# ════════════════════════════════════════════════════════════════════════════
tnb = copy.deepcopy(nb)
for c in tnb['cells']:
    if c['cell_type'] == 'code':
        c['outputs'], c['execution_count'] = [], None
tc = cells_of(tnb)

put(tc['md-01'], '''# AnalyzeTCSPC — time-lapse XYZT
TCSPC-FLIM analysis for ISS Vista exports of repeated z-stacks (time-lapse, one position or an Olympus MATL tile grid;
`..._Decay<nbin>_<nx>x<ny>_Z<timepoints·tiles·z>_...tif`).
**Step 0** converts the raw float32 file to uint8 stacks per timepoint (and tile), measures the drift of every
timepoint, and writes on one drift-corrected canvas the time-summed TCSPC volume and one TCSPC volume per timepoint.
Steps 1–16 are the `AnalyzeTCSPC.ipynb` pipeline run on the time sum (segmentation, per-cell fits); **Step 17**
measures every cell at every timepoint with the same masks. Single acquisitions: `AnalyzeTCSPC_XYTZTile.ipynb`.

Run cells **in order**. Edit the **Step 0a** cell for each experiment.''')

sub(tc, 'md-tile0', '## Step 0: Tiled XYTZ acquisition → uint8 tiles → registration → stitched TCSPC\n',
    '''## Step 0: Time-lapse acquisition → uint8 stacks per timepoint → drift correction → stitched TCSPC
Vista blocks are ordered time > tile > z. One position: `tile_grid = (1, 1)` (tile registration is skipped).
''')

_h = src(tc['code-tile0a']).splitlines()[0]
_new_h = '# ── Step 0a: time-lapse acquisition parameters '
sub(tc, 'code-tile0a', _h, _new_h + '─' * max(3, len(_h) - len(_new_h)))
sub(tc, 'code-tile0a', f'# raw_path = r"{TILED_RAW}"\n', '')
sub(tc, 'code-tile0a', 'n_timepoints = 1         # repeated z-stacks (time-lapse) <- Olympus .oir; 1 = no time series\n',
    'n_timepoints = 4         # timepoints (repeated stacks)  <- Olympus .oir\n')
sub(tc, 'code-tile0a', '''      + (f'  (reference t{time_ref:02d}, drift correction {"on" if time_drift else "off"})' if n_timepoints > 1 else ''))
''', '''      + (f'  (reference t{time_ref:02d}, drift correction {"on" if time_drift else "off"})' if n_timepoints > 1 else ''))
if n_timepoints == 1:
    warnings.warn('n_timepoints = 1: not a time series (Step 17 is skipped). Single acquisitions: '
                  'AnalyzeTCSPC_XYTZTile.ipynb', RuntimeWarning)
''')

parallel_step14(tc)
sub(tc, 'md-17', '''- pooled 3D decay → `Fit_decay` (`aw_LT`, `chi2`, `tau1-3`, `amp1-3`), with the IRF, peak alignment, `t_stop` and
  `min_photons` of Step 14;''', '''- pooled 3D decay → `Fit_decay` (`aw_LT`, `chi2`, `tau1-3`, `amp1-3`), with the IRF, peak alignment, `t_stop`,
  `min_photons` and parallel `fit_workers` of Step 14;''')
sub(tc, 'code-17', '''    print(f'Fitting {n_cells} cells x {n_timepoints} timepoints  (min_photons={min_photons}, '
          + (f't_stop={t_stop_ns:.3f} ns)...' if t_stop_ns else 't_stop=None)...'))
    _t0, _recs = _time.perf_counter(), []
''', '''    _ok    = _dec_t[:, 1:].sum(axis=2) >= min_photons               # (timepoint, cell): decays to fit
    print(f'Fitting {int(_ok.sum())} of {_ok.size} decays ({n_cells} cells x {n_timepoints} timepoints, min_photons={min_photons}, '
          + (f't_stop={t_stop_ns:.3f} ns)...' if t_stop_ns else 't_stop=None)...'))
    _t0 = _time.perf_counter()
    _fit_tc = {(int(t), int(c) + 1): r for (t, c), r in
               zip(np.argwhere(_ok), fit_decays(IRF2_fit, _dec_t[:, 1:][_ok], Tcycle, dt, t_stop_ns=t_stop_ns,
                                                n_workers=fit_workers))}
    _recs = []
''')
sub(tc, 'code-17', '''            if _nph >= min_photons:
                _t2, _a2, _, _, _chi2, _ = Fit_decay(IRF2_fit, _dec, Tcycle, dt, t_stop_ns=t_stop_ns)
''', '''            if (_t, cid) in _fit_tc:
                _t2, _a2, _, _, _chi2 = _fit_tc[(_t, cid)]
''')
sub(tc, 'code-17', '''        print(f'  t{_t:02d}  {sum(r["timepoint"] == _t and np.isfinite(r["aw_LT"]) for r in _recs)} cells fitted',
              flush=True)
''', '''        print(f'  t{_t:02d}  {sum(r["timepoint"] == _t and np.isfinite(r["aw_LT"]) for r in _recs)} cells fitted')
''')
save(tnb, TIME)

# ════════════════════════════════════════════════════════════════════════════
# Tile notebook: single acquisition again
# ════════════════════════════════════════════════════════════════════════════
nb['cells'] = [c for c in nb['cells'] if c.get('id') not in ('md-17', 'code-17')]
kc = cells_of(nb)
for cid in ('md-tile0', 'code-tile0a', 'code-tile0b', 'code-tile0c', 'code-tile0d', 'code-tile0e'):
    put(kc[cid], src(before_ts[cid]))

put(kc['md-01'], src(kc['md-01']).replace(
    'Steps 1–16 are the `AnalyzeTCSPC.ipynb` pipeline run on that volume.',
    'Steps 1–16 are the `AnalyzeTCSPC.ipynb` pipeline run on that volume. Time series (repeated z-stacks): '
    '`AnalyzeTCSPC_XYTZTime.ipynb`.', 1))
assert 'XYTZTime' in src(kc['md-01'])

sub(kc, 'code-tile0a', f'''# raw_path = r"{TILED_RAW}"
raw_path = r"{SLICE_RAW}"
''', f'''raw_path = r"{TILED_RAW}"
''')
sub(kc, 'code-tile0a', 'tile_grid    = (1, 1)    #', 'tile_grid    = (3, 2)    #')
sub(kc, 'code-tile0a', 'z_step_um    = 8.0       #', 'z_step_um    = 3.0       #')
sub(kc, 'code-tile0a', '''grid{"" if matl_info else ", default"}): set tile_grid = (1, 1) for a single position')''',
    '''grid{"" if matl_info else ", default"}): set tile_grid = (1, 1) for a single position; '''
    '''a time series goes to AnalyzeTCSPC_XYTZTime.ipynb')''')
sub(kc, 'code-tile0a', '''    oir_ch   = oo.pick_channel(oir_meta, oir_channel, path=_oir0)
''', '''    oir_ch   = oo.pick_channel(oir_meta, oir_channel, path=_oir0)
if oir_meta and (oir_meta['n_t'] > 1 or (n_z_tile > oir_meta['n_z'] and n_z_tile % oir_meta['n_z'] == 0)):
    raise ValueError(f'Time-lapse acquisition (Olympus .oir: {max(oir_meta["n_t"], n_z_tile // oir_meta["n_z"])} '
                     f'timepoints of {oir_meta["n_z"]} planes): use AnalyzeTCSPC_XYTZTime.ipynb')
''')
parallel_step14(kc)
save(nb, TILE)
print('written:', TIME)
print('updated:', TILE)
