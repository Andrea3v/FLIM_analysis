"""AnalyzeTCSPC_XYTZTile.ipynb: time series (repeated z-stacks, blocks ordered time > tile > z).
Step 0a n_timepoints (<- Olympus .oir, else manual) + drift settings; 0b converts per timepoint; 0c measures the
drift of every timepoint vs time_ref; 0d stitches the time sum (stitched/, Steps 1-16) and every timepoint
(stitched_T<tt>/) on one canvas; new Step 17: per-cell FLIM per timepoint with the same masks."""
import json

NB = r'X:\FLIM_2026\AnalyzeTCSPC_XYTZTile.ipynb'
raw = open(NB, encoding='utf-8', newline='').read()
indent = len(raw.splitlines()[1]) - len(raw.splitlines()[1].lstrip(' '))
nb = json.loads(raw)
cells = {c.get('id'): c for c in nb['cells']}
assert 'n_timepoints' not in ''.join(cells['code-tile0a']['source']), 'already patched'


def sub(cid, old, new):
    c = cells[cid]
    s = ''.join(c['source'])
    assert s.count(old) == 1, (cid, old[:70], s.count(old))
    c['source'] = s.replace(old, new).splitlines(keepends=True)
    if c['cell_type'] == 'code':
        c['outputs'], c['execution_count'] = [], None


# ── md-tile0 ──────────────────────────────────────────────────────────────────
sub('md-tile0', '''  stitched with the **same tile positions** as 0c → `<base>_stitched_<CH>.tif`, the default segmentation input of Step 2.
''', '''  stitched with the **same tile positions** as 0c → `<base>_stitched_<CH>.tif`, the default segmentation input of Step 2.
- **Time series** (`n_timepoints` > 1: repeated z-stacks; Vista blocks are ordered time > tile > z). `n_timepoints`
  comes from the Olympus `.oir` when present (its time axis, else Vista blocks / (tiles × planes)), otherwise from 0a.
  Tiles are registered on timepoint `time_ref`; the drift of every timepoint vs `time_ref` is measured as an integer
  (dz, dy, dx) (`time_drift`; integer, so photon counts are never resampled; `time_max_dz` bounds z). 0d puts all
  timepoints on one canvas: their **sum** → `stitched/` (Steps 1–16; Step 2 segments the time-summed intensity) and
  **each timepoint** → `stitched_T<tt>/` (Step 17: per-cell FLIM per timepoint with the same masks). The Olympus block
  check and 0e are skipped for time series until a time-lapse `.oir` has been seen.
''')

# ── 0a: n_timepoints (<- .oir, else manual) and drift settings ─────────────────
sub('code-tile0a', '''Tcycle       = 12.5      # laser period in ns (80 MHz)  <- TacTimeRange in the ISS .fbs.xml
''', '''Tcycle       = 12.5      # laser period in ns (80 MHz)  <- TacTimeRange in the ISS .fbs.xml
n_timepoints = 1         # repeated z-stacks (time-lapse) <- Olympus .oir; 1 = no time series
''')
sub('code-tile0a', '''tile_reg_source      = 'auto'       # 'auto' = Olympus PMT channel if its tiles exist, else 'flim'; or 'flim' / 'olympus'
''', '''tile_reg_source      = 'auto'       # 'auto' = Olympus PMT channel if its tiles exist, else 'flim'; or 'flim' / 'olympus'

# ── time series (n_timepoints > 1): Vista blocks ordered time > tile > z ───────
time_ref    = 0          # reference timepoint: tile registration and drift correction (-1 = last)
time_drift  = True       # shift every timepoint onto time_ref by an integer (dz, dy, dx): counts are never resampled
time_max_dz = 4          # max z drift of a timepoint vs time_ref (planes)
''')
sub('code-tile0a', '''n_tiles  = tile_grid[0] * tile_grid[1]
if vista_info['n_blocks'] % n_tiles:
    raise ValueError(f'_Z{vista_info["n_blocks"]} is not a multiple of {n_tiles} tiles ({tile_grid[0]}x{tile_grid[1]} '
                     f'grid{"" if matl_info else ", default"}): set tile_grid = (1, 1) for a single position')
n_z_tile = vista_info['n_blocks'] // n_tiles
print(f'  {vista_info["nbin"]} bins  |  tile {vista_info["ny"]}x{vista_info["nx"]} px  |  '
      f'{vista_info["n_blocks"]} z-blocks = {n_tiles} tiles x {n_z_tile} z')

# voxel size <- Olympus .oir metadata (the ISS .fbs.xml ImageRegion is a placeholder)
oir_meta, oir_ch = None, None
_oir0 = next((f for f in matl_info['files'] if f), None) if matl_info else None
if _oir0:
    oir_meta = oo.oir_metadata(_oir0)
    oir_ch   = oo.pick_channel(oir_meta, oir_channel, path=_oir0)
if oir_meta''', '''n_tiles  = tile_grid[0] * tile_grid[1]

# Olympus .oir metadata: voxel size, planes per tile, timepoints (the ISS .fbs.xml ImageRegion is a placeholder)
oir_meta, oir_ch = None, None
_oir0 = next((f for f in matl_info['files'] if f), None) if matl_info else None
if _oir0:
    oir_meta = oo.oir_metadata(_oir0)
    oir_ch   = oo.pick_channel(oir_meta, oir_channel, path=_oir0)

# timepoints <- Olympus .oir (its time axis, else Vista blocks / (tiles x planes)); the manual value without it
_nb    = vista_info['n_blocks']
_t_src = 'MANUAL (no Olympus .oir)'
if oir_meta:
    _t_oir = oir_meta['n_t'] if oir_meta['n_t'] > 1 else _nb // (n_tiles * oir_meta['n_z'])
    if _t_oir >= 1 and _t_oir * n_tiles * oir_meta['n_z'] == _nb:
        if _t_oir != n_timepoints:
            print(f'n_timepoints {n_timepoints} -> {_t_oir} (Olympus .oir)')
        n_timepoints = _t_oir
        _t_src = ('Olympus .oir time axis' if oir_meta['n_t'] > 1 else
                  f'Olympus .oir ({_nb} Vista blocks / {n_tiles} tiles x {oir_meta["n_z"]} planes)')
    else:
        _t_src = (f'MANUAL (Olympus .oir: {oir_meta["n_t"]} timepoint(s) x {n_tiles} tiles x {oir_meta["n_z"]} planes '
                  f'!= {_nb} Vista blocks)')
if _nb % (n_tiles * n_timepoints):
    raise ValueError(f'_Z{_nb} is not a multiple of {n_timepoints} timepoint(s) x {n_tiles} tiles ({tile_grid[0]}x'
                     f'{tile_grid[1]} grid{"" if matl_info else ", default"}): set tile_grid = (1, 1) for a single '
                     f'position, n_timepoints for a time series')
n_z_tile = _nb // (n_tiles * n_timepoints)
time_ref = time_ref % n_timepoints
print(f'  {vista_info["nbin"]} bins  |  tile {vista_info["ny"]}x{vista_info["nx"]} px  |  {_nb} z-blocks = '
      + (f'{n_timepoints} timepoints x ' if n_timepoints > 1 else '') + f'{n_tiles} tiles x {n_z_tile} z')
print(f'Timepoints   <- {_t_src}:  {n_timepoints}'
      + (f'  (reference t{time_ref:02d}, drift correction {"on" if time_drift else "off"})' if n_timepoints > 1 else ''))

if oir_meta''')

# ── 0b: convert per timepoint ─────────────────────────────────────────────────
sub('code-tile0b', '''conv = vt.convert_to_uint8(raw_path, tiles_dir, n_pos=n_tiles, xy_pixel_um=xy_pixel_um,
                           z_step_um=z_step_um, overwrite=rerun_convert)
tile_int = conv['tile_int']          # (tile, z, y, x) photons summed over bins, file block order
''', '''conv = vt.convert_to_uint8(raw_path, tiles_dir, n_pos=n_tiles, n_t=n_timepoints, xy_pixel_um=xy_pixel_um,
                           z_step_um=z_step_um, overwrite=rerun_convert)
tile_int_t   = conv['tile_int'].reshape(n_timepoints, n_tiles, *conv['tile_int'].shape[1:])   # (time, tile, z, y, x)
tile_paths_t = [conv['tile_paths'][t * n_tiles:(t + 1) * n_tiles] for t in range(n_timepoints)]
tile_int     = tile_int_t[time_ref]   # (tile, z, y, x) photons summed over bins, file block order (reference timepoint)
''')
sub('code-tile0b', '''_empty    = np.flatnonzero(tile_int.reshape(n_tiles * n_z_tile, -1).max(axis=1) == 0)''',
    '''_empty    = np.flatnonzero(tile_int_t.reshape(n_timepoints * n_tiles * n_z_tile, -1).max(axis=1) == 0)''')
sub('code-tile0b', '''if (matl_info and oir_ch and all(matl_info['files'])''',
    '''if (n_timepoints == 1 and matl_info and oir_ch and all(matl_info['files'])''')
sub('code-tile0b', '''elif _empty.size:
    _t0, _z0 = divmod(int(_empty[0]), n_z_tile)
    warnings.warn(f'{_empty.size} of {n_tiles * n_z_tile} empty z-blocks (first: tile {_t0} z {_z0}): Vista recorded '
                  f'fewer planes than the microscope acquired. Blocks after the first dropped plane sit at the wrong '
                  f'tile / z and there are no Olympus tiles to re-assign them: registration and stitching are '
                  f'unreliable.', RuntimeWarning)
''', '''elif _empty.size:
    _tt, _r0 = divmod(int(_empty[0]), n_tiles * n_z_tile)
    _t0, _z0 = divmod(_r0, n_z_tile)
    warnings.warn(f'{_empty.size} of {_nb} empty z-blocks (first: ' + (f't{_tt:02d} ' if n_timepoints > 1 else '')
                  + f'tile {_t0} z {_z0}): Vista recorded fewer planes than the microscope acquired. Blocks after the '
                  f'first dropped plane sit at the wrong tile / z and cannot be re-assigned (that needs the Olympus '
                  f'tiles of a single-timepoint scan): registration and stitching are unreliable.', RuntimeWarning)
if n_timepoints > 1 and matl_info and oir_ch:
    print('Olympus block check skipped: time series (reading a time-lapse .oir is not implemented yet)')
if n_timepoints == 1:
    tile_int_t = tile_int[np.newaxis]   # with the blocks at their true tile / z
''')
sub('code-tile0b', '''_ph = tile_int.sum(axis=(2, 3))
fig, axs = plt.subplots(2, n_tiles, figsize=(max(3.2 * n_tiles, 4.5), 5.2), squeeze=False,
                        gridspec_kw=dict(height_ratios=[3, 1.2]))
for p in range(n_tiles):
    _mp = tile_int[p].max(axis=0)
    axs[0, p].imshow(_mp, cmap='gray', vmax=np.percentile(_mp, 99.5))
    axs[0, p].set_title(f'tile {p}  (max over z)', fontsize=9)
    axs[0, p].axis('off')
    axs[1, p].plot(_ph[p] / 1e3, '.-', lw=0.8)
''', '''_ph   = tile_int_t.sum(axis=(3, 4))     # (time, tile, z)
_tcol = plt.cm.Blues(np.linspace(0.45, 1.0, n_timepoints)) if n_timepoints > 1 else ['C0']
fig, axs = plt.subplots(2, n_tiles, figsize=(max(3.2 * n_tiles, 4.5), 5.2), squeeze=False,
                        gridspec_kw=dict(height_ratios=[3, 1.2]))
for p in range(n_tiles):
    _mp = tile_int[p].max(axis=0)
    axs[0, p].imshow(_mp, cmap='gray', vmax=np.percentile(_mp, 99.5))
    axs[0, p].set_title(f'tile {p}  (max over z' + (f', t{time_ref:02d})' if n_timepoints > 1 else ')'), fontsize=9)
    axs[0, p].axis('off')
    for t in range(n_timepoints):
        _lbl = f't{t:02d}' if n_timepoints > 1 and (n_timepoints <= 8 or t in (0, n_timepoints - 1)) else None
        axs[1, p].plot(_ph[t, p] / 1e3, '.-', lw=0.8, ms=3, color=_tcol[t], label=_lbl)
''')
sub('code-tile0b', '''if block_of_slot is not None:
    axs[1, 0].legend(fontsize=7)
plt.suptitle(f'{vista_info["base"]}  —  tiles in acquisition order'
             + ('  (blocks moved''', '''if block_of_slot is not None or n_timepoints > 1:
    axs[1, 0].legend(fontsize=7)
plt.suptitle(f'{vista_info["base"]}  —  tiles in acquisition order'
             + (f'  |  {n_timepoints} timepoints' if n_timepoints > 1 else '')
             + ('  (blocks moved''')

# ── 0c: drift of every timepoint vs time_ref ──────────────────────────────────
sub('code-tile0c', '''                max_dz=max_dz, source=_reg_src, block_of_slot=_bos)     # any change -> register again
''', '''                max_dz=max_dz, source=_reg_src, block_of_slot=_bos)     # any change -> register again
if n_timepoints > 1:
    _reg_key.update(n_timepoints=n_timepoints, time_ref=time_ref, time_drift=time_drift, time_max_dz=time_max_dz)
''')
sub('code-tile0c', '''        tile_reg['norm_hi'] = [float(np.percentile(t, 99.5)) for t in tile_int]
    vt.save_tile_positions(''', '''        tile_reg['norm_hi'] = [float(np.percentile(t, 99.5)) for t in tile_int]
    if n_timepoints > 1:   # drift of every timepoint vs time_ref (the tiles keep their registered layout)
        print(f'Drift of {n_timepoints} timepoints vs t{time_ref:02d}'
              + ('' if time_drift else '  (measured, not applied: time_drift=False)'))
        tile_reg['time'] = vt.register_timepoints(tile_int_t, ref=time_ref, max_dz=time_max_dz)
    vt.save_tile_positions(''')
sub('code-tile0c', '''print(_df_pairs.to_string(index=False) if len(_df_pairs) else 'Single position: no tile pairs, nothing to register')
''', '''print(_df_pairs.to_string(index=False) if len(_df_pairs) else 'Single position: no tile pairs, nothing to register')
if n_timepoints > 1:   # tile origins of every timepoint on one canvas (Step 0d)
    _tr = tile_reg['time']
    _sh = np.array(_tr['shifts'])
    pos_t, canvas_t = vt.time_positions(tile_reg, _sh if time_drift else np.zeros_like(_sh))
    _um = _sh * np.array([z_step_um, xy_pixel_um, xy_pixel_um])
    print(pd.DataFrame(dict(timepoint=[f't{t:02d}' for t in range(n_timepoints)],
                            drift_zyx=[tuple(int(v) for v in s) for s in _sh],
                            drift_um=[tuple(round(float(v), 1) for v in u) for u in _um],
                            NCC_before=np.round(_tr['ncc0'], 3), NCC_after=np.round(_tr['ncc'], 3))).to_string(index=False))
    print(f'Canvas of all timepoints (z, y, x) = {tuple(canvas_t)}' + ('' if time_drift else '  (drift NOT applied)'))
''')
sub('code-tile0c', '''ax.set_title(f'Registered tiles (max over z)  —  {tile_reg["layout"]}', fontsize=10)
plt.tight_layout()
plt.show()''', '''ax.set_title(f'Registered tiles (max over z)  —  {tile_reg["layout"]}', fontsize=10)
plt.tight_layout()
plt.show()

if n_timepoints > 1:   # QC: drift per timepoint, and the most-drifted timepoint on the reference
    _tw = int(np.argmax(np.abs(_sh).sum(axis=1)))
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.6), gridspec_kw=dict(width_ratios=[1, 1, 1.2]))
    for _k, _ax_name in enumerate(('z', 'y', 'x')):
        axs[0].plot(_um[:, _k], 'o-', lw=2, ms=5, color=f'C{_k}', label=_ax_name)
    axs[0].axhline(0, color='0.75', lw=0.8, zorder=0)
    axs[0].set(xlabel='timepoint', ylabel='drift vs reference (µm)', title=f'Drift vs t{time_ref:02d}')
    axs[1].plot(_tr['ncc0'], 'o-', lw=2, ms=5, color='0.6', label='no correction')
    axs[1].plot(_tr['ncc'], 'o-', lw=2, ms=5, color='C0', label='drift corrected')
    axs[1].set(xlabel='timepoint', ylabel='NCC with the reference', title='Match with the reference', ylim=(0, 1.02))
    for _a in axs[:2]:
        _a.set_xticks(range(n_timepoints))
        _a.spines[['top', 'right']].set_visible(False)
        _a.legend(fontsize=8, frameon=False)
    _ov = np.zeros((2, canvas_t[1], canvas_t[2]), np.float32)
    for _i, _t in enumerate((time_ref, _tw)):
        _m = tile_int_t[_t, 0].max(axis=0)
        _lo, _hi = np.percentile(_m, [1, 99.7])
        _y0, _x0 = pos_t[_t, 0, 1], pos_t[_t, 0, 2]
        _ov[_i, _y0:_y0 + _ny_t, _x0:_x0 + _nx_t] = np.clip((_m - _lo) / max(_hi - _lo, 1e-9), 0, 1)
    axs[2].imshow(np.dstack([_ov[1], _ov[0], _ov[1]]), interpolation='nearest')
    axs[2].set_title(f'tile 0, max over z:  t{time_ref:02d} green, t{_tw:02d} magenta  '
                     f'({"drift corrected" if time_drift else "NOT corrected"})', fontsize=9)
    axs[2].axis('off')
    plt.tight_layout()
    plt.show()''')

# ── 0d: time sum + every timepoint on one canvas ──────────────────────────────
sub('code-tile0d', '''stitch = vt.fuse_tiles(conv['tile_paths'], tile_reg, tile_int, raw_dir, vista_info['base'],
                       overwrite=rerun_fuse, block_of_slot=block_of_slot)
''', '''stitch_t = None    # per-timepoint stacks of a time series (Step 17)
if n_timepoints == 1:
    stitch = vt.fuse_tiles(conv['tile_paths'], tile_reg, tile_int, raw_dir, vista_info['base'],
                           overwrite=rerun_fuse, block_of_slot=block_of_slot)
else:   # one canvas for all timepoints: their sum -> stitched/ (Steps 1-16), timepoint t -> stitched_T<tt>/ (Step 17)
    _ti_flat = tile_int_t.reshape(-1, *tile_int_t.shape[2:])
    _lo = [float(np.percentile(v, 1)) for v in _ti_flat]       # per tile and timepoint (bleaching)
    _hi = [float(np.percentile(v, 99.5)) for v in _ti_flat]
    stitch = vt.fuse_tiles(sum(tile_paths_t, []),
                           dict(tile_reg, positions=pos_t.reshape(-1, 3).tolist(), canvas_shape=canvas_t,
                                norm_lo=_lo, norm_hi=_hi),
                           _ti_flat, raw_dir, vista_info['base'], overwrite=rerun_fuse)
    stitch_t = []
    for t in range(n_timepoints):
        _s = slice(t * n_tiles, (t + 1) * n_tiles)
        stitch_t.append(vt.fuse_tiles(tile_paths_t[t],
                                      dict(tile_reg, positions=pos_t[t].tolist(), canvas_shape=canvas_t,
                                           norm_lo=_lo[_s], norm_hi=_hi[_s]),
                                      tile_int_t[t], raw_dir, f'{vista_info["base"]}_T{t:02d}',
                                      overwrite=rerun_fuse, stitched_name=f'stitched_T{t:02d}'))
''')
sub('code-tile0d', '''axs[2].set_title('tiles per voxel  (max over z)', fontsize=10); axs[2].axis('off')''',
    '''axs[2].set_title(('tiles x timepoints' if n_timepoints > 1 else 'tiles') + ' per voxel  (max over z)', fontsize=10)
axs[2].axis('off')''')
sub('code-tile0d', '''print(f'data_folder for Steps 1-16: {stitched_dir}  ({len(stitch["plane_paths"])} planes)')''',
    '''print(f'data_folder for Steps 1-16: {stitched_dir}  ({len(stitch["plane_paths"])} planes'
      + (f', sum of {n_timepoints} timepoints)' if n_timepoints > 1 else ')'))
if stitch_t:
    print(f'Per timepoint (Step 17): {", ".join(os.path.basename(s["stitched_dir"]) for s in stitch_t)}')
    print('  photons per timepoint: ' + '  '.join(f't{t:02d} {v / 1e6:.2f}M'
                                                  for t, v in enumerate(tile_int_t.sum(axis=(1, 2, 3, 4)))))''')

# ── 0e: no time-lapse .oir yet ────────────────────────────────────────────────
sub('code-tile0e', '''    print('Step 0e skipped: no Olympus multi-area tiles (matl.omp2info + .oir) for this acquisition.')
''', '''    print('Step 0e skipped: no Olympus multi-area tiles (matl.omp2info + .oir) for this acquisition.')
elif n_timepoints > 1:
    print('Step 0e skipped: time series (reading a time-lapse .oir is not implemented yet).')
''')

# ── Step 17 ───────────────────────────────────────────────────────────────────
MD17 = '''## Step 17: Per-cell FLIM per timepoint (time series)
Only for a time series (`n_timepoints` > 1, Step 0a). The masks of Step 2 (or 2b) were segmented on the time-summed
stack, and Step 0 put every timepoint on the same drift-corrected canvas, so cell *i* is the same cell at every
timepoint. For each timepoint stack (`stitched_T<tt>/`), per cell:
- pooled 3D decay → `Fit_decay` (`aw_LT`, `chi2`, `tau1-3`, `amp1-3`), with the IRF, peak alignment, `t_stop` and
  `min_photons` of Step 14;
- `tav_pooled`: mean arrival time of the pooled decay minus the IRF's (the Step 8 definition, per cell);
- `G_phasor`, `S_phasor`: corrected phasor of the pooled decay (as Step 15);
- `n_photons`, and `fov_frac`: fraction of the cell's voxels inside that timepoint's field of view (drift moves cells
  at the edge out).

→ `*_per_cell_timecourse.csv` (one row per cell × timepoint) + `*_per_cell_timecourse.png`. A timepoint has about
1/`n_timepoints` of the Step 14 photons, so its fits are noisier. The summary and the plot use rows with
`fov_frac ≥ tc_min_fov`. Cache: `*_cell_acc_T.npz` (per-timepoint cell decays), rebuilt when the masks or the stacks
are newer, or with `rerun_timecourse = True`.'''

CODE17 = '''# ── Step 17: per-cell FLIM per timepoint (same masks, timepoint stacks of Step 0d) ──
from scipy import sparse
rerun_timecourse = False   # True: re-accumulate the per-timepoint cell decays
tc_min_fov       = 0.9     # summary / plot: rows whose cell has at least this fraction of voxels in the timepoint's FOV

if n_timepoints < 2 or not stitch_t:
    print('Step 17 skipped: single timepoint (n_timepoints = 1 in Step 0a).')
else:
    _tc_acc = os.path.join(newFolder, f'{FName}{cell_tag}_cell_acc_T.npz')
    _tc_csv = os.path.join(newFolder, f'{FName}{cell_tag}_per_cell_timecourse.csv')
    _src_t  = max([os.path.getmtime(cells_path)] + [os.path.getmtime(s['cov_path']) for s in stitch_t])
    _acc_t  = (np.load(_tc_acc) if os.path.exists(_tc_acc) and not rerun_timecourse
               and os.path.getmtime(_tc_acc) > _src_t else None)

    # ── per-timepoint cell decays (cached) ────────────────────────────────────
    if (_acc_t is not None and _acc_t['cell_decays'].shape == (n_timepoints, n_cells + 1, nbin)
            and int(_acc_t['shift_data']) == shift_data):
        _cell_mat_t, _fov_t = _acc_t['cell_decays'], _acc_t['fov_frac']
        print(f'Loaded per-timepoint cell decays <- {os.path.basename(_tc_acc)}')
    else:
        _lab = cp_masks.reshape(cp_masks.shape[0], -1)
        _vox = np.bincount(_lab.ravel(), minlength=n_cells + 1).astype(float)
        _cell_mat_t = np.zeros((n_timepoints, n_cells + 1, nbin))
        _fov_t      = np.zeros((n_timepoints, n_cells + 1))
        print(f'Accumulating {n_cells} cells x {n_timepoints} timepoints  ({_lab.shape[0]} planes each)...')
        _t0 = _time.perf_counter()
        for _t, _st in enumerate(stitch_t):
            if len(_st['plane_paths']) != _lab.shape[0]:
                raise RuntimeError(f't{_t:02d} has {len(_st["plane_paths"])} planes, the masks {_lab.shape[0]}: '
                                   f're-run Step 0d')
            for _z, _p in enumerate(_st['plane_paths']):
                _sel = np.flatnonzero(_lab[_z])
                if _sel.size:
                    _D  = _to_yxt(tifffile.imread(_p), dim_data).reshape(-1, nbin)[_sel].astype(float)
                    _oh = sparse.csr_matrix((np.ones(_sel.size), (_lab[_z, _sel], np.arange(_sel.size))),
                                            shape=(n_cells + 1, _sel.size))
                    _cell_mat_t[_t] += _oh @ np.roll(_D, shift_data, axis=1)
            _in = (tifffile.imread(_st['cov_path']) > 0).ravel().astype(float)
            _fov_t[_t] = np.bincount(_lab.ravel(), weights=_in, minlength=n_cells + 1) / np.maximum(_vox, 1)
            print(f'  t{_t:02d}  {_cell_mat_t[_t, 1:].sum():,.0f} photons in cells', flush=True)
        print(f'Done in {_time.perf_counter() - _t0:.1f} s.')
        np.savez_compressed(_tc_acc, cell_decays=_cell_mat_t, fov_frac=_fov_t, shift_data=shift_data)
        print(f'Saved accumulation cache -> {os.path.basename(_tc_acc)}')

    # ── fits, tav and phasor per cell and timepoint (definitions of Steps 8, 14, 15) ──
    _dec_t = np.roll(_cell_mat_t, _shift_to_target, axis=2)           # peak alignment of Step 14
    _H     = float(np.dot(tau, IRF2_fit) / IRF2_fit.sum())            # IRF mean arrival time
    _tau_p = (np.arange(1, nbin + 1) - 0.5) * dt
    _T_c   = 2.0 * np.pi * (_tau_p - np.dot(_tau_p, IRF2) / IRF2.sum()) / Tcycle
    print(f'Fitting {n_cells} cells x {n_timepoints} timepoints  (min_photons={min_photons}, '
          + (f't_stop={t_stop_ns:.3f} ns)...' if t_stop_ns else 't_stop=None)...'))
    _t0, _recs = _time.perf_counter(), []
    for _t in range(n_timepoints):
        for cid in range(1, n_cells + 1):
            _dec = _dec_t[_t, cid]
            _nph = float(_dec.sum())
            rec  = dict(cell_id=cid, timepoint=_t, fov_frac=round(float(_fov_t[_t, cid]), 3), n_photons=int(_nph),
                        aw_LT=np.nan, chi2=np.nan, tav_pooled=np.nan, G_raw=np.nan, S_raw=np.nan)
            if _nph >= min_photons:
                _t2, _a2, _, _, _chi2, _ = Fit_decay(IRF2_fit, _dec, Tcycle, dt, t_stop_ns=t_stop_ns)
                _t2a, _a2a = np.asarray(_t2).ravel(), np.asarray(_a2).ravel()
                _v = _a2a > 0
                rec.update(aw_LT=round(float(np.dot(_t2a[_v], _a2a[_v]) / _a2a[_v].sum()), 4) if _v.any() else np.nan,
                           chi2=round(float(_chi2), 4), tav_pooled=round(float(np.dot(tau, _dec) / _nph - _H), 4),
                           G_raw=float(np.dot(np.cos(_T_c), _dec) / _nph), S_raw=float(np.dot(np.sin(_T_c), _dec) / _nph))
                for i in range(3):
                    rec[f'tau{i + 1}'] = round(float(_t2a[i]), 4) if i < _t2a.size else np.nan
                    rec[f'amp{i + 1}'] = round(float(_a2a[i]), 4) if i < _a2a.size else np.nan
            _recs.append(rec)
        print(f'  t{_t:02d}  {sum(r["timepoint"] == _t and np.isfinite(r["aw_LT"]) for r in _recs)} cells fitted',
              flush=True)
    df_tc = pd.DataFrame(_recs)
    if G_ref is not None and S_ref is not None:
        _gc, _sc, _, _ = phasor_ref_correction(df_tc['G_raw'].values, df_tc['S_raw'].values, G_ref, S_ref,
                                               tau_ref_ns=0.0, freq_mhz=freq_mhz)
        df_tc['G_phasor'], df_tc['S_phasor'] = np.round(_gc, 5), np.round(_sc, 5)
    else:
        print('G_ref / S_ref not available (Step 11): no phasor columns')
    df_tc = df_tc.drop(columns=['G_raw', 'S_raw'])
    df_tc.to_csv(_tc_csv, index=False)
    print(f'Done in {_time.perf_counter() - _t0:.1f} s.  Saved: {_tc_csv}')

    # ── summary per timepoint ─────────────────────────────────────────────────
    _in_fov = df_tc[df_tc['fov_frac'] >= tc_min_fov]
    _mcols  = ['aw_LT', 'tav_pooled'] + (['G_phasor', 'S_phasor'] if 'G_phasor' in df_tc else [])
    _summ   = _in_fov.groupby('timepoint').agg(cells_fitted=('aw_LT', 'count'), photons_median=('n_photons', 'median'),
                                               **{f'{c}_median': (c, 'median') for c in _mcols})
    print(f'\\nPer timepoint  (rows with fov_frac >= {tc_min_fov}):')
    print(_summ.round(4).to_string())

    # ── plot: single cells (grey) and median + IQR per timepoint ──────────────
    _x = np.arange(n_timepoints)
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4))
    for _a, (_c, _ylab) in zip(axs, [('aw_LT', 'aw_LT (ns)'), ('tav_pooled', 'tav pooled (ns)'),
                                     ('n_photons', 'photons per cell')]):
        _w = _in_fov.pivot(index='cell_id', columns='timepoint', values=_c).reindex(columns=_x).values.astype(float)
        _a.plot(_x, _w.T, color='0.55', lw=0.6, alpha=0.2)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            _q = np.nanpercentile(_w, [25, 50, 75], axis=0)
        _a.fill_between(_x, _q[0], _q[2], color='C0', alpha=0.18, lw=0, label='IQR')
        _a.plot(_x, _q[1], 'o-', color='C0', lw=2, ms=6, label='median')
        _a.set(xlabel='timepoint', ylabel=_ylab, xticks=_x)
        _a.spines[['top', 'right']].set_visible(False)
    axs[2].set_yscale('log')
    axs[0].legend(fontsize=8, frameon=False)
    plt.suptitle(f'{FName}{cell_tag}  —  per-cell FLIM per timepoint  ({_in_fov["cell_id"].nunique()} cells with '
                 f'fov_frac >= {tc_min_fov}; grey: single cells)', fontsize=10)
    plt.tight_layout()
    _fig_tc = os.path.join(newFolder, f'{FName}{cell_tag}_per_cell_timecourse.png')
    plt.savefig(_fig_tc, facecolor='white', bbox_inches='tight', dpi=150)
    plt.show()
    print(f'Saved: {_fig_tc}')'''

_i = next(i for i, c in enumerate(nb['cells']) if c.get('id') == '61a152a1') + 1
assert 'code-17' not in cells
nb['cells'][_i:_i] = [
    dict(cell_type='markdown', id='md-17', metadata={}, source=MD17.splitlines(keepends=True)),
    dict(cell_type='code', id='code-17', metadata={}, execution_count=None, outputs=[],
         source=CODE17.splitlines(keepends=True)),
]

with open(NB, 'w', encoding='utf-8', newline='') as f:
    f.write(json.dumps(nb, indent=indent, ensure_ascii=False) + ('\n' if raw.endswith('\n') else ''))
print('patched md-tile0, code-tile0a..0e; added md-17, code-17')
