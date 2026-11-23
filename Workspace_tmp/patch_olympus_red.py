"""Patch AnalyzeTCSPC_XYTZTile.ipynb:
- Step 0a: tile grid / overlap / order from matl.omp2info, voxel size from the Olympus .oir, Tcycle from the
  ISS .fbs.xml TacTimeRange (defaults + message when missing); new raw file (2026-11-06 session)
- Step 0c: layout from the acquisition order
- new Step 0e: Olympus PMT channel (mCherry) stitched with the TCSPC tile positions
- Step 1: data metadata from Step 0a (the .fbs.xml ImageRegion is a placeholder: no voxel-size override,
  no Tcycle override from the IRF XML)
- Step 2: segmentation source (Olympus channel or FLIM intensity)
"""
import json

NB = r'X:\FLIM_2026\AnalyzeTCSPC_XYTZTile.ipynb'
raw = open(NB, encoding='utf-8', newline='').read()
indent = len(raw.splitlines()[1]) - len(raw.splitlines()[1].lstrip(' '))
trailing_nl = raw.endswith('\n')
nb = json.loads(raw)
cells = {c.get('id'): c for c in nb['cells']}
assert 'code-tile0e' not in cells, 'already patched'


def src(cid):
    return ''.join(cells[cid]['source'])


def put(cid, text):
    c = cells[cid]
    c['source'] = text.splitlines(keepends=True)
    if c['cell_type'] == 'code':
        c['outputs'], c['execution_count'] = [], None


def sub(cid, old, new):
    s = src(cid)
    assert s.count(old) == 1, (cid, old[:60], s.count(old))
    put(cid, s.replace(old, new))


def new_cell(cid, kind, text):
    c = dict(cell_type=kind, id=cid, metadata={}, source=text.splitlines(keepends=True))
    if kind == 'code':
        c.update(execution_count=None, outputs=[])
    return c


# ── markdown: intro + Step 0 ──────────────────────────────────────────────────
sub('md-01', 'writes a stitched TCSPC volume (one TYX TIFF per z);',
    'writes a stitched TCSPC volume (one TYX TIFF per z), plus the Olympus PMT channel (e.g. mCherry) '
    'stitched with the same tile positions when its `.oir` tiles are present;')

put('md-tile0', '''## Step 0: Tiled XYTZ acquisition → uint8 tiles → registration → stitched TCSPC
- **0a** parameters & metadata. Bins, tile size and number of z-blocks come from the Vista file name (`_Decay128`,
  `_512x512`, `_Z210`). Read automatically when present, otherwise the defaults in the cell are used (printed):
  - tile grid (max `yIndex`/`xIndex` + 1), overlap and acquisition order ← Olympus `matl.omp2info`
    (searched below the raw file's folder);
  - voxel size (xy pixel, z step) ← the Olympus `.oir` tiles;
  - laser period `Tcycle` ← `TacTimeRange` in the ISS `<name>.fbs.xml` next to the raw file, checked against the
    laser frequency.
- **0b** float32 → uint8: one ImageJ hyperstack per tile (`ZCYX`, decay bins as channels) in `tiles_uint8/`.
  Vista writes classic TIFFs past 4 GB: pages after ~4 GB have no IFD, and their IFDs are written over plane data
  in the first 4 GB. Those pixels are located exactly and zeroed (reported as *corrupted bins*).
- **0c** registration: masked normalised cross-correlation on neighbour overlaps → integer (dz, dy, dx) per tile,
  global least squares. The tile order is the acquisition order from `matl.omp2info`; without it,
  `tile_layout='auto'` scores raster/snake × start corner (and the transposed grid) and `tile_layout_fallback` is
  used when nothing matches or it ties with the best.
- **0d** stitching: TCSPC overlaps are **summed** (raw counts, more photons per decay) → `stitched/` (= `data_folder`
  for Steps 1–16). Segmentation intensity is per-tile normalised and feather-blended → `<base>_stitched_intensity.tif`
  (used by Step 2 instead of summing `data_paths`, so overlaps are not double-counted).
- **0e** Olympus PMT channel (e.g. mCherry, red light sent to the Olympus PMT by the beamsplitter): the `.oir` tiles
  are read, their pixel offset to the TCSPC tiles is measured (same scan, two detectors) and corrected, and they are
  stitched with the **same tile positions** as 0c → `<base>_stitched_<CH>.tif`, the default segmentation input of Step 2.

Outputs are reused when present; set `rerun_convert / rerun_register / rerun_fuse / rerun_oir` to redo a step
(re-registering re-stitches automatically). Tile offsets: `<base>_tile_positions.json`.
''')

# ── Step 0a ───────────────────────────────────────────────────────────────────
put('code-tile0a', '''# ── Step 0a: tiled acquisition parameters ─────────────────────────────────────
import vista_tiles as vt
import olympus_oir as oo
importlib.reload(vt)
importlib.reload(oo)

def _pick_raw():
    root = tk.Tk(); root.withdraw(); root.attributes('-topmost', True)
    p = filedialog.askopenfilename(title='Select ISS Vista XYTZ TIFF',
                                   filetypes=[('TIFF files', '*.tif *.tiff'), ('All files', '*.*')])
    root.destroy()
    return p or None

# ── raw file — swap comments to switch between hardcoded and interactive ──────
raw_path = r"D:\\Peredox_FLIM\\XYZTTile1_336_X0.4Y-2.8_Decay128_512x512_Z210_Ch1_T1_P1_S1\\XYZTTile1_336_X0.4Y-2.8_Decay128_512x512_Z210_Ch1_T1_P1_S1.tif"
# raw_path = _pick_raw()
if not raw_path:
    raise RuntimeError('No raw file selected.')

# ── defaults: used only where the metadata read below are missing ─────────────
tile_grid    = (3, 2)    # (rows, cols)                <- matl.omp2info
tile_overlap = 0.20      # nominal fractional overlap  <- matl.omp2info (the true step is measured)
xy_pixel_um  = 0.99      # XY pixel size in µm          <- Olympus .oir
z_step_um    = 3.0       # Z step in µm                 <- Olympus .oir
Tcycle       = 12.5      # laser period in ns (80 MHz)  <- TacTimeRange in the ISS .fbs.xml

# ── tile registration ─────────────────────────────────────────────────────────
tile_layout          = 'auto'       # 'auto': acquisition order from matl.omp2info, else detected from the
                                    #  overlaps (also tests the transposed grid); or one of vt.LAYOUTS
tile_layout_fallback = 'snake_tl'   # detection only: used when nothing matches or it ties with the best
tile_shape_px        = None         # (ny, nx) per tile; None = from file name '_<nx>x<ny>_'
max_dz               = 2            # max z offset between neighbouring tiles (slices)

# ── Olympus multi-area folder: matl.omp2info + one .oir per tile ──────────────
olympus_dir = 'auto'     # 'auto' = search below the raw file's folder; or a path; None = ignore
oir_channel = 'auto'     # Step 0e channel: 'auto' = red emission filter (else brightest), or a name, e.g. 'CH1'

rerun_convert  = False
rerun_register = False
rerun_fuse     = False

# ── derived ───────────────────────────────────────────────────────────────────
vista_info = vt.parse_vista_name(raw_path)
if tile_shape_px is not None and tuple(tile_shape_px) != (vista_info['ny'], vista_info['nx']):
    raise ValueError(f'tile_shape_px {tile_shape_px} != file name {vista_info["ny"]}x{vista_info["nx"]}')
raw_dir   = os.path.dirname(raw_path)
tiles_dir = os.path.join(raw_dir, 'tiles_uint8')
pos_json  = os.path.join(raw_dir, f'{vista_info["base"]}_tile_positions.json')
print(f'Raw file: {raw_path}  ({os.path.getsize(raw_path) / 1e9:.1f} GB)')

# tile layout <- Olympus multi-area protocol
matl_info = None
if olympus_dir == 'auto':
    olympus_dir = oo.find_matl_dir(raw_dir)
if olympus_dir:
    matl_info = oo.read_matl_info(olympus_dir)
    tile_grid = matl_info['grid']
    tile_overlap = matl_info['overlap'] if matl_info['overlap'] is not None else tile_overlap
    _rc = ' '.join(f'({r},{c})' for r, c in matl_info['grid_rc'])
    print(f'Tile config  <- matl.omp2info  ({os.path.relpath(olympus_dir, raw_dir)})')
    print(f'  grid {tile_grid[0]} rows x {tile_grid[1]} cols  |  overlap {tile_overlap:.0%}  |  '
          f'acquisition order {matl_info["layout"] or "not a standard layout"}  (row,col): {_rc}')
else:
    print(f'Tile config  <- DEFAULTS (no matl.omp2info below the raw folder)')
    print(f'  grid {tile_grid[0]} rows x {tile_grid[1]} cols  |  overlap {tile_overlap:.0%}  |  '
          f'layout {"detected from the overlaps" if tile_layout == "auto" else tile_layout}')
_reg_layout = (matl_info['layout'] if tile_layout == 'auto' and matl_info and matl_info['layout']
               else tile_layout)
n_tiles  = tile_grid[0] * tile_grid[1]
if vista_info['n_blocks'] % n_tiles:
    raise ValueError(f'_Z{vista_info["n_blocks"]} is not a multiple of {n_tiles} tiles')
n_z_tile = vista_info['n_blocks'] // n_tiles
print(f'  {vista_info["nbin"]} bins  |  tile {vista_info["ny"]}x{vista_info["nx"]} px  |  '
      f'{vista_info["n_blocks"]} z-blocks = {n_tiles} tiles x {n_z_tile} z')

# voxel size <- Olympus .oir metadata (the ISS .fbs.xml ImageRegion is a placeholder)
oir_meta, oir_ch = None, None
_oir0 = next((f for f in matl_info['files'] if f), None) if matl_info else None
if _oir0:
    oir_meta = oo.oir_metadata(_oir0)
    oir_ch   = oo.pick_channel(oir_meta, oir_channel, path=_oir0)
if oir_meta and oir_meta['pixel_um'] and oir_meta['z_step_um']:
    xy_pixel_um, z_step_um = float(oir_meta['pixel_um'][1]), float(oir_meta['z_step_um'])
    print(f'Voxel size   <- Olympus .oir:  xy {xy_pixel_um:.4f} µm  |  z {z_step_um:g} µm  '
          f'({oir_meta["n_z"]} planes per tile)')
    if (oir_meta['n_z'], oir_meta['ny'], oir_meta['nx']) != (n_z_tile, vista_info['ny'], vista_info['nx']):
        print(f'  WARNING: Olympus tiles are {oir_meta["n_z"]}x{oir_meta["ny"]}x{oir_meta["nx"]} (z,y,x), '
              f'Vista tiles {n_z_tile}x{vista_info["ny"]}x{vista_info["nx"]} — check the pairing (Step 0e is skipped)')
else:
    print(f'Voxel size   <- DEFAULTS (no Olympus .oir):  xy {xy_pixel_um} µm  |  z {z_step_um} µm')
if oir_ch:
    _others = ', '.join(f'{c["name"]} ({c["filter"]})' for c in oir_meta['channels'] if c['id'] != oir_ch['id'])
    print(f'Olympus channel for Step 0e: {oir_ch["name"]} ({oir_ch["filter"]}, {oir_ch["voltage"]:g} V)'
          + (f'  |  not used: {_others}' if _others else ''))

# laser period <- TacTimeRange in the ISS .fbs.xml
fbs_xml  = vt.find_fbs_xml(raw_path)
fbs_meta = read_fbs_metadata(fbs_xml) if fbs_xml else None
if fbs_meta and fbs_meta['Tcycle']:
    Tcycle = float(fbs_meta['Tcycle'])
    print(f'Laser period <- TacTimeRange in {os.path.basename(fbs_xml)}:  Tcycle {Tcycle:.4f} ns  '
          f'(dt {Tcycle / vista_info["nbin"]:.4f} ns/bin)')
    if fbs_meta['freq_mhz'] and abs(1000.0 / fbs_meta['freq_mhz'] - Tcycle) > 0.01 * Tcycle:
        warnings.warn(f'TacTimeRange {Tcycle:.4f} ns != laser period {1000.0 / fbs_meta["freq_mhz"]:.4f} ns '
                      f'(MacroTimeClockFrequency): check the Vista acquisition settings', RuntimeWarning)
else:
    print(f'Laser period <- DEFAULT Tcycle {Tcycle} ns  '
          f'({"no TacTimeRange in " + os.path.basename(fbs_xml) if fbs_xml else "no .fbs.xml next to the raw file"})')
''')

# ── Step 0c: layout from the acquisition order ────────────────────────────────
sub('code-tile0c', 'layout=tile_layout,', 'layout=_reg_layout,')

# ── Step 0e (new) ─────────────────────────────────────────────────────────────
md_0e = '''### Step 0e: Olympus PMT channel stitched with the TCSPC tile positions
Both detectors record the same scan, so the Olympus tiles take the tile positions of Step 0c. Their pixel offset to the
TCSPC tiles (detector timing, typically ~1 px in x) is measured per tile by phase correlation and its median is
corrected (`oir_align`, linear interpolation). The per-tile NCC with the FLIM intensity checks the tile pairing.
Output: `<base>_stitched_<CH>.tif` (z, y, x; per-tile normalised, feather-blended) + `.json` report.
'''
code_0e = '''# ── Step 0e: Olympus PMT channel (e.g. mCherry) stitched with the same tile positions ──
oir_align = True    # correct the measured Olympus-vs-TCSPC pixel offset (one sub-pixel shift for all tiles)
rerun_oir = False

stitch_oir = None   # stitched Olympus channel; None -> Step 2 segments the FLIM intensity
if not (matl_info and oir_ch):
    print('Step 0e skipped: no Olympus multi-area tiles (matl.omp2info + .oir) for this acquisition.')
elif not all(matl_info['files']):
    print(f'Step 0e skipped: missing .oir tiles {[p for p, f in enumerate(matl_info["files"]) if not f]}')
elif (oir_meta['n_z'], oir_meta['ny'], oir_meta['nx']) != tuple(tile_int.shape[1:]):
    print(f'Step 0e skipped: Olympus tiles {(oir_meta["n_z"], oir_meta["ny"], oir_meta["nx"])} != '
          f'TCSPC tiles {tuple(tile_int.shape[1:])}')
else:
    stitch_oir = os.path.join(raw_dir, f'{vista_info["base"]}_stitched_{oir_ch["name"]}.tif')
    _oir_json  = stitch_oir.replace('.tif', '.json')
    _old = json.load(open(_oir_json)) if os.path.exists(_oir_json) else {}
    if (not rerun_oir and os.path.exists(stitch_oir) and _old.get('channel') == oir_ch['id']
            and _old.get('aligned') == oir_align and os.path.getmtime(stitch_oir) > os.path.getmtime(pos_json)):
        oir_report = _old
        print(f'Reusing {os.path.basename(stitch_oir)}')
    else:
        _oir_tiles = np.stack([oo.read_oir_channel(f, oir_ch['id'], oir_meta['n_z'], oir_meta['ny'], oir_meta['nx'])
                               for f in matl_info['files']])
        oir_report = vt.channel_offset(tile_int, _oir_tiles)
        vt.fuse_intensity(_oir_tiles, tile_reg, stitch_oir, shift_yx=oir_report['offset'] if oir_align else None)
        oir_report.update(channel=oir_ch['id'], channel_name=oir_ch['name'], filter=oir_ch['filter'],
                          aligned=oir_align, files=matl_info['files'],
                          saturated_px=int((_oir_tiles >= 2 ** (oir_meta['bit_depth'] or 16) - 1).sum()))
        with open(_oir_json, 'w') as _f:
            json.dump(oir_report, _f, indent=1)
        del _oir_tiles
        print(f'Saved -> {stitch_oir}')
    _dy, _dx = oir_report['offset']
    print(f'{oir_ch["name"]} vs TCSPC pixel offset (dy, dx) = ({_dy:+.2f}, {_dx:+.2f}) px  '
          f'(MAD {oir_report["spread"][0]:.2f}, {oir_report["spread"][1]:.2f})  -> '
          f'{"corrected" if oir_align else "NOT corrected (oir_align=False)"}  |  saturated px: {oir_report["saturated_px"]}')
    print('  per tile (dy, dx): ' + '  '.join(f'{p}:({a:+.1f},{b:+.1f})' for p, (a, b) in enumerate(oir_report['per_tile'])))
    print('  NCC with the FLIM intensity, before -> after shift: ' +
          '  '.join(f'{p}:{a:.2f}->{b:.2f}' for p, (a, b) in enumerate(zip(oir_report['ncc_before'], oir_report['ncc_after']))))
    if min(oir_report['ncc_after']) < 0.3:
        warnings.warn('An Olympus tile matches its TCSPC tile poorly (NCC < 0.3): check the tile pairing / order',
                      RuntimeWarning)

    # QC: FLIM intensity vs Olympus channel, middle plane + overlay
    _ri = tifffile.imread(stitch_oir)
    _gi = tifffile.imread(stitch['int_path'])
    _zm = _ri.shape[0] // 2
    _n01 = lambda im: np.clip((im - np.percentile(im, 1)) / (np.percentile(im, 99.7) - np.percentile(im, 1)), 0, 1)
    _g, _r = _n01(_gi[_zm]), _n01(_ri[_zm])
    _ov = np.dstack([_r, _g, _r])                      # FLIM green, Olympus magenta, overlap white
    _cy, _cx = np.unravel_index(np.argmax(gaussian_filter(_r, 20)), _r.shape)
    _cy = int(np.clip(_cy, 75, _r.shape[0] - 75)); _cx = int(np.clip(_cx, 75, _r.shape[1] - 75))
    fig, axs = plt.subplots(1, 4, figsize=(22, 6.5 * _r.shape[0] / _r.shape[1] / 1.1))
    for _a, _img, _ttl in [(axs[0], _g, f'FLIM intensity  z={_zm}'),
                           (axs[1], _r, f'Olympus {oir_ch["name"]} ({oir_ch["filter"]})  z={_zm}'),
                           (axs[2], _ov, 'overlay: FLIM green / Olympus magenta'),
                           (axs[3], _ov[_cy - 75:_cy + 75, _cx - 75:_cx + 75], f'overlay zoom (150 px at y={_cy}, x={_cx})')]:
        _a.imshow(_img, cmap='gray', interpolation='nearest')
        _a.set_title(_ttl, fontsize=10); _a.axis('off')
    axs[2].add_patch(plt.Rectangle((_cx - 75, _cy - 75), 150, 150, fill=False, ec='y', lw=1))
    plt.tight_layout()
    plt.show()
'''
i = next(k for k, c in enumerate(nb['cells']) if c.get('id') == 'code-tile0d')
nb['cells'][i + 1:i + 1] = [new_cell('md-tile0e', 'markdown', md_0e), new_cell('code-tile0e', 'code', code_0e)]

# ── Step 1: metadata from Step 0a ─────────────────────────────────────────────
sub('code-03', '# Tcycle: set in Step 0a (overridden if a .fbs.xml is found next to the data)',
    '# Tcycle, xy_pixel_um, z_step_um: set in Step 0a (ISS .fbs.xml / Olympus .oir metadata, or the defaults)')
sub('code-03', '''# ── load metadata from middle plane ──────────────────────────────────────────
meta     = None
xml_data = _find_xml(data_paths[mid_idx])
if xml_data:
    meta   = read_fbs_metadata(xml_data)
    Tcycle = meta['Tcycle']
    print(f'\\nData metadata: {os.path.basename(xml_data)}')
    print(f'  Tcycle={Tcycle:.4f} ns  nbin={meta["nbin"]}  '
          f'pixel={meta["pixel_size_um"]:.4f} µm  '
          f'gain={meta["detector_gain"]}%  dwell={meta["dwell_time_ms"]} ms/px  '
          f'excitation={meta["excitation_nm"]} nm')
    if meta['pixel_size_um'] is not None and meta['pixel_size_um'] > 0:
        xy_pixel_um = float(meta['pixel_size_um'])
        print(f'  xy_pixel_um updated from XML: {xy_pixel_um:.4f} µm  '
              f'(z_step_um={z_step_um} µm — set manually above)')
else:
    warnings.warn('No .fbs.xml found for data — using manual Tcycle.', RuntimeWarning, stacklevel=2)
    print(f'\\n  No XML found — using manual Tcycle = {Tcycle} ns')
''', '''# ── data metadata: the ISS .fbs.xml found in Step 0a (Tcycle is set there) ────
# Its ImageRegion is a Vista placeholder (±50 µm), so the voxel size stays the Step 0a one.
meta = fbs_meta
if meta:
    print(f'\\nData metadata: {os.path.basename(fbs_xml)}')
    print(f'  Tcycle={Tcycle:.4f} ns  nbin={meta["nbin"]}  '
          f'gain={meta["detector_gain"]}%  dwell={meta["dwell_time_ms"]} ms/px  '
          f'excitation={meta["excitation_nm"]} nm')
else:
    print(f'\\n  No .fbs.xml for the data — Tcycle = {Tcycle} ns (Step 0a default)')
''')
sub('code-03', '''elif meta_irf and not meta:
    Tcycle = meta_irf['Tcycle']
    print(f'Using Tcycle from IRF XML: {Tcycle:.4f} ns')
''', '''elif meta_irf and not meta:
    print(f'IRF XML period {meta_irf["Tcycle"]:.4f} ns  (data: Step 0a default {Tcycle} ns, not overridden)')
''')

# ── Step 2: segmentation source ───────────────────────────────────────────────
sub('c001adff', '''- **`use_subject_reference = True`** skips Cellpose''',
    '''- **Segmentation channel** `seg_source`: `'auto'` = the Olympus channel stitched in Step 0e when present (mCherry:
  no diffuse green background), else the time-projected FLIM intensity. The pre-processing (`vol_norm_3d`, `mask3d`)
  stays on the FLIM intensity: it defines the photon masks of Steps 3–13.
- **`use_subject_reference = True`** skips Cellpose''')
sub('a5874f7b', '''# ── Cellpose parameters (sizes in µm; pre-processing parameters are set in Step 1) ──
cp_input            = 'smoothed'   # 'smoothed' raw intensity, or 'nlm' (NLM + rolling ball, the pre-2026-11 input)''',
    '''# ── Cellpose parameters (sizes in µm; pre-processing parameters are set in Step 1) ──
seg_source          = 'auto'       # 'auto' = Olympus channel (Step 0e) if present, else 'flim'; or 'olympus' / 'flim'
cp_input            = 'smoothed'   # 'smoothed' raw intensity, or 'nlm' (FLIM NLM + rolling ball, the pre-2026-11 input)''')
sub('a5874f7b', '''_voxel_sp    = (z_step_um, xy_pixel_um, xy_pixel_um)
_cp_settings = dict(input=cp_input, mode=cp_mode, diameter_um=cp_diameter_um,''',
    '''_voxel_sp    = (z_step_um, xy_pixel_um, xy_pixel_um)
if seg_source == 'auto':
    seg_source = 'olympus' if stitch_oir else 'flim'
if seg_source == 'olympus' and not stitch_oir:
    raise RuntimeError("seg_source='olympus' but Step 0e produced no Olympus channel")
if seg_source == 'olympus' and cp_input == 'nlm':
    raise ValueError("cp_input='nlm' is the FLIM pre-processing; use cp_input='smoothed' with seg_source='olympus'")
_seg_name    = f'Olympus {oir_ch["name"]} ({oir_ch["filter"]})' if seg_source == 'olympus' else 'FLIM intensity'
_cp_settings = dict(source=seg_source, channel=oir_ch['id'] if seg_source == 'olympus' else None,
                    spacing=list(_voxel_sp), input=cp_input, mode=cp_mode, diameter_um=cp_diameter_um,''')
sub('a5874f7b', '''vol_sum3d = tifffile.imread(stitch['int_path']).astype(np.float32)
''', '''vol_sum3d = tifffile.imread(stitch['int_path']).astype(np.float32)
vol_seg3d = tifffile.imread(stitch_oir).astype(np.float32) if seg_source == 'olympus' else vol_sum3d
if vol_seg3d.shape != vol_sum3d.shape:
    raise ValueError(f'{_seg_name} volume {vol_seg3d.shape} != FLIM intensity {vol_sum3d.shape}')
print(f'Segmentation input: {_seg_name}')
''')
sub('a5874f7b', "_cp_in = vol_norm_3d if cp_input == 'nlm' else md.seg_input(vol_sum3d, _voxel_sp)",
    "_cp_in = vol_norm_3d if cp_input == 'nlm' else md.seg_input(vol_seg3d, _voxel_sp)")
sub('a5874f7b', '''        print(f'Running Cellpose ({cp_mode})  d={cp_diameter_um} µm  cellprob={cp_cellprob_thresh}  '
              f'input={cp_input}  gpu={cp_gpu}...', flush=True)''',
    '''        print(f'Running Cellpose ({cp_mode}) on {_seg_name}  d={cp_diameter_um} µm  cellprob={cp_cellprob_thresh}  '
              f'input={cp_input}  gpu={cp_gpu}...', flush=True)''')
sub('a5874f7b', '''_raw_mid = md.seg_input(vol_sum3d[mid_idx:mid_idx + 1], _voxel_sp, sigma_um=(0, 0.7, 0.7))[0]
_raw_mid = np.clip(_raw_mid / np.percentile(_raw_mid, 99.7), 0, 1)
_panels  = [(_raw_mid, f'stitched intensity (σ 0.7 µm)  z={mid_idx}'),
            (vol_norm_3d[mid_idx], f'NLM + rolling ball  z={mid_idx}'),
            (mask3d[mid_idx].astype(float), f'mask  (thresh_norm={thresh_norm})')]
if cp_masks is not None:
    _rgb = np.dstack([_raw_mid] * 3)
    _rgb[find_boundaries(cp_masks[mid_idx], mode='inner')] = (1.0, 0.25, 0.1)
    _panels.append((_rgb, f'Cellpose outlines  z={mid_idx}  ({n_cells} cells)'))
_fig_pre, _axes_pre = plt.subplots(1, len(_panels), figsize=(6 * len(_panels), 6 * vol_sum3d.shape[1] / vol_sum3d.shape[2] / 1.2))
for _a, (_img, _ttl) in zip(_axes_pre, _panels):
    _a.imshow(_img, cmap='gray' if _img.ndim == 2 else None, interpolation='nearest')
    _a.set_title(_ttl, fontsize=10)
    _a.axis('off')
''', '''def _mid01(v):   # middle plane, lightly smoothed, scaled to [0, 1]
    m = md.seg_input(v[mid_idx:mid_idx + 1], _voxel_sp, sigma_um=(0, 0.7, 0.7))[0]
    return np.clip(m / np.percentile(m, 99.7), 0, 1)
_flim_mid = _mid01(vol_sum3d)
_seg_mid  = _mid01(vol_seg3d) if seg_source == 'olympus' else _flim_mid
_panels   = [(_flim_mid, f'FLIM intensity (σ 0.7 µm)  z={mid_idx}')]
if seg_source == 'olympus':
    _panels.append((_seg_mid, f'{_seg_name} (σ 0.7 µm)  z={mid_idx}'))
_panels  += [(vol_norm_3d[mid_idx], f'NLM + rolling ball  z={mid_idx}'),
             (mask3d[mid_idx].astype(float), f'mask  (thresh_norm={thresh_norm})')]
if cp_masks is not None:
    _bnd = find_boundaries(cp_masks[mid_idx], mode='inner')
    for _im, _nm in [(_seg_mid, _seg_name)] + ([(_flim_mid, 'FLIM intensity')] if seg_source == 'olympus' else []):
        _rgb = np.dstack([_im] * 3)
        _rgb[_bnd] = (1.0, 0.25, 0.1)
        _panels.append((_rgb, f'Cellpose outlines on {_nm}  ({n_cells} cells)'))
_ncol = min(3, len(_panels))
_nrow = -(-len(_panels) // _ncol)
_fig_pre, _axes_pre = plt.subplots(_nrow, _ncol, squeeze=False,
                                   figsize=(6 * _ncol, 6 * _nrow * vol_sum3d.shape[1] / vol_sum3d.shape[2]))
for _a in _axes_pre.ravel():
    _a.axis('off')
for _a, (_img, _ttl) in zip(_axes_pre.ravel(), _panels):
    _a.imshow(_img, cmap='gray' if _img.ndim == 2 else None, interpolation='nearest')
    _a.set_title(_ttl, fontsize=10)
''')

with open(NB, 'w', encoding='utf-8', newline='') as f:
    f.write(json.dumps(nb, indent=indent, ensure_ascii=False) + ('\n' if trailing_nl else ''))
print('patched:', len(nb['cells']), 'cells')
