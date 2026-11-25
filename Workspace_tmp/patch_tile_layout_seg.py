"""
Port two changes of AnalyzeTCSPC_XYTZTime.ipynb to AnalyzeTCSPC_XYTZTile.ipynb (2026-11-24):

1. Output layout: every Step 0-16 output below <acquisition folder>/analysis/ (tiles_uint8/, intensity_coverage/,
   stitched/, the json files); vt.move_legacy_outputs moves the old flat layout on the next Step 0a run.
   The hard-coded stitched_CH1 paths of MultiDay_Reference.ipynb and Cellpose_FineTune.ipynb follow.
2. Segmentation tuning (Time notebook, 2026-11-20): photon-sum reference (Step 0d), mask pre-processing parameters
   (Step 1), Cellpose d 10 / cellprob -3 / flow 0 + cell quality filter (Step 2), mask3d = intensity mask | cells
   (Step 3), Step 13 RGB on mask3d pixels. Single acquisition: no timepoint flag, a `usable` flag instead.

Cells shared verbatim with the Time notebook are copied from it. Idempotent: skips when already applied.
"""
import json
import os
import re
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else 'X:/FLIM_2026/'     # a folder with copies of the notebooks, for tests
TILE = ROOT + 'AnalyzeTCSPC_XYTZTile.ipynb'
TIME = ROOT + 'AnalyzeTCSPC_XYTZTime.ipynb'
MARK = 'vt.move_legacy_outputs'


def load(path):
    txt = open(path, encoding='utf-8').read()
    indent = len(re.match(r'\{\n( *)"', txt).group(1))
    return json.loads(txt), indent


def save(path, nb, indent):
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(nb, f, ensure_ascii=False, indent=indent)
        f.write('\n')


def cell(nb, cid):
    return next(c for c in nb['cells'] if c.get('id') == cid)


def src(nb, cid):
    return ''.join(cell(nb, cid)['source'])


def set_src(nb, cid, text, clear=True):
    c = cell(nb, cid)
    lines = text.split('\n')
    c['source'] = [l + '\n' for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
    if c['cell_type'] == 'code' and clear:
        c['outputs'] = []
        c['execution_count'] = None


def rep(text, old, new, count=1):
    n = text.count(old)
    if n != count:
        raise ValueError(f'expected {count} match(es), found {n}:\n{old[:300]}')
    return text.replace(old, new)


tile, tile_ind = load(TILE)
time_, _ = load(TIME)
if MARK in src(tile, 'code-tile0a'):
    print('Tile notebook already patched: nothing to do')
    raise SystemExit

# ════════════════════════════════════════════════════════════════════════════════
# 1. output layout
# ════════════════════════════════════════════════════════════════════════════════
s = src(tile, 'code-tile0a')
s = rep(s, '''raw_dir   = os.path.dirname(raw_path)
tiles_dir = os.path.join(raw_dir, 'tiles_uint8')
pos_json  = os.path.join(raw_dir, f'{vista_info["base"]}_tile_positions.json')
print(f'Raw file: {raw_path}  ' + (f'({os.path.getsize(raw_path) / 1e9:.1f} GB)' if raw_exists else
      '(deleted after its uint8 conversion: Step 0b uses the uint8 tiles)'))
''', '''raw_dir   = os.path.dirname(raw_path)
print(f'Raw file: {raw_path}  ' + (f'({os.path.getsize(raw_path) / 1e9:.1f} GB)' if raw_exists else
      '(deleted after its uint8 conversion: Step 0b uses the uint8 tiles)'))

# ── outputs: everything Steps 0-16 write goes below <raw folder>/analysis/ ─────
out_root  = os.path.join(raw_dir, 'analysis')
tiles_dir = os.path.join(out_root, 'tiles_uint8')            # uint8 tile stacks (0b)
int_dir   = os.path.join(out_root, 'intensity_coverage')     # segmentation reference, intensity, coverage (0d, 0e)
pos_json  = os.path.join(out_root, f'{vista_info["base"]}_tile_positions.json')
vt.move_legacy_outputs(raw_dir, vista_info['base'], out_root, int_dir,
                       os.path.join(out_root, 'stitched_timepoints'))   # old flat layout -> analysis/
os.makedirs(int_dir, exist_ok=True)
print(f'Outputs   -> {out_root}')
''')
s = rep(s, "# its name comes from the conversion report in tiles_uint8/ and Step 0b uses the uint8 tiles",
        "# its name comes from the conversion report in analysis/tiles_uint8/ and Step 0b uses the uint8 tiles")
set_src(tile, 'code-tile0a', s)

s = src(tile, 'code-tile0b')
s = rep(s, "    with open(os.path.join(raw_dir, f'{vista_info[\"base\"]}_block_map.json'), 'w') as _f:",
        "    with open(os.path.join(out_root, f'{vista_info[\"base\"]}_block_map.json'), 'w') as _f:")
s = rep(s, '''                  f'unreliable.', RuntimeWarning)

_ph = tile_int.sum(axis=(2, 3))''', '''                  f'unreliable.', RuntimeWarning)
tile_int_t = tile_int[np.newaxis]   # (time = 1, tile, z, y, x), blocks at their true tile / z: Step 0d reference, Step 2 filter

_ph = tile_int.sum(axis=(2, 3))''')
set_src(tile, 'code-tile0b', s)

set_src(tile, 'code-tile0d', '''# ── Step 0d: stitched TCSPC (overlaps summed) + segmentation reference ─────────
stitch = vt.fuse_tiles(conv['tile_paths'], tile_reg, tile_int, out_root, vista_info['base'],
                       overwrite=rerun_fuse, block_of_slot=block_of_slot, int_dir=int_dir)
stitched_dir = stitch['stitched_dir']

# ── segmentation reference (Step 2): Σ photons / Σ brightness of the tiles covering each voxel ──
canvas_pos = np.asarray([tile_reg['positions']], int)            # (time = 1, tile, zyx) canvas origins
canvas_shp = tuple(int(v) for v in tile_reg['canvas_shape'])
seg_ref, seg_ntp, seg_b = vt.photon_sum_reference(tile_int_t, canvas_pos, canvas_shp)
seg_ref_path = os.path.join(int_dir, f'{vista_info["base"]}_photonsum.tif')
if not (os.path.exists(seg_ref_path) and np.array_equal(tifffile.imread(seg_ref_path), seg_ref)):
    tifffile.imwrite(seg_ref_path, seg_ref)   # rewritten only when it changes (the Step 2 caches check its date)

_fi  = seg_ref
_cov = tifffile.imread(stitch['cov_path'])
if _fi.shape != _cov.shape:
    raise ValueError(f'segmentation reference {_fi.shape} != stitched canvas {_cov.shape}: rerun_fuse = True')
_zm  = _fi.shape[0] // 2
fig, axs = plt.subplots(1, 3, figsize=(16, 7))
for _a, _img, _ttl in [(axs[0], _fi[_zm], f'segmentation reference (photon sum / brightness)  z={_zm}'),
                       (axs[1], _fi.max(axis=0), 'segmentation reference  (max over z)')]:
    _a.imshow(_img, cmap='gray', vmin=0, vmax=np.percentile(_img, 99.5))
    _a.set_title(_ttl, fontsize=10); _a.axis('off')
_im = axs[2].imshow(_cov.max(axis=0), cmap='viridis', interpolation='nearest')
axs[2].set_title('tiles per voxel  (max over z)', fontsize=10); axs[2].axis('off')
plt.colorbar(_im, ax=axs[2], fraction=0.046)
plt.tight_layout()
plt.show()
print(f'data_folder for Steps 1-16: {stitched_dir}  ({len(stitch["plane_paths"])} planes)')
print(f'Intensity / coverage: {int_dir}  (segmentation reference: {os.path.basename(seg_ref_path)})')''')

s = src(tile, 'code-tile0e')
s = rep(s, "    stitch_oir = os.path.join(raw_dir, f'{vista_info[\"base\"]}_stitched_{oir_ch[\"name\"]}.tif')",
        "    stitch_oir = os.path.join(int_dir, f'{vista_info[\"base\"]}_stitched_{oir_ch[\"name\"]}.tif')")
set_src(tile, 'code-tile0e', s)

s = src(tile, 'md-tile0')
s = rep(s, "one ImageJ hyperstack per tile (`ZCYX`, decay bins as channels) in `tiles_uint8/`, in the",
        "one ImageJ hyperstack per tile (`ZCYX`, decay bins as channels) in `analysis/tiles_uint8/`, in the")
s = rep(s, '''- **0d** stitching: TCSPC overlaps are **summed** (raw counts, more photons per decay) → `stitched/` (= `data_folder`
  for Steps 1–16). Segmentation intensity is per-tile normalised and feather-blended → `<base>_stitched_intensity.tif`
  (used by Step 2 instead of summing `data_paths`, so overlaps are not double-counted).''',
        '''- **0d** stitching: TCSPC overlaps are **summed** (raw counts, more photons per decay) → `analysis/stitched/`
  (= `data_folder` for Steps 1–16). In `analysis/intensity_coverage/`:
  - `<base>_photonsum.tif`, the FLIM segmentation reference of Step 2 (`vt.photon_sum_reference`, as in the
    time-lapse notebook): Σ photons ÷ Σ brightness *b* of the tiles covering each voxel (*b* = mean photons of the
    tile; overlaps feather-blended). Overlaps are not double-counted and every voxel keeps its photon weighting;
  - `<base>_stitched_intensity.tif`: every tile scaled to its own 1–99.5 % range, then feather-blended (0e check;
    Step 2's input until 2026-11-24);
  - `<base>_coverage.tif`: tiles per voxel.''')
s = rep(s, "stitched with the **same tile positions** as 0c → `<base>_stitched_<CH>.tif`, the default segmentation input of Step 2.",
        "stitched with the **same tile positions** as 0c → `intensity_coverage/<base>_stitched_<CH>.tif`, the default segmentation input of Step 2.")
s = s.rstrip('\n') + '''

**Output layout:** everything goes below `analysis/` in the acquisition folder (the folder keeps only the raw data):
```
analysis/
  <base>_tile_positions.json, <base>_block_map.json
  tiles_uint8/            uint8 tile stacks + conversion report                       (0b)
  intensity_coverage/     <base>_photonsum.tif, <base>_stitched_intensity.tif, <base>_coverage.tif,
                          <base>_stitched_<CH>.tif + .json (Olympus)                  (0d, 0e)
  stitched/               TCSPC planes = data_folder of Steps 1–16                    (0d)
    <name>_<YYYYMMDD_HHMM>_FLIM_Analysis/ results of Steps 1–16
```
Outputs of the old flat layout (loose next to the raw file) are moved here by Step 0a: a rename on the same drive,
so the modification times and every cache stay valid; nothing is overwritten.
'''
set_src(tile, 'md-tile0', s)

set_src(tile, 'md-tile0e', src(time_, 'md-tile0e'))   # differs only in the output path
assert 'analysis/intensity_coverage/<base>_stitched_<CH>.tif' in src(tile, 'md-tile0e')
set_src(tile, 'md-03', src(time_, 'md-03'))

# ════════════════════════════════════════════════════════════════════════════════
# 2. segmentation tuning
# ════════════════════════════════════════════════════════════════════════════════
# ── Step 1: mask pre-processing parameters (block of the Time notebook) ────────
_t3 = src(time_, 'code-03')
_blk = re.search(r'# ── mask pre-processing \(Step 2.*?\n# before 2026-11-20:[^\n]*\n', _t3, re.S).group(0)
_blk = _blk.replace("# before 2026-11-20: 3D NLM h 0.8 sigma, no smoothing, 'ball' radius 30 px, thresh_norm 0.55",
                    "# before 2026-11-24 (here): 3D NLM h 0.8 sigma, no smoothing, 'ball' radius 30 px, thresh_norm 0.55")
assert '2026-11-24 (here)' in _blk
s = src(tile, 'code-03')
s = rep(s, '''
# ── voxel dimensions (µm): xy_pixel_um / z_step_um set in Step 0a ─────────────

thresh_norm    = 0.55
nlm_h          = None
nlm_patch_size = 5
nlm_patch_dist = 6
nlm_fast       = True
rb_radius      = 30
''', '')
s = rep(s, '# ── lifetime per cell (Steps 14, 15, 17)',
        '# ── voxel dimensions (µm): xy_pixel_um / z_step_um set in Step 0a ─────────────\n\n'
        + _blk + '\n# ── lifetime per cell (Steps 14, 15, 17)')
set_src(tile, 'code-03', s)

# ── Step 2 code: the Time cell, single acquisition ─────────────────────────────
s = src(time_, 'a5874f7b')
s = rep(s, "# before 2026-11-20: d 8 µm, cellprob -1.5, flow 0.4, on the mean of per-timepoint-normalised intensities",
        "# before 2026-11-24: d 8 µm, cellprob -1.5, flow 0.4, on the per-tile normalised intensity (<base>_stitched_intensity.tif)")
s = rep(s, '''cell_min_fov      = 0.9     # a timepoint counts when this fraction of the cell was recorded (Step 17: tc_min_fov)
cell_min_photons  = 500     # photons per cell and timepoint (as min_photons of Steps 14 / 17)
cell_min_tp       = 'all'   # timepoints meeting both for `cells_time_ok` (int or 'all'): a flag, cells are kept
''', '''cell_min_fov      = 0.9     # `usable` flag (cells are kept): at least this fraction of the cell recorded ...
cell_min_photons  = 500     # ... and at least this many photons (as min_photons of Step 14)
''')
s = rep(s, "vol_valid = seg_ntp > 0                       # voxels recorded by at least one timepoint x tile",
        "vol_valid = seg_ntp > 0                       # voxels recorded by at least one tile")
s = rep(s, '''cells_time_ok = None     # cell ids usable at every timepoint (Step 17 summary); None = not judged (Step 2b cells)
df_cellqc     = None
cell_tag   = ''          # file-name tag of the Steps 14-16 outputs (Step 2b sets it for reference cells)
cells_path = _cp_path    # label file the Steps 14-17 caches must be newer than''',
        '''df_cellqc  = None        # quality table of the Cellpose cells; None = not judged (Step 2b cells)
cell_tag   = ''          # file-name tag of the Steps 14-16 outputs (Step 2b sets it for reference cells)
cells_path = _cp_path    # label file the Steps 14-16 caches must be newer than''')
s = rep(s, '''        if os.path.exists(_cp_path) and _old is None:   # masks from before the settings file: keep them
            os.replace(_cp_path, _cp_path.replace('.tif', '_old.tif'))
            print(f'Kept previous masks as {os.path.basename(_cp_path).replace(".tif", "_old.tif")}')''',
        '''        _bak = _cp_path.replace('.tif', '_old.tif')
        if os.path.exists(_cp_path) and not os.path.exists(_cp_all) and not os.path.exists(_bak):
            os.replace(_cp_path, _bak)   # masks of the Step 2 before the quality filter: keep them
            print(f'Kept previous masks as {os.path.basename(_bak)}')''')
s = rep(s, '''    # ── quality filter: drop cells no brighter than their surround, flag cells usable at every timepoint ──''',
        '''    # ── quality filter: drop cells no brighter than their surround, flag cells with few photons ──''')
s = rep(s, '''    _need = n_timepoints if cell_min_tp == 'all' else int(cell_min_tp)
    df_cellqc['keep']    = ~(df_cellqc['contrast'] < cell_min_contrast)   # NaN (no ring: surrounded by cells) kept
    df_cellqc['time_ok'] = df_cellqc['tp_usable'] >= _need''',
        '''    df_cellqc['keep']   = ~(df_cellqc['contrast'] < cell_min_contrast)   # NaN (no ring: surrounded by cells) kept
    df_cellqc['usable'] = df_cellqc['tp_usable'] >= 1    # >= cell_min_fov recorded and >= cell_min_photons photons''')
s = rep(s, "    cells_time_ok = df_cellqc.loc[df_cellqc['keep'] & df_cellqc['time_ok'], 'cell_id'].to_numpy()\n", '')
s = rep(s, "        tifffile.imwrite(_cp_path, cp_masks)   # only when it changes: its date invalidates the Steps 14-17 caches",
        "        tifffile.imwrite(_cp_path, cp_masks)   # only when it changes: its date invalidates the Steps 14-16 caches")
s = rep(s, '''    print(f'  usable at >= {_need}/{n_timepoints} timepoints (>= {cell_min_fov:.0%} recorded, >= {cell_min_photons} '
          f'photons): {cells_time_ok.size} of the kept cells  -> Step 17 summary')''',
        '''    print(f'  usable (>= {cell_min_fov:.0%} recorded, >= {cell_min_photons} photons): '
          f'{int((df_cellqc["keep"] & df_cellqc["usable"]).sum())} of the kept cells')''')
s = rep(s, '''if df_cellqc is not None:          # Cellpose cells: red kept, magenta kept but not usable at every timepoint, yellow dropped''',
        '''if df_cellqc is not None:          # Cellpose cells: red kept, magenta kept but not usable, yellow dropped''')
s = rep(s, "np.select([~df_cellqc['keep'].to_numpy(), df_cellqc['time_ok'].to_numpy()],",
        "np.select([~df_cellqc['keep'].to_numpy(), df_cellqc['usable'].to_numpy()],")
s = rep(s, "axs[1].set_title('red: kept  ·  magenta: kept, not usable at every timepoint  ·  yellow: dropped'",
        "axs[1].set_title('red: kept  ·  magenta: kept, few photons / partly recorded  ·  yellow: dropped'")
s = rep(s, '''    _ph = np.where(_qc_arr['fov'] >= cell_min_fov, _qc_arr['photons'], np.nan)[
        :, df_cellqc.loc[df_cellqc['keep'], 'cp_id'].to_numpy() - 1]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        _q = np.nanpercentile(_ph, [10, 50, 90], axis=1)
    _x = np.arange(_ph.shape[0])
    axs[3].fill_between(_x, _q[0], _q[2], color='C0', alpha=0.25, label='10-90 %')
    axs[3].plot(_x, _q[1], 'o-', color='C0', label='median')
    axs[3].axhline(cell_min_photons, color='r', ls='--', label=f'{cell_min_photons}')
    axs[3].set(yscale='log', xlabel='timepoint', ylabel='photons per cell', xticks=_x)
    axs[3].set_title(f'kept cells >= {cell_min_fov:.0%} recorded', fontsize=9); axs[3].legend(fontsize=8)''',
        '''    _ph = _qc_arr['photons'][0, df_cellqc.loc[df_cellqc['keep'], 'cp_id'].to_numpy() - 1]
    _ph = _ph[_ph > 0]
    if _ph.size:
        axs[3].hist(_ph, bins=np.geomspace(_ph.min(), _ph.max() * 1.001, 50), color='C0')
    axs[3].axvline(cell_min_photons, color='r', ls='--', label=f'cell_min_photons {cell_min_photons}')
    axs[3].set(xscale='log', xlabel='photons per cell', ylabel='kept cells')
    axs[3].set_title(f'kept cells: median {np.median(_ph) if _ph.size else 0:.0f} photons', fontsize=9)
    axs[3].legend(fontsize=8)''')
for bad in ('n_timepoints', 'time_ok', 'Step 17', 'timepoint'):
    assert bad not in s, bad
set_src(tile, 'a5874f7b', s)

# ── Step 2 markdown ────────────────────────────────────────────────────────────
set_src(tile, 'c001adff', '''## Step 2: 3D pre-processing, Cellpose segmentation & cell quality filter
**Segmentation settings (2026-11-24: the tuning of the time-lapse notebook, made 2026-11-20 on the Slice3 brain slice)**
- *Reference*: the FLIM photon-sum reference of Step 0d (`<base>_photonsum.tif`) replaces the per-tile normalised,
  feather-blended intensity.
- *Mask*: the old rolling ball (radius 30 px on 0..1 data, so 30 units tall) fitted inside 8–10 µm somata and hollowed
  them (Slice3: the mask kept 40% of the cell voxels). Gaussian 0.7 µm → 2D NLM (h 3σ) → ellipsoid rolling ball (40 µm,
  height = data range) → `thresh_norm` 0.6 kept 57% at the same background. 2D vs 3D NLM made no difference.
- *Cellpose*: `flow_threshold` 0.4 removed nearly every soma on Slice3 (every mask had a flow error of 0.75–0.92, so the
  check does not tell good from bad cells); with it off (0), d 10 µm and cellprob −3: 687 cells instead of 64.
  Before here: d 8 µm, cellprob −1.5, flow 0.4.
- *Quality filter*: cells no brighter than their surround (contrast < 1.25: faint blobs in background / neuropil) are
  dropped.
- *Pixel mask*: the intensity mask covered a median of ~10% of each cell and favours bright pixels, which can bias tav
  (cpGFP brightness follows NADH binding): Step 3 adds the cells to it.
- These values come from a brain slice segmented on the green photon sum: check the figure below on in-vivo tiles
  segmented on the Olympus red channel.

**What the cell does**
- **Pre-processing** of the photon-sum reference (parameters in Step 1) → `vol_norm_3d` and the intensity mask
  `mask_int` (`vol_norm_3d >= thresh_norm`). Reused when `*_sumTCSPC_3Dfilt_norm.tif` was made with the same settings
  (`*_sumTCSPC_3Dfilt_norm.json`).
- **Cellpose** (`multiday.segment_cellpose`, sizes in µm) on the lightly smoothed segmentation channel →
  `*_cellpose_masks_all.tif`, reused only with the same settings (`*_cellpose_masks.json`). Masks of the Step 2 before
  the filter are kept as `*_cellpose_masks_old.tif`.
- **Quality filter** (`multiday.cell_quality`): contrast = cell mean / median of a `cell_ring_um` ring around it; the
  cell's photons and recorded fraction, counted as in the stitched TCSPC. Kept cells, relabelled 1..n → `cp_masks`
  (`*_cellpose_masks.tif`, rewritten only when it changes, so the caches of Steps 14–16 stay valid); all Cellpose cells
  (`cp_id`; `cell_id` in `cp_masks`, 0 = dropped; `usable` = ≥ `cell_min_fov` recorded and ≥ `cell_min_photons`)
  → `*_cellpose_qc.csv`. Figure: middle plane (red kept, magenta kept but few photons / partly recorded, yellow
  dropped, cyan intensity mask), contrast histogram, photons per kept cell.
- **Segmentation channel** `seg_source`: `'auto'` = the Olympus channel of Step 0e when present (mCherry: no diffuse
  green background), else the FLIM photon-sum reference. The mask pre-processing always uses the FLIM reference: it
  defines the photon masks of Steps 3–13.
- **Fine-tuned model** `cp_model_path` (from `Cellpose_FineTune.ipynb`; None = generic `cpsam`): `cp_diameter_um` is then
  taken from the model (its training scale). A Cellpose settings grid on a crop: `md.cellpose_grid`
  (`MultiDay_Reference.ipynb`).
- **`use_subject_reference = True`** skips Cellpose and the filter: the cells come from the subject's curated
  reference (Step 2b), which flags cells its own way.''')

# ── Steps 3, 13, 16: identical to the Time notebook ───────────────────────────
for cid, old in [('md-04', 'Extracts the middle-plane slice of `vol_norm_3d`'),
                 ('code-04', '# ── masks from 3D pre-processed volume (middle plane)'),
                 ('87ad2b23', '| `*_mask_stack.tif` | Threshold masks (float32, multi-page) |'),
                 ('98f90894', 'FastFLIM_mod_AT0(_png_base, sumZ_norm, tav_filt_z, thresh_norm, limits,'),
                 ('27bf2bbb', 'produced by Step 13.\n')]:
    assert old in src(tile, cid) or (cid == '27bf2bbb' and src(tile, cid).endswith('produced by Step 13.')), cid
    set_src(tile, cid, src(time_, cid))

save(TILE, tile, tile_ind)
print('patched', TILE)

# ════════════════════════════════════════════════════════════════════════════════
# 3. notebooks that point at the Tile 336 stitched red channel
# ════════════════════════════════════════════════════════════════════════════════
B = 'XYZTTile1_336_X0.4Y-2.8_Decay128_512x512_Z210_Ch1_T1_P1_S1'
OLD = B + '\\' + B + '_stitched_CH1.tif'
NEW = B + '\\analysis\\intensity_coverage\\' + B + '_stitched_CH1.tif'
for path, code_id, md_id, md_old, md_new in [
        (ROOT + 'MultiDay_Reference.ipynb', 'coder-02', 'mdr-00',
         '(`<base>_stitched_CH1.tif`, complete', '(`analysis/intensity_coverage/<base>_stitched_CH1.tif`, complete'),
        (ROOT + 'Cellpose_FineTune.ipynb', 'ft-03', 'ft-02',
         '`<base>_stitched_CH1.tif`). Crops', '`analysis/intensity_coverage/<base>_stitched_CH1.tif`). Crops')]:
    nb, ind = load(path)
    set_src(nb, code_id, rep(src(nb, code_id), OLD, NEW), clear=False)   # path only: outputs kept
    set_src(nb, md_id, rep(src(nb, md_id), md_old, md_new))
    save(path, nb, ind)
    print('patched', path)
