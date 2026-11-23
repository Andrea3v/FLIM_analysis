"""Merge the Step 2t segmentation sandbox (2026-11-20) into Steps 0d / 1 / 2 / 3 / 13 / 16 / 17 of
AnalyzeTCSPC_XYTZTime.ipynb, remove the sandbox cells, and add a note to the Tile notebook's Step 2.
Every text replacement is checked (exactly one hit) before anything is written."""
import json
import sys

NB = r'X:\FLIM_2026\AnalyzeTCSPC_XYTZTime.ipynb'
NB_TILE = r'X:\FLIM_2026\AnalyzeTCSPC_XYTZTile.ipynb'
SANDBOX = ['md-2t', 'code-2t-ref', 'code-2t-mask', 'code-2t-cp', 'code-2t-qc']

# ── Step 0d ───────────────────────────────────────────────────────────────────
MD_0D_OLD = """- **Segmentation intensity** `analysis/intensity_coverage/<base>_stitched_intensity.tif` (Step 2): every timepoint and tile scaled to its own
  1–99.5 % range (evens out bleaching), then feather-blended: an average, not a sum."""
MD_0D_NEW = """- **Segmentation reference** `analysis/intensity_coverage/<base>_photonsum.tif` (Step 2): Σ photons ÷ Σ brightness *b* of
  the timepoints × tiles covering each voxel (*b* = mean photons of that timepoint × tile on the tissue all timepoints
  share; tile overlaps feather-blended). Where every timepoint covers a voxel it is the photon sum ÷ a constant:
  brighter timepoints weigh more (e.g. Peredox cpGFP brightening on NADH binding) and nothing is rescaled per
  timepoint. At the drift edges it keeps the same level where a plain sum would dim.
- **Display intensity** `<base>_stitched_intensity.tif`: every timepoint and tile scaled to its own 1–99.5 % range,
  then feather-blended (an average; Step 0e check). Step 2's segmentation input until 2026-11-20."""

C_0D_OLD = """stitched_dir = stitch['stitched_dir']

_fi  = tifffile.imread(stitch['int_path'])
_cov = tifffile.imread(stitch['cov_path'])
_zm  = _fi.shape[0] // 2
fig, axs = plt.subplots(1, 3, figsize=(16, 7))
for _a, _img, _ttl in [(axs[0], _fi[_zm], f'segmentation intensity  z={_zm}'),
                       (axs[1], _fi.max(axis=0), 'segmentation intensity  (max over z)')]:"""
C_0D_NEW = """stitched_dir = stitch['stitched_dir']

# ── segmentation reference (Step 2): Σ photons / Σ brightness of the timepoints x tiles covering each voxel ──
canvas_pos = np.asarray(pos_t if n_timepoints > 1 else [tile_reg['positions']], int)   # (time, tile, zyx) origins
canvas_shp = tuple(int(v) for v in (canvas_t if n_timepoints > 1 else tile_reg['canvas_shape']))
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
                       (axs[1], _fi.max(axis=0), 'segmentation reference  (max over z)')]:"""

C_0D_OLD2 = """print(f'Intensity / coverage: {int_dir}')"""
C_0D_NEW2 = """print(f'Intensity / coverage: {int_dir}  (segmentation reference: {os.path.basename(seg_ref_path)})')
if n_timepoints > 1:
    print('  brightness per timepoint (mean photons on the shared tissue / mean' + (', tile 0' if n_tiles > 1 else '')
          + '): ' + '  '.join(f't{t:02d} {r:.2f}' for t, r in enumerate((seg_b / seg_b.mean(axis=0))[:, 0])))"""

# ── Step 1 ────────────────────────────────────────────────────────────────────
C_1_OLD = """thresh_norm    = 0.55
nlm_h          = None
nlm_patch_size = 5
nlm_patch_dist = 6
nlm_fast       = True
rb_radius      = 30
"""
C_1_NEW = """# ── mask pre-processing (Step 2; tuned 2026-11-20, see Step 2) ──────────────────
thresh_norm    = 0.6           # intensity mask = vol_norm_3d >= thresh_norm (Step 3 adds the cells -> mask3d)
mask_smooth_um = 0.7           # Gaussian sigma in xy (µm) before NLM; 0 = none
nlm_mode       = '2d'          # NLM per plane ('2d'), on the volume ('3d'), or None
nlm_hfac       = 3.0           # NLM strength h = nlm_hfac x estimated noise sigma
nlm_patch_size = 5
nlm_patch_dist = 6
nlm_fast       = True
bg_mode        = 'ellipsoid'   # background: 'ellipsoid' (rolling ball scaled to the data), 'ball', 'gauss', None
bg_um          = 40.0          # background scale (µm): ball radius / Gaussian sigma, >> cell size
bg_height      = 1.0           # 'ellipsoid' height in normalised intensity (data 0..1); lower removes more
# before 2026-11-20: 3D NLM h 0.8 sigma, no smoothing, 'ball' radius 30 px, thresh_norm 0.55
"""

# ── Step 2 ────────────────────────────────────────────────────────────────────
MD_2 = r"""## Step 2: 3D pre-processing, Cellpose segmentation & cell quality filter
**Segmentation tuning (2026-11-20, Slice3 DG/CA3 brain slice, Peredox; sandbox Step 2t merged here)**
- *Reference*: the photon-sum reference of Step 0d replaces the mean of per-timepoint-normalised intensities
  (contrast/noise 28 vs 26, correct level at the drift edges).
- *Mask*: the old rolling ball (radius 30 px on 0..1 data, so 30 units tall) fitted inside 8–10 µm somata and hollowed
  them (mask kept 40% of the cell voxels). Gaussian 0.7 µm → 2D NLM (h 3σ) → ellipsoid rolling ball (40 µm, height =
  data range) → `thresh_norm` 0.6 keeps 57% at the same background. 2D vs 3D NLM made no difference.
- *Cellpose*: `flow_threshold` 0.4 removed nearly every soma. Every mask here has a flow error of 0.75–0.92, so the check
  does not tell good from bad cells; with it off (0), d 10 µm and cellprob −3: 687 cells instead of 64. Input
  normalisation made no difference. With 8 µm z steps most cells span 1–2 planes (stitch IoU ≥ 0.25).
- *Quality filter*: cells no brighter than their surround (contrast < 1.25: faint blobs in background / neuropil, 219 of
  687) are dropped. Cells usable at every timepoint are flagged (74 of 468, limited by the 4-plane z drift and
  ~700 photons per cell and timepoint) and used for Step 17's summary.
- *Pixel mask*: the intensity mask covered a median of ~10% of each new cell and favours bright pixels, which can bias
  tav (cpGFP brightness follows NADH binding): Step 3 adds the cells to it.
- The red channel of these slices is PI (dead cells), brighter than the Peredox mCherry: not used for segmentation.

**What the cell does**
- **Pre-processing** of the photon-sum reference (parameters in Step 1) → `vol_norm_3d` and the intensity mask
  `mask_int` (`vol_norm_3d >= thresh_norm`). Reused when `*_sumTCSPC_3Dfilt_norm.tif` was made with the same settings
  (`*_sumTCSPC_3Dfilt_norm.json`).
- **Cellpose** (`multiday.segment_cellpose`, sizes in µm) on the lightly smoothed reference →
  `*_cellpose_masks_all.tif`, reused only with the same settings (`*_cellpose_masks.json`).
- **Quality filter** (`multiday.cell_quality`): contrast = cell mean / median of a `cell_ring_um` ring around it;
  per timepoint the cell's photons and recorded fraction, counted as in Step 17. Kept cells, relabelled 1..n →
  `cp_masks` (`*_cellpose_masks.tif`, rewritten only when it changes, so the caches of Steps 14–17 stay valid);
  all Cellpose cells (`cp_id`; `cell_id` in `cp_masks`, 0 = dropped) → `*_cellpose_qc.csv`. Figure: middle plane
  (red kept, magenta kept but not usable at every timepoint, yellow dropped, cyan intensity mask), contrast histogram,
  photons per cell and timepoint.
- **Segmentation channel** `seg_source`: `'auto'` = the Olympus channel of Step 0e when present (mCherry), else the FLIM
  photon-sum reference (always for time series: Step 0e skips them).
- **Fine-tuned model** `cp_model_path` (from `Cellpose_FineTune.ipynb`; None = generic `cpsam`): `cp_diameter_um` is then
  taken from the model (its training scale). A Cellpose settings grid on a crop: `md.cellpose_grid`
  (`MultiDay_Reference.ipynb`).
- **`use_subject_reference = True`** skips Cellpose and the filter: the cells come from the subject's curated
  reference (Step 2b), which flags cells its own way."""

C_2 = r"""# ── cell source ───────────────────────────────────────────────────────────────
use_subject_reference = False   # True: cells = the subject's curated reference labels (Step 2b); Cellpose skipped

# ── Cellpose (sizes in µm; mask pre-processing parameters are set in Step 1) ───
seg_source          = 'auto'       # 'auto' = Olympus channel (Step 0e) if present, else 'flim' (photon-sum reference)
cp_mode             = '2d_stitch'  # '2d_stitch' (per plane, linked across z) or '3d' (very slow at small diameters)
cp_diameter_um      = 10.0         # expected object diameter in xy
cp_cellprob_thresh  = -3.0         # lower -> more / larger objects (Cellpose default 0)
cp_flow_threshold   = 0.0          # largest flow error a mask may have: 0 = check off, higher = more lenient (default 0.4)
cp_stitch_threshold = 0.25         # 2D: masks on neighbouring planes are one cell when IoU >= this (best match only)
cp_min_volume_um3   = None         # None = sphere of half the diameter
cp_model_path       = None         # fine-tuned model (Cellpose_FineTune.ipynb); None = generic cpsam
cp_gpu              = True
_rerun_cp           = False        # True: re-run Cellpose even if masks with these settings exist
# before 2026-11-20: d 8 µm, cellprob -1.5, flow 0.4, on the mean of per-timepoint-normalised intensities

# ── quality filter of the Cellpose cells ──────────────────────────────────────
cell_min_contrast = 1.25    # dropped below: cell mean / median of a ring around it (segmentation input)
cell_ring_um      = 3.0     # ring width (µm, in-plane; other cells and unrecorded voxels left out)
cell_min_fov      = 0.9     # a timepoint counts when this fraction of the cell was recorded (Step 17: tc_min_fov)
cell_min_photons  = 500     # photons per cell and timepoint (as min_photons of Steps 14 / 17)
cell_min_tp       = 'all'   # timepoints meeting both for `cells_time_ok` (int or 'all'): a flag, cells are kept

from skimage.segmentation import find_boundaries
from scipy.ndimage import uniform_filter as _ndi_uniform
_voxel_sp = (z_step_um, xy_pixel_um, xy_pixel_um)
if cp_model_path:   # the model was trained at one scale: use its diameter
    _ft_meta = json.load(open(cp_model_path + '.json'))
    if _ft_meta['diameter_um'] != cp_diameter_um:
        print(f'cp_diameter_um {cp_diameter_um} -> {_ft_meta["diameter_um"]} µm (training scale of '
              f'{os.path.basename(cp_model_path)})')
        cp_diameter_um = _ft_meta['diameter_um']
if seg_source == 'auto':
    seg_source = 'olympus' if stitch_oir else 'flim'
if seg_source == 'olympus' and not stitch_oir:
    raise RuntimeError("seg_source='olympus' but Step 0e produced no Olympus channel")
_seg_name = f'Olympus {oir_ch["name"]} ({oir_ch["filter"]})' if seg_source == 'olympus' else 'FLIM photon-sum reference'

_filt_path = os.path.join(newFolder, f'{FName}_sumTCSPC_3Dfilt_norm.tif')
_pre_json  = _filt_path.replace('.tif', '.json')                          # settings of vol_norm_3d
_mask_path = os.path.join(newFolder, f'{FName}_mask3d.tif')               # written by Step 3
_cp_all    = os.path.join(newFolder, f'{FName}_cellpose_masks_all.tif')   # Cellpose output (before the filter)
_cp_path   = os.path.join(newFolder, f'{FName}_cellpose_masks.tif')       # kept cells = cp_masks
_cp_json   = _cp_path.replace('.tif', '.json')                            # Cellpose settings of _cp_all
_cp_qc_csv = os.path.join(newFolder, f'{FName}_cellpose_qc.csv')
_newer = lambda p, *srcs: os.path.exists(p) and all(os.path.getmtime(p) > os.path.getmtime(s) for s in srcs)

# ── volumes: photon-sum reference of Step 0d (vol_sum3d keeps its name for the later steps) ──
vol_sum3d = seg_ref
vol_valid = seg_ntp > 0                       # voxels recorded by at least one timepoint x tile
vol_seg3d = tifffile.imread(stitch_oir).astype(np.float32) if seg_source == 'olympus' else vol_sum3d
if vol_seg3d.shape != vol_sum3d.shape:
    raise ValueError(f'{_seg_name} volume {vol_seg3d.shape} != FLIM reference {vol_sum3d.shape}')
print(f'Segmentation input: {_seg_name}')

# ── mask pre-processing -> vol_norm_3d, intensity mask (Step 3 adds the cells -> mask3d) ──
_pre_settings = dict(source=os.path.basename(seg_ref_path), smooth_um=mask_smooth_um, nlm=nlm_mode, nlm_hfac=nlm_hfac,
                     patch_size=nlm_patch_size, patch_dist=nlm_patch_dist, fast=nlm_fast, bg=bg_mode, bg_um=bg_um,
                     bg_height=bg_height, xy_um=xy_pixel_um)
if (_newer(_filt_path, seg_ref_path) and os.path.exists(_pre_json)
        and json.load(open(_pre_json)) == _pre_settings):
    vol_norm_3d = tifffile.imread(_filt_path).astype(np.float32)
    print(f'Reloaded  vol_norm_3d  {vol_norm_3d.shape}  <- {os.path.basename(_filt_path)}')
else:
    _t0 = datetime.now()
    vol_norm_3d = md.preprocess_volume(vol_sum3d, nlm=nlm_mode, nlm_hfac=nlm_hfac, patch_size=nlm_patch_size,
                                       patch_distance=nlm_patch_dist, fast_mode=nlm_fast, bg=bg_mode,
                                       rb_radius=bg_um / xy_pixel_um, bg_height=bg_height,
                                       smooth_px=(0, mask_smooth_um / xy_pixel_um, mask_smooth_um / xy_pixel_um),
                                       valid=vol_valid)
    tifffile.imwrite(_filt_path, vol_norm_3d)
    with open(_pre_json, 'w') as _f:
        json.dump(_pre_settings, _f, indent=1)
    print(f'Saved: {_filt_path}  ({(datetime.now() - _t0).total_seconds():.0f} s)')
mask_int = vol_norm_3d >= thresh_norm
print(f'Intensity mask: {mask_int.sum():,} voxels  ({100 * mask_int[vol_valid].mean():.1f}% of the recorded volume)  '
      f'thresh_norm={thresh_norm}')

# ── Cellpose + quality filter (skipped when the cells come from the subject reference) ──
cp_masks, n_cells = None, 0
cells_time_ok = None     # cell ids usable at every timepoint (Step 17 summary); None = not judged (Step 2b cells)
df_cellqc     = None
cell_tag   = ''          # file-name tag of the Steps 14-16 outputs (Step 2b sets it for reference cells)
cells_path = _cp_path    # label file the Steps 14-17 caches must be newer than
if use_subject_reference:
    print('Cellpose skipped: cells come from the subject reference (Step 2b).')
else:
    _cp_settings = dict(source=seg_source, channel=oir_ch['id'] if seg_source == 'olympus' else None,
                        input='photonsum' if seg_source == 'flim' else 'olympus', spacing=list(_voxel_sp),
                        mode=cp_mode, diameter_um=cp_diameter_um, cellprob_threshold=cp_cellprob_thresh,
                        flow_threshold=cp_flow_threshold, stitch_threshold=cp_stitch_threshold,
                        min_volume_um3=cp_min_volume_um3,
                        model=[cp_model_path, os.path.getmtime(cp_model_path)] if cp_model_path else None)
    _old = json.load(open(_cp_json)) if os.path.exists(_cp_json) else None
    _seg_src = stitch_oir if seg_source == 'olympus' else seg_ref_path
    if _newer(_cp_all, _seg_src) and _old == _cp_settings and not _rerun_cp:
        _cp_raw = tifffile.imread(_cp_all)
        print(f'Reloaded  Cellpose masks  {_cp_raw.shape}  <- {os.path.basename(_cp_all)}')
    else:
        if os.path.exists(_cp_path) and _old is None:   # masks from before the settings file: keep them
            os.replace(_cp_path, _cp_path.replace('.tif', '_old.tif'))
            print(f'Kept previous masks as {os.path.basename(_cp_path).replace(".tif", "_old.tif")}')
        print(f'Running Cellpose ({cp_mode}, {os.path.basename(cp_model_path) if cp_model_path else "cpsam"}) on '
              f'{_seg_name}  d={cp_diameter_um} µm  cellprob={cp_cellprob_thresh}  flow={cp_flow_threshold}  '
              f'gpu={cp_gpu}...', flush=True)
        _t0 = datetime.now()
        _cp_raw = md.segment_cellpose(md.seg_input(vol_seg3d, _voxel_sp), _voxel_sp, cp_diameter_um, mode=cp_mode,
                                      cellprob_threshold=cp_cellprob_thresh, flow_threshold=cp_flow_threshold,
                                      stitch_threshold=cp_stitch_threshold, min_volume_um3=cp_min_volume_um3,
                                      gpu=cp_gpu, model_path=cp_model_path)
        tifffile.imwrite(_cp_all, _cp_raw)
        with open(_cp_json, 'w') as _f:
            json.dump(_cp_settings, _f, indent=1)
        print(f'Done in {(datetime.now() - _t0).total_seconds():.0f} s.  Saved: {_cp_all}')
    _st = md.mask_stats(_cp_raw, _voxel_sp)
    print(f'Cellpose: {_st["n"]} cells  |  median xy diameter {_st["diam_xy_um"]:.1f} µm  |  '
          f'median z extent {_st["z_extent_um"]:.1f} µm')

    # ── quality filter: drop cells no brighter than their surround, flag cells usable at every timepoint ──
    df_cellqc, _qc_arr = md.cell_quality(_cp_raw, vol_seg3d, tile_int_t, canvas_pos, xy_pixel_um, recorded=vol_valid,
                                         ring_um=cell_ring_um, min_fov=cell_min_fov, min_photons=cell_min_photons)
    _need = n_timepoints if cell_min_tp == 'all' else int(cell_min_tp)
    df_cellqc['keep']    = ~(df_cellqc['contrast'] < cell_min_contrast)   # NaN (no ring: surrounded by cells) kept
    df_cellqc['time_ok'] = df_cellqc['tp_usable'] >= _need
    _keep = df_cellqc.loc[df_cellqc['keep'], 'cell_id'].to_numpy()
    _lut  = np.zeros(int(_cp_raw.max()) + 1, np.uint16 if _keep.size < 65535 else np.uint32)
    _lut[_keep] = np.arange(1, _keep.size + 1)
    cp_masks = _lut[_cp_raw]
    df_cellqc.insert(0, 'cp_id', df_cellqc.pop('cell_id'))
    df_cellqc.insert(0, 'cell_id', _lut[df_cellqc['cp_id'].to_numpy()])  # id in cp_masks, 0 = dropped
    df_cellqc.to_csv(_cp_qc_csv, index=False)
    cells_time_ok = df_cellqc.loc[df_cellqc['keep'] & df_cellqc['time_ok'], 'cell_id'].to_numpy()
    if not (os.path.exists(_cp_path) and np.array_equal(tifffile.imread(_cp_path), cp_masks)):
        tifffile.imwrite(_cp_path, cp_masks)   # only when it changes: its date invalidates the Steps 14-17 caches
        print(f'Saved: {_cp_path}')
    n_cells = int(cp_masks.max())
    print(f'Quality filter: {n_cells} kept, {int((~df_cellqc["keep"]).sum())} dropped (contrast < {cell_min_contrast:g})'
          f'  -> cp_masks, {os.path.basename(_cp_qc_csv)}')
    print(f'  usable at >= {_need}/{n_timepoints} timepoints (>= {cell_min_fov:.0%} recorded, >= {cell_min_photons} '
          f'photons): {cells_time_ok.size} of the kept cells  -> Step 17 summary')

# ── QC figure: middle plane, outlines by verdict over the intensity mask ──────
_zq  = mid_idx
_im  = md.seg_input(vol_seg3d[_zq:_zq + 1], _voxel_sp, sigma_um=(0, 0.7, 0.7))[0]
_sel = _im[vol_valid[_zq]] if vol_valid[_zq].any() else _im.ravel()
_lo, _hi = np.percentile(_sel, 1), np.percentile(_sel, 99.7)
_rgb = np.dstack([np.clip((_im - _lo) / max(_hi - _lo, 1e-12), 0, 1)] * 3)
_rgb[mask_int[_zq]] = 0.6 * _rgb[mask_int[_zq]] + 0.4 * np.array([0.1, 0.8, 0.9])
if df_cellqc is not None:          # Cellpose cells: red kept, magenta kept but not usable at every timepoint, yellow dropped
    _cat = np.zeros(int(_cp_raw.max()) + 1, np.uint8)
    _cat[df_cellqc['cp_id'].to_numpy()] = np.select([~df_cellqc['keep'].to_numpy(), df_cellqc['time_ok'].to_numpy()],
                                                    [3, 1], 2)
    _b = find_boundaries(_cp_raw[_zq], mode='inner')
    for _k, _c in {1: (1.0, 0.25, 0.1), 2: (1.0, 0.2, 1.0), 3: (1.0, 0.9, 0.0)}.items():
        _rgb[_b & (_cat[_cp_raw[_zq]] == _k)] = _c
_cz = min(int(round(200 / xy_pixel_um)), *_rgb.shape[:2])       # 200 µm zoom where the kept cells are densest
_dens = _ndi_uniform((cp_masks[_zq] > 0).astype(float), _cz) if cp_masks is not None else np.zeros(_rgb.shape[:2])
_cy, _cx = np.unravel_index(np.argmax(_dens), _dens.shape) if _dens.any() else (_rgb.shape[0] // 2, _rgb.shape[1] // 2)
_y0 = int(np.clip(_cy - _cz // 2, 0, _rgb.shape[0] - _cz)); _x0 = int(np.clip(_cx - _cz // 2, 0, _rgb.shape[1] - _cz))

_npan = 4 if df_cellqc is not None else 2
fig, axs = plt.subplots(1, _npan, figsize=(6.2 * _npan, 6.4),
                        gridspec_kw=dict(width_ratios=[1, 1, 0.8, 0.8][:_npan]))
axs[0].imshow(_rgb, interpolation='nearest')
axs[0].add_patch(plt.Rectangle((_x0 - 0.5, _y0 - 0.5), _cz, _cz, fill=False, ec='w', lw=1, ls='--'))
axs[0].set_title(f'{_seg_name}  z={_zq}  (cyan: intensity mask)', fontsize=9)
axs[1].imshow(_rgb[_y0:_y0 + _cz, _x0:_x0 + _cz], interpolation='nearest')
axs[1].set_title('red: kept  ·  magenta: kept, not usable at every timepoint  ·  yellow: dropped'
                 if df_cellqc is not None else 'intensity mask (cells from the subject reference: Step 2b)', fontsize=9)
for _a in axs[:2]:
    _a.axis('off')
if df_cellqc is not None:
    axs[2].hist(np.clip(df_cellqc['contrast'].dropna(), 0, 4), bins=60, color='0.5')
    axs[2].axvline(cell_min_contrast, color='r', ls='--')
    axs[2].set(xlabel='cell mean / ring median (clipped at 4)', ylabel='Cellpose cells')
    axs[2].set_title(f'contrast: {int((~df_cellqc["keep"]).sum())} of {len(df_cellqc)} below '
                     f'{cell_min_contrast:g}', fontsize=9)
    _ph = np.where(_qc_arr['fov'] >= cell_min_fov, _qc_arr['photons'], np.nan)[
        :, df_cellqc.loc[df_cellqc['keep'], 'cp_id'].to_numpy() - 1]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        _q = np.nanpercentile(_ph, [10, 50, 90], axis=1)
    _x = np.arange(_ph.shape[0])
    axs[3].fill_between(_x, _q[0], _q[2], color='C0', alpha=0.25, label='10-90 %')
    axs[3].plot(_x, _q[1], 'o-', color='C0', label='median')
    axs[3].axhline(cell_min_photons, color='r', ls='--', label=f'{cell_min_photons}')
    axs[3].set(yscale='log', xlabel='timepoint', ylabel='photons per cell', xticks=_x)
    axs[3].set_title(f'kept cells >= {cell_min_fov:.0%} recorded', fontsize=9); axs[3].legend(fontsize=8)
plt.suptitle(f'{FName}  —  Step 2: segmentation & cell quality  ({n_cells} cells)', fontsize=9)
plt.tight_layout()
plt.show()
"""

# ── Step 3 ────────────────────────────────────────────────────────────────────
MD_3 = """## Step 3: Mask preparation (middle plane)
`mask3d` = intensity mask of Step 2 (`vol_norm_3d >= thresh_norm`) ∪ the cells (Step 2 or 2b): the pixel maps of
Steps 3–13 and the per-cell pixel statistics of Step 16 cover whole cells (a threshold alone keeps only the bright
part of dim cells, and cpGFP brightness follows NADH binding). Saved as `*_mask3d.tif`. Middle plane → `maskTCSPC`
(float, 0/1) and `maskFilt_bool` (boolean)."""
C_3_OLD = """# ── masks from 3D pre-processed volume (middle plane) ────────────────────────
maskTCSPC     = mask3d[mid_idx].astype(float)"""
C_3_NEW = """# ── pixel mask of Steps 3-13: intensity mask (Step 2) ∪ cells (Step 2 / 2b) ────
mask3d = (mask_int | (cp_masks > 0)) if cp_masks is not None else mask_int.copy()
tifffile.imwrite(_mask_path, mask3d.astype(np.uint8))
print(f'mask3d: {100 * mask3d[vol_valid].mean():.1f}% of the recorded volume  (intensity mask '
      f'{100 * mask_int[vol_valid].mean():.1f}%'
      + (f', cells {100 * (cp_masks > 0)[vol_valid].mean():.1f}%)' if cp_masks is not None else ')'))

# ── middle plane ──────────────────────────────────────────────────────────────
maskTCSPC     = mask3d[mid_idx].astype(float)"""

# ── Step 13 ───────────────────────────────────────────────────────────────────
C_13_OLD = """        LTImg_z, _ = FastFLIM_mod_AT0(_png_base, sumZ_norm, tav_filt_z, thresh_norm, limits,
                                       show=(z == mid_idx))"""
C_13_NEW = """        LTImg_z, _ = FastFLIM_mod_AT0(_png_base, sumZ_norm, np.where(maskZ_bool, tav_filt_z, 0.0), 0.0, limits,
                                       show=(z == mid_idx))   # mask3d pixels (dim cell pixels too), brightness-weighted"""
MD_13_OLD = "| `*_mask_stack.tif` | Threshold masks (float32, multi-page) |"
MD_13_NEW = "| `*_mask_stack.tif` | `mask3d` planes: intensity mask ∪ cells (float32, multi-page) |"

# ── Step 16 ───────────────────────────────────────────────────────────────────
MD_16_OLD = "Computes per-cell lifetime statistics from the bilateral-filtered `tav_filt_stack.tif` produced by Step 13."
MD_16_NEW = ("Computes per-cell lifetime statistics from the bilateral-filtered `tav_filt_stack.tif` produced by Step 13. "
             "`mask3d` contains the cells (Step 3), so every pixel of a cell counts, not only the bright ones.")

# ── Step 17 ───────────────────────────────────────────────────────────────────
MD_17_OLD = """1/`n_timepoints` of the Step 14 photons, so its fits are noisier. The summary and the plot use rows with
`fov_frac ≥ tc_min_fov`."""
MD_17_NEW = """1/`n_timepoints` of the Step 14 photons, so its fits are noisier. The summary and the plot use the cells usable at
every timepoint (Step 2 `cells_time_ok`: ≥ `cell_min_fov` recorded and ≥ `cell_min_photons` photons; all cells for
subject-reference cells) and rows with `fov_frac ≥ tc_min_fov` (= `cell_min_fov`). The CSV keeps every cell, with
`cell_time_ok`."""
C_17 = [
    ("tc_min_fov       = 0.9     # summary / plot: rows whose cell has at least this fraction of voxels in the timepoint's FOV",
     "tc_min_fov       = cell_min_fov   # summary / plot: rows whose cell has at least this fraction of voxels in the FOV (Step 2)"),
    ("""    df_tc = df_tc.drop(columns=['G_raw', 'S_raw'])
    df_tc.to_csv(_tc_csv, index=False)""",
     """    df_tc = df_tc.drop(columns=['G_raw', 'S_raw'])
    if cells_time_ok is not None:      # Step 2 quality filter: usable at every timepoint
        df_tc['cell_time_ok'] = df_tc['cell_id'].isin(cells_time_ok)
    df_tc.to_csv(_tc_csv, index=False)"""),
    ("""    _in_fov = df_tc[df_tc['fov_frac'] >= tc_min_fov]""",
     """    _in_fov = df_tc[df_tc['fov_frac'] >= tc_min_fov]
    if cells_time_ok is not None:
        _in_fov = _in_fov[_in_fov['cell_id'].isin(cells_time_ok)]
    _sel_txt = f'fov_frac >= {tc_min_fov}' + (', cells usable at every timepoint' if cells_time_ok is not None else '')"""),
    ("""    print(f'\\nPer timepoint  (rows with fov_frac >= {tc_min_fov}):')""",
     """    print(f'\\nPer timepoint  ({_sel_txt}):')"""),
    ("""({_in_fov["cell_id"].nunique()} cells with '
                 f'fov_frac >= {tc_min_fov}; grey: single cells)'""",
     """({_in_fov["cell_id"].nunique()} cells: '
                 f'{_sel_txt}; grey: single cells)'"""),
]

# ── Tile notebook: note in its Step 2 ─────────────────────────────────────────
TILE_NOTE = """
> **To try here (2026-11-20):** the time-lapse notebook (`AnalyzeTCSPC_XYTZTime.ipynb`, Step 2) now segments a
> brightness-normalised photon sum (`vista_tiles.photon_sum_reference`, n_t = 1 works), builds the mask with 2D NLM +
> an ellipsoid rolling ball (`md.preprocess_volume(nlm='2d', bg='ellipsoid')`; the radius-30 px ball hollows small
> cells), runs Cellpose with `flow_threshold = 0` (0.4 dropped most somata in brain slices: 64 → 687 cells), drops
> low-contrast cells (`md.cell_quality`) and adds the cells to `mask3d`. Worth testing on tile data, especially for
> multi-day tracking: a more complete, consistent reference segmentation means more cells matched across sessions."""


def _src(c):
    return ''.join(c['source'])


def _set(c, s):
    c['source'] = s.splitlines(keepends=True)


def _replace(c, old, new, what):
    s = _src(c)
    n = s.count(old)
    if n != 1:
        sys.exit(f'{what}: expected 1 match in cell {c.get("id")}, found {n} - nothing written')
    _set(c, s.replace(old, new))


def main(nb_path=NB, tile_path=NB_TILE):
    nb = json.load(open(nb_path, encoding='utf-8'))
    cells = nb['cells']
    by = {c.get('id'): c for c in cells}
    for cid in ['md-tile0d', 'code-tile0d', 'code-03', 'c001adff', 'a5874f7b', 'md-04', 'code-04', '87ad2b23',
                '98f90894', '27bf2bbb', 'md-17', 'code-17']:
        if cid not in by:
            sys.exit(f'cell {cid} not found - nothing written')
    _replace(by['md-tile0d'], MD_0D_OLD, MD_0D_NEW, 'Step 0d markdown')
    _replace(by['code-tile0d'], C_0D_OLD, C_0D_NEW, 'Step 0d reference')
    _replace(by['code-tile0d'], C_0D_OLD2, C_0D_NEW2, 'Step 0d print')
    _replace(by['code-03'], C_1_OLD, C_1_NEW, 'Step 1 parameters')
    _set(by['c001adff'], MD_2)
    _set(by['a5874f7b'], C_2)
    _set(by['md-04'], MD_3)
    _replace(by['code-04'], C_3_OLD, C_3_NEW, 'Step 3 mask')
    _replace(by['98f90894'], C_13_OLD, C_13_NEW, 'Step 13 FastFLIM')
    _replace(by['87ad2b23'], MD_13_OLD, MD_13_NEW, 'Step 13 markdown')
    _replace(by['27bf2bbb'], MD_16_OLD, MD_16_NEW, 'Step 16 markdown')
    _replace(by['md-17'], MD_17_OLD, MD_17_NEW, 'Step 17 markdown')
    for k, (old, new) in enumerate(C_17):
        _replace(by['code-17'], old, new, f'Step 17 code #{k}')
    removed = [c.get('id') for c in cells if c.get('id') in SANDBOX]
    nb['cells'] = [c for c in cells if c.get('id') not in SANDBOX]

    tile = json.load(open(tile_path, encoding='utf-8'))
    tc = [c for c in tile['cells'] if c.get('id') == 'c001adff']
    if len(tc) != 1 or not _src(tc[0]).startswith('## Step 2'):
        sys.exit('Tile notebook Step 2 markdown (c001adff) not found - nothing written')
    if 'To try here (2026-11-20)' not in _src(tc[0]):
        _set(tc[0], _src(tc[0]).rstrip('\n') + '\n' + TILE_NOTE)

    for path, obj in [(nb_path, nb), (tile_path, tile)]:
        with open(path, 'w', encoding='utf-8', newline='\n') as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
            f.write('\n')
    print(f'merged into {nb_path}: removed {removed}; Tile note added to {tile_path}')


if __name__ == '__main__':
    main(*sys.argv[1:])
