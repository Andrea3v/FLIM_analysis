"""Multi-day tracking: Step 2 (Cellpose in µm, multiday.py) + Step 2b (subject reference -> session labels)
in AnalyzeTCSPC_XYTZTile.ipynb, cell_tag in the Steps 14-16 output names, and the new
MultiDay_Reference.ipynb (reference segmentation, napari curation, collection across sessions)."""
import json

TILE = r'X:\FLIM_2026\AnalyzeTCSPC_XYTZTile.ipynb'
REF = r'X:\FLIM_2026\MultiDay_Reference.ipynb'


def lines(s):
    out = s.split('\n')
    return [x + '\n' for x in out[:-1]] + ([out[-1]] if out[-1] else [])


def md(cid, text):
    return {'cell_type': 'markdown', 'id': cid, 'metadata': {}, 'source': lines(text)}


def code(cid, text):
    return {'cell_type': 'code', 'execution_count': None, 'id': cid, 'metadata': {},
            'outputs': [], 'source': lines(text)}


def cell(nb, cid):
    return next(c for c in nb['cells'] if c.get('id') == cid)


def set_source(c, text):
    c['source'] = lines(text)
    if c['cell_type'] == 'code':
        c['outputs'], c['execution_count'] = [], None


def replace(c, old, new):
    s = ''.join(c['source'])
    assert s.count(old) == 1, f'{c.get("id")}: expected exactly one match for {old[:60]!r}'
    set_source(c, s.replace(old, new))


def dump(nb, path, indent, trailing_nl):
    with open(path, 'w', encoding='utf8', newline='') as f:
        f.write(json.dumps(nb, indent=indent, ensure_ascii=False) + ('\n' if trailing_nl else ''))


raw = open(TILE, encoding='utf8', newline='').read()
indent = 1 if raw.startswith('{\n "') else 2
nb = json.loads(raw)
assert not any(c.get('id') == 'code-2b' for c in nb['cells']), 'already patched'

# ── imports ───────────────────────────────────────────────────────────────────
c01 = cell(nb, 'code-01')
replace(c01, 'import csv\n', 'import csv\nimport json\n')
replace(c01, 'from flim_helpers import uplowthresh, Fit_decay, Calc_Phasor_Mat, phasor_ref_correction, read_fbs_metadata\n',
        'from flim_helpers import uplowthresh, Fit_decay, Calc_Phasor_Mat, phasor_ref_correction, read_fbs_metadata\n'
        'import multiday as md\n')

# ── Step 2: pre-processing + Cellpose (sizes in µm) ───────────────────────────
set_source(cell(nb, 'c001adff'), """## Step 2: 3D pre-processing & Cellpose segmentation
- **Pre-processing** (NLM + rolling ball on the stitched segmentation intensity) → `vol_norm_3d`, `mask3d`
  (pixel masks of Steps 3–13). Reused when `*_sumTCSPC_3Dfilt_norm.tif` / `*_mask3d.tif` exist in `newFolder`.
- **Cellpose** (`multiday.segment_cellpose`, sizes in µm) → `cp_masks`. Masks are reused only if they were made
  with the same settings (`*_cellpose_masks.json`). Default input is the lightly smoothed raw intensity: the NLM
  output has a fibrous texture that Cellpose segments as cells. Tune the settings on a crop with the grid in
  `MultiDay_Reference.ipynb`.
- **`use_subject_reference = True`** skips Cellpose: the cells come from the subject's curated reference (Step 2b).""")

set_source(cell(nb, 'a5874f7b'), r"""# ── cell source ───────────────────────────────────────────────────────────────
use_subject_reference = False   # True: cells = the subject's curated reference labels (Step 2b); Cellpose skipped

# ── Cellpose parameters (sizes in µm; pre-processing parameters are set in Step 1) ──
cp_input            = 'smoothed'   # 'smoothed' raw intensity, or 'nlm' (NLM + rolling ball, the pre-2026-11 input)
cp_mode             = '2d_stitch'  # '2d_stitch' (per plane, linked across z) or '3d' (very slow at small diameters)
cp_diameter_um      = 8.0          # expected object diameter in xy
cp_cellprob_thresh  = -1.5         # lower -> more / larger objects (Cellpose default 0)
cp_flow_threshold   = 0.4          # 2D only (Cellpose ignores it in 3D); raise to accept more objects
cp_stitch_threshold = 0.25         # 2D: IoU needed to link masks of neighbouring planes
cp_min_volume_um3   = None         # None = sphere of half the diameter
cp_gpu              = True
_rerun_cp           = False        # True: re-run Cellpose even if masks with these settings exist
# previous settings, for reference: cp_input='nlm', cp_mode='3d', cp_diameter_um=49.5, cp_cellprob_thresh=0.9,
# cp_min_volume_um3=1470 (=500 voxels) -> masks ~42 µm tall that follow the NLM texture

_voxel_sp    = (z_step_um, xy_pixel_um, xy_pixel_um)
_cp_settings = dict(input=cp_input, mode=cp_mode, diameter_um=cp_diameter_um,
                    cellprob_threshold=cp_cellprob_thresh, flow_threshold=cp_flow_threshold,
                    stitch_threshold=cp_stitch_threshold, min_volume_um3=cp_min_volume_um3)

_filt_path = os.path.join(newFolder, f'{FName}_sumTCSPC_3Dfilt_norm.tif')
_mask_path = os.path.join(newFolder, f'{FName}_mask3d.tif')
_cp_path   = os.path.join(newFolder, f'{FName}_cellpose_masks.tif')
_cp_json   = _cp_path.replace('.tif', '.json')        # settings the masks were made with

# ── 3D intensity: per-tile normalised, feather-blended (Step 0d) ──────────────
# (summing data_paths would double-count photons in the tile overlaps)
vol_sum3d = tifffile.imread(stitch['int_path']).astype(np.float32)

# ── pre-processing: NLM + rolling ball -> vol_norm_3d, mask3d ─────────────────
if os.path.exists(_filt_path) and os.path.exists(_mask_path):
    vol_norm_3d = tifffile.imread(_filt_path).astype(np.float32)
    mask3d      = tifffile.imread(_mask_path).astype(bool)
    print(f'Reloaded  vol_norm_3d / mask3d  {vol_norm_3d.shape}  <- {os.path.basename(_filt_path)}')
else:
    print(f'Loaded stitched intensity {vol_sum3d.shape}  <- {os.path.basename(stitch["int_path"])}')
    vol_norm_3d = md.preprocess_volume(vol_sum3d, nlm_h=nlm_h, patch_size=nlm_patch_size,
                                       patch_distance=nlm_patch_dist, fast_mode=nlm_fast, rb_radius=rb_radius)
    mask3d = vol_norm_3d >= thresh_norm
    tifffile.imwrite(_filt_path, vol_norm_3d)
    tifffile.imwrite(_mask_path, mask3d.astype(np.uint8))
    print(f'Saved: {_filt_path}')
print(f'Valid voxels: {mask3d.sum():,} / {mask3d.size:,}  ({100*mask3d.mean():.1f}%)  thresh_norm={thresh_norm}')

# ── Cellpose (skipped when the cells come from the subject reference) ────────
cp_masks, n_cells = None, 0
cell_tag   = ''          # file-name tag of the Steps 14-16 outputs (Step 2b sets it for reference cells)
cells_path = _cp_path    # label file the Steps 14-16 caches must be newer than
if use_subject_reference:
    print('Cellpose skipped: cells come from the subject reference (Step 2b).')
else:
    _old = json.load(open(_cp_json)) if os.path.exists(_cp_json) else None
    if os.path.exists(_cp_path) and _old == _cp_settings and not _rerun_cp:
        cp_masks = tifffile.imread(_cp_path)
        print(f'Reloaded  cp_masks  {cp_masks.shape}  <- {os.path.basename(_cp_path)}')
    else:
        if os.path.exists(_cp_path) and _old is None:   # masks from before the settings file: keep them
            os.replace(_cp_path, _cp_path.replace('.tif', '_old.tif'))
            print(f'Kept previous masks as {os.path.basename(_cp_path).replace(".tif", "_old.tif")}')
        _cp_in = vol_norm_3d if cp_input == 'nlm' else md.seg_input(vol_sum3d, _voxel_sp)
        print(f'Running Cellpose ({cp_mode})  d={cp_diameter_um} µm  cellprob={cp_cellprob_thresh}  '
              f'input={cp_input}  gpu={cp_gpu}...', flush=True)
        _t0 = datetime.now()
        cp_masks = md.segment_cellpose(_cp_in, _voxel_sp, cp_diameter_um, mode=cp_mode,
                                       cellprob_threshold=cp_cellprob_thresh, flow_threshold=cp_flow_threshold,
                                       stitch_threshold=cp_stitch_threshold, min_volume_um3=cp_min_volume_um3,
                                       gpu=cp_gpu)
        tifffile.imwrite(_cp_path, cp_masks)
        with open(_cp_json, 'w') as _f:
            json.dump(_cp_settings, _f, indent=1)
        print(f'Done in {(datetime.now() - _t0).total_seconds():.0f} s.  Saved: {_cp_path}')
    n_cells = int(cp_masks.max())
    _st = md.mask_stats(cp_masks, _voxel_sp)
    print(f'{_st["n"]} cells  |  median xy diameter {_st["diam_xy_um"]:.1f} µm  |  '
          f'median z extent {_st["z_extent_um"]:.1f} µm')

# ── QC: middle plane ──────────────────────────────────────────────────────────
from skimage.segmentation import find_boundaries
_raw_mid = md.seg_input(vol_sum3d[mid_idx:mid_idx + 1], _voxel_sp, sigma_um=(0, 0.7, 0.7))[0]
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
plt.suptitle(f'{FName}  —  Step 2: pre-processing & segmentation  (z={mid_idx}/{n_planes-1})', fontsize=9)
plt.tight_layout()
plt.show()""")

# ── Step 2b: subject reference -> session labels ──────────────────────────────
i2 = next(i for i, c in enumerate(nb['cells']) if c.get('id') == 'a5874f7b')
nb['cells'][i2 + 1:i2 + 1] = [
    md('md-2b', """## Step 2b: cells from the subject reference (multi-day tracking)
Runs only with `use_subject_reference = True` (Step 2). Needs the subject's curated `reference_labels.tif`
(`MultiDay_Reference.ipynb`).
1. **Register** this session's stitched intensity (fixed) to the reference (moving), in µm: MIP phase correlation
   + rotation sweep + depth search, then SimpleITK rigid (optionally a bounded B-spline) → `registration/*.h5`.
2. **Transfer labels**: reference labels → this session's grid (nearest neighbour; the TCSPC data are never
   resampled), then a per-cell local shift search (±3 µm) → `registration/<FName>_ref<id>_labels.tif`.
3. **QC per cell** (`<FName>_ref<id>_cell_qc.csv`): `ok`, `partial` (cut by the FOV or coverage), `low_match`
   (local NCC < 0.3), `outside`. Cells are flagged, not dropped; `cell_id` = the global reference ID.
4. The session is added to `<subject>/sessions.json`; Steps 14–16 then write `<FName>_ref<id>_*.csv`."""),
    code('code-2b', r"""# ── Step 2b: subject reference -> session labels ──────────────────────────────
subject_id        = vista_info['base'].split('_')[1]   # 'XYZT1_336_...' -> '336'
subject_root      = r'D:\Peredox_FLIM\subjects'
reg_stages        = ('rigid',)          # add 'bspline' if the overlay shows residual local warping
refine_shift_um   = (3.0, 3.0, 3.0)     # per-cell local search (z, y, x) after the label transfer
rerun_subject_reg = False               # True: redo registration + label transfer

if not use_subject_reference:
    print('Step 2b skipped (use_subject_reference = False): cells from Cellpose (Step 2).')
else:
    importlib.reload(md)
    subject_dir = os.path.join(subject_root, subject_id)
    _ref = md.load_reference(subject_dir)
    if _ref['labels'] is None:
        raise RuntimeError(f'No curated reference_labels.tif in {subject_dir} — run MultiDay_Reference.ipynb first.')
    cell_tag   = f'_ref{subject_id}'
    _reg_dir   = os.path.join(newFolder, 'registration')
    os.makedirs(_reg_dir, exist_ok=True)
    _tf_path   = os.path.join(_reg_dir, f'{FName}_to_ref{subject_id}.h5')
    cells_path = os.path.join(_reg_dir, f'{FName}{cell_tag}_labels.tif')
    _qc_path   = os.path.join(newFolder, f'{FName}{cell_tag}_cell_qc.csv')
    _ses_sp    = (z_step_um, xy_pixel_um, xy_pixel_um)
    _ses_cov   = tifffile.imread(stitch['cov_path'])
    _ses_p     = md.reg_prep(vol_sum3d, _ses_sp)
    _ref_p     = md.reg_prep(_ref['vol'], _ref['spacing'])

    # ── registration: session (fixed) -> reference (moving) ───────────────────
    if os.path.exists(_tf_path) and not rerun_subject_reg:
        T_ref, reg_summary = md.load_transform(_tf_path)
        print(f'Reloaded transform <- {os.path.basename(_tf_path)}')
    else:
        _reg = md.register_session(_ref_p, _ref['spacing'], _ses_p, _ses_sp, mov_mask=_ses_cov > 0,
                                   stages=reg_stages)
        md.save_transform(_tf_path, _reg, session=FName, reference=_ref['meta']['source'])
        T_ref, reg_summary = md.load_transform(_tf_path)
        print(f'Saved transform -> {_tf_path}')
    print(f'NCC per stage: { {k: round(v, 3) for k, v in reg_summary["ncc"].items()} }')

    # ── label transfer + per-cell refinement + QC ─────────────────────────────
    _ref_w = md.warp_to_session(_ref_p, _ref['spacing'], T_ref, vol_sum3d.shape, _ses_sp)
    _newer = lambda p, *srcs: os.path.exists(p) and all(os.path.getmtime(p) > os.path.getmtime(s) for s in srcs)
    if (not rerun_subject_reg and _newer(cells_path, _tf_path, md.subject_paths(subject_dir)['ref_labels'])
            and os.path.exists(_qc_path)):
        cp_masks   = tifffile.imread(cells_path)
        df_cell_qc = pd.read_csv(_qc_path)
        print(f'Reloaded session labels <- {os.path.basename(cells_path)}')
    else:
        _lab0 = md.warp_to_session(_ref['labels'], _ref['spacing'], T_ref, vol_sum3d.shape, _ses_sp, labels=True)
        cp_masks, _df_ref = md.refine_cells(_lab0, _ses_p, _ref_w, _ses_sp, max_shift_um=refine_shift_um)
        df_cell_qc = md.cell_qc(_ref['labels'], _ref['spacing'], cp_masks, _ses_sp, coverage=_ses_cov,
                                refine_df=_df_ref)
        tifffile.imwrite(cells_path, cp_masks.astype(np.uint16), compression='zlib')
        df_cell_qc.to_csv(_qc_path, index=False)
        print(f'Saved session labels -> {cells_path}')
    n_cells = int(_ref['labels'].max())     # global IDs: rows 1 .. max reference ID
    md.update_sessions(subject_dir, session=FName, date=md.session_date(raw_path), raw_path=raw_path,
                       analysis_dir=newFolder, cell_tag=cell_tag, transform=_tf_path, labels=cells_path,
                       qc=_qc_path)
    print(f'Reference cells: {len(df_cell_qc)}  |  flags: {df_cell_qc.flag.value_counts().to_dict()}')

    md.plot_registration_qc(_ses_p, _ref_w, _ses_sp, mask=_ses_cov > 0, labels=cp_masks, z=mid_idx,
                            title=f'{FName}  ->  reference {subject_id}')
    plt.show()"""),
]

# ── Steps 14-16: output names carry cell_tag; caches must be newer than the labels ──
c14 = cell(nb, 'b3fb5433')
replace(c14, """_cache_acc  = os.path.join(newFolder, f'{FName}_cell_acc.npz')
csv_out     = os.path.join(newFolder, f'{FName}_per_cell_lifetimes.csv')
_fits_path  = os.path.join(newFolder, f'{FName}_cell_fits.npz')   # decay + model per cell
_have_acc   = os.path.exists(_cache_acc) and not reRunPerCellDecay
_have_csv   = os.path.exists(csv_out)   and not reRunPerCellDecay
_have_fits  = os.path.exists(_fits_path) and not reRunPerCellDecay
""", """_cache_acc  = os.path.join(newFolder, f'{FName}{cell_tag}_cell_acc.npz')
csv_out     = os.path.join(newFolder, f'{FName}{cell_tag}_per_cell_lifetimes.csv')
_fits_path  = os.path.join(newFolder, f'{FName}{cell_tag}_cell_fits.npz')   # decay + model per cell

def _fresh(p):
    # a cache counts only if it is newer than the label volume it was built from (Step 2 / 2b)
    return (os.path.exists(p) and not reRunPerCellDecay
            and os.path.getmtime(p) > os.path.getmtime(cells_path))

_have_acc   = _fresh(_cache_acc)
_have_csv   = _fresh(csv_out)
_have_fits  = _fresh(_fits_path)
""")
replace(cell(nb, '61a152a1'), "_tav_csv_out = os.path.join(newFolder, f'{FName}_per_cell_tav.csv')",
        "_tav_csv_out = os.path.join(newFolder, f'{FName}{cell_tag}_per_cell_tav.csv')")
replace(cell(nb, '6e4891ec'), "_fig_cp_phas = os.path.join(newFolder, FName + '_per_cell_phasor.png')",
        "_fig_cp_phas = os.path.join(newFolder, FName + cell_tag + '_per_cell_phasor.png')")
replace(cell(nb, 'decay-inspect'), "pathCellDecay = os.path.join(newFolder, f'{FName}_cell{CELL_NUM}_decay.svg')",
        "pathCellDecay = os.path.join(newFolder, f'{FName}{cell_tag}_cell{CELL_NUM}_decay.svg')")
replace(cell(nb, 'ad5b1d96'), "`_cell_decays` dict is always reconstructed",
        "Output names carry `cell_tag` (`''` for Cellpose cells, `_ref<subject>` for reference cells, Step 2b), and the "
        "caches are reused only when newer than the label file they were built from (`cells_path`).\n\n"
        "`_cell_decays` dict is always reconstructed")
dump(nb, TILE, indent, raw.endswith('\n'))
print('tile notebook cells:', len(nb['cells']))

# ── MultiDay_Reference.ipynb ──────────────────────────────────────────────────
ref_cells = [
    md('mdr-00', """# MultiDay_Reference — per-subject reference cells
Cells are segmented **once per subject** on a reference volume and keep a global `cell_id`. Each FLIM session is
registered to this reference in `AnalyzeTCSPC_XYTZTile.ipynb` (Step 2: `use_subject_reference = True` → Step 2b), and
the session's own TCSPC data give every cell's lifetime.

1. **Reference volume** — a dedicated intensity stack (finer z, larger area than the daily FLIM FOV, more averaging;
   Fluoview-stitched ZYX TIFF), or, until it exists, a session's `<base>_stitched_intensity.tif`.
2. **Cellpose tuning** on a crop (settings grid).
3. **Full segmentation** → `reference/reference_labels_auto.tif`.
4. **Curation in napari** → `reference/reference_labels.tif` (IDs frozen from here on; new cells get new IDs).
5. **Across sessions** — after Steps 2b–16 ran for the sessions: `cells_long.csv` + plots."""),
    code('coder-01', r"""import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'   # prevent OpenMP conflict between MKL and torch on Windows
import torch   # must be imported before numpy/scipy to avoid OpenMP runtime clash

import sys
import json
import importlib
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tifffile

sys.path.insert(0, 'X:/FLIM_2026')
import multiday as md
importlib.reload(md)
print(f'torch {torch.__version__}  GPU available: {torch.cuda.is_available()}')"""),
    md('mdr-01', """## 1. Reference volume
The reference is recorded as a path + voxel size in `reference/reference_meta.json` (the TIFF is not copied).
Changing it after curation would invalidate `reference_labels.tif`, so that needs `replace_reference = True`."""),
    code('coder-02', r"""subject_id   = '336'
subject_root = r'D:\Peredox_FLIM\subjects'
# dedicated reference stack (ZYX TIFF) + its voxel size, or a session's stitched intensity (interim reference)
ref_path     = r'D:\Peredox_FLIM\XYZT1_336_X+0.50_Y+0.80_Decay128_512x512_Z186_Ch1_T1_P1_S1\XYZT1_336_X+0.50_Y+0.80_Decay128_512x512_Z186_Ch1_T1_P1_S1_stitched_intensity.tif'
ref_spacing  = (3.0, 0.99, 0.99)   # (dz, dy, dx) µm
replace_reference = False

subject_dir = os.path.join(subject_root, subject_id)
P = md.subject_paths(subject_dir)
if os.path.exists(P['ref_meta']):
    _old = json.load(open(P['ref_meta']))
    if (_old['source'] != ref_path or list(_old['spacing_zyx_um']) != list(ref_spacing)) \
            and os.path.exists(P['ref_labels']) and not replace_reference:
        raise RuntimeError(f'Subject {subject_id} already has curated labels for {_old["source"]}; '
                           'set replace_reference = True to switch reference (labels must be redone).')
ref_meta = md.save_reference_meta(subject_dir, ref_path, ref_spacing, subject=subject_id)
ref_vol, _ = md.load_volume(ref_path, ref_spacing)
print(f'Reference {ref_vol.shape} (z, y, x) at {ref_spacing} µm  ->  {subject_dir}')

_mip = ref_vol.max(axis=0)
_xz  = ref_vol[:, ref_vol.shape[1] // 2]
fig, axs = plt.subplots(1, 3, figsize=(18, 6), gridspec_kw=dict(width_ratios=[1, 1.4, 0.5]))
axs[0].imshow(_mip, cmap='gray', vmax=np.percentile(_mip, 99.5)); axs[0].set_title('max over z')
axs[1].imshow(_xz, cmap='gray', aspect=ref_spacing[0] / ref_spacing[2], vmax=np.percentile(_xz, 99.5))
axs[1].set_title('xz section (middle y)')
axs[2].plot(ref_vol.mean(axis=(1, 2)), np.arange(ref_vol.shape[0]) * ref_spacing[0])
axs[2].invert_yaxis(); axs[2].set_xlabel('mean intensity'); axs[2].set_ylabel('depth (µm)')
plt.tight_layout(); plt.show()"""),
    md('mdr-02', """## 2. Cellpose tuning on a crop
Each row of the montage is one setting (outlines on the smoothed input), columns are z planes. Pick the setting whose
outlines follow the objects you want to track. Notes:
- `2d_stitch` segments each plane and links masks across z; `3d` is far slower at small diameters (minutes per crop).
- Lower `cellprob_threshold` grows/merges objects rather than adding many new ones on this data.
- On the 2026-11-02 session Cellpose found few objects at any setting: the bright objects are small (5–8 µm) and
  often split into horizontal bars (brain motion during line scanning). Expect to add cells by hand in step 4, or
  fine-tune Cellpose on curated planes later."""),
    code('coder-03', r"""crop_yx      = (250, 700)            # top-left (y, x) of the crop, px
crop_size    = 256
z_show       = None                  # planes in the montage; None = 4 evenly spaced
seg_sigma_um = (1.0, 0.7, 0.7)       # light smoothing of the Cellpose input (z, y, x) µm
grid = [
    dict(mode='2d_stitch', diameter_um=8,  cellprob_threshold=-1.5),
    dict(mode='2d_stitch', diameter_um=8,  cellprob_threshold=0.0),
    dict(mode='2d_stitch', diameter_um=12, cellprob_threshold=-1.5),
    dict(mode='2d_stitch', diameter_um=16, cellprob_threshold=-1.5),
]

_y0, _x0 = crop_yx
_crop = md.seg_input(ref_vol[:, _y0:_y0 + crop_size, _x0:_x0 + crop_size], ref_spacing, sigma_um=seg_sigma_um)
_zs   = z_show or [int(z) for z in np.linspace(0, _crop.shape[0] - 1, 6)[1:-1].round()]
grid_res = md.cellpose_grid(_crop, ref_spacing, grid)
md.plot_cellpose_grid(_crop, grid_res, _zs, title=f'crop y0={_y0} x0={_x0}  ({crop_size} px)')
plt.show()
pd.DataFrame([dict(**r['settings'], **{k: round(v, 1) for k, v in r['stats'].items()}, seconds=round(r['seconds']))
              for r in grid_res])"""),
    md('mdr-03', """## 3. Full segmentation
Runs the chosen setting on the whole reference → `reference_labels_auto.tif` (+ `.json` with the settings).
This is the starting point of the curation; it is never used directly by the sessions."""),
    code('coder-04', r"""seg_settings = dict(mode='2d_stitch', diameter_um=8, cellprob_threshold=-1.5)   # chosen from the grid
rerun_seg    = False

_json = P['ref_labels_auto'].replace('.tif', '.json')
_want = dict(seg_settings, seg_sigma_um=list(seg_sigma_um), source=ref_path)
if os.path.exists(P['ref_labels_auto']) and os.path.exists(_json) and json.load(open(_json)) == _want and not rerun_seg:
    labels_auto = tifffile.imread(P['ref_labels_auto'])
    print(f'Reloaded {os.path.basename(P["ref_labels_auto"])}')
else:
    _inp = md.seg_input(ref_vol, ref_spacing, sigma_um=seg_sigma_um)
    labels_auto = md.segment_cellpose(_inp, ref_spacing, **seg_settings)
    tifffile.imwrite(P['ref_labels_auto'], labels_auto, compression='zlib')
    with open(_json, 'w') as f:
        json.dump(_want, f, indent=1)
    print(f'Saved -> {P["ref_labels_auto"]}')
print(md.mask_stats(labels_auto, ref_spacing))"""),
    md('mdr-04', """## 4. Curation in napari
Opens the reference and the labels in napari (`napari-env`); the cell waits until the window is closed.
Start: `reference_labels.tif` if it exists (continue curating), else `reference_labels_auto.tif`.
- select the **cells** layer: paint `P`, erase `E`, fill `F`, pick a label `L`; the brush works in the current plane
  (use the z slider), or in 3D with *n edit dim* = 3
- new cell: set the label number to an unused ID (the layer starts at max ID + 1); delete a cell: fill it with 0;
  merge: fill one with the other's ID
- **Ctrl+Shift+S** saves to `reference_labels.tif`; closing the window saves too
- IDs are global and kept from here on: do not renumber; removed IDs are simply absent"""),
    code('coder-05', r"""napari_env = r'X:\conda\envs\napari-env'
_start = P['ref_labels'] if os.path.exists(P['ref_labels']) else P['ref_labels_auto']
_env = dict(os.environ)
_env['PATH'] = ';'.join([napari_env, rf'{napari_env}\Library\mingw-w64\bin', rf'{napari_env}\Library\usr\bin',
                         rf'{napari_env}\Library\bin', rf'{napari_env}\Scripts', rf'{napari_env}\bin', _env['PATH']])
_env.pop('KMP_DUPLICATE_LIB_OK', None)
print(f'napari: {os.path.basename(_start)} -> {os.path.basename(P["ref_labels"])}   (close the window when done)')
_r = subprocess.run([rf'{napari_env}\python.exe', r'X:\FLIM_2026\napari_curate.py', ref_meta['source'], _start,
                     P['ref_labels'], ','.join(str(s) for s in ref_spacing)],
                    env=_env, capture_output=True, text=True)
print(_r.stdout[-2000:] or '(no output)')
if _r.returncode:
    print(_r.stderr[-3000:])"""),
    code('coder-06', r"""# ── freeze the curated cells: table of IDs, volumes and centroids ─────────────
labels_ref = tifffile.imread(P['ref_labels']).astype(np.uint16)
cells_ref  = md.reference_cells_table(labels_ref, ref_spacing)
if os.path.exists(P['ref_cells']):
    _prev = set(pd.read_csv(P['ref_cells']).cell_id)
    _gone, _new = sorted(_prev - set(cells_ref.cell_id)), sorted(set(cells_ref.cell_id) - _prev)
    print(f'vs previous table: {len(_new)} new IDs, {len(_gone)} removed IDs' + (f'  removed: {_gone[:20]}' if _gone else ''))
cells_ref.to_csv(P['ref_cells'], index=False)
print(f'{len(cells_ref)} reference cells (IDs {cells_ref.cell_id.min()}–{cells_ref.cell_id.max()})  ->  {P["ref_cells"]}')
print(md.mask_stats(labels_ref, ref_spacing))"""),
    md('mdr-05', """## 5. Across sessions
Collects every session registered in `sessions.json` (Step 2b adds them) whose Steps 14–16 have run: per-cell
lifetimes, phasor, tav statistics and the registration QC → `cells_long.csv` (one row per session × cell).
Only cells flagged `ok` are plotted by default."""),
    code('coder-07', r"""long = md.collect_subject(subject_dir)
if len(long):
    md.plot_subject(long, value='aw_LT', flags=('ok',), min_sessions=2, title=f'subject {subject_id}: aw_LT')
    plt.show()
    if 'median_tav' in long:
        md.plot_subject(long, value='median_tav', flags=('ok',), min_sessions=2,
                        title=f'subject {subject_id}: median tav')
        plt.show()
    display(long.groupby(['date', 'flag']).size().unstack(fill_value=0))"""),
]
ref_nb = dict(cells=ref_cells, metadata=nb['metadata'], nbformat=nb['nbformat'], nbformat_minor=nb['nbformat_minor'])
dump(ref_nb, REF, indent, True)
print('reference notebook cells:', len(ref_cells))
