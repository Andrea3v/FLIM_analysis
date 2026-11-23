"""Insert the Step 2t segmentation-tuning sandbox (photon-sum reference, gentler mask pre-processing,
Cellpose grid on a crop) after Step 2 of AnalyzeTCSPC_XYTZTime.ipynb.  Re-running replaces the cells."""
import json
import sys

NB = r'X:\FLIM_2026\AnalyzeTCSPC_XYTZTime.ipynb'
AFTER = 'a5874f7b'   # Step 2 code cell

T4 = r"""# ── Step 2t-4: per-cell quality filter (sandbox: cp_masks untouched) ─────────
tn_qc_labels       = 'auto'   # 'auto' = tn_cp_full (2t-3 full run) if present, else cp_masks; or 'tuned' / 'step2'
tn_qc_min_contrast = 1.25     # cell mean / median of a ring around it (tn_ref); 1 = no brighter than its surround
tn_qc_ring_um      = 3.0      # ring width (µm, in-plane; other cells and unrecorded voxels left out)
tn_qc_min_fov      = 0.9      # a timepoint counts when this fraction of the cell was recorded (Step 17: tc_min_fov)
tn_qc_min_photons  = 500      # photons per cell and timepoint (Step 14 / 17: min_photons)
tn_qc_min_tp       = 'all'    # timepoints that must meet both: int, or 'all' (= n_timepoints)

import warnings as _tn_w
from scipy import ndimage as _tn_ndi
from skimage.morphology import disk as _tn_disk
from matplotlib.patches import Rectangle as _tn_Rect

_tn_full = globals().get('tn_cp_full')
_tn_full = _tn_full if _tn_full is not None and _tn_full.shape == tn_ref.shape else None
if tn_qc_labels == 'tuned' and _tn_full is None:
    raise RuntimeError("tn_qc_labels='tuned' needs the full run of 2t-3 (tn_run_full = True)")
_tn_use_full = tn_qc_labels == 'tuned' or (tn_qc_labels == 'auto' and _tn_full is not None)
_tn_lab  = _tn_full if _tn_use_full else cp_masks
_tn_labn = 'tn_cp_full (2t-3)' if _tn_use_full else 'cp_masks (Step 2)'
if _tn_lab is None or _tn_lab.shape != tn_ref.shape:
    raise RuntimeError(f'{_tn_labn}: no labels with the canvas shape {tn_ref.shape}')
_tn_n    = int(_tn_lab.max())
_tn_ids  = np.arange(1, _tn_n + 1)
_tn_flat = _tn_lab.ravel()
_tn_vox  = np.bincount(_tn_flat, minlength=_tn_n + 1)[1:]
_tn_has  = _tn_vox > 0
_tn_T    = tile_int_t.shape[0]
_tn_need = _tn_T if tn_qc_min_tp == 'all' else int(tn_qc_min_tp)

# ── shape: centre plane, planes spanned, equivalent xy diameter of the largest cross-section ──
_tn_obj  = _tn_ndi.find_objects(_tn_lab)
_tn_zc   = np.array([(o[0].start + o[0].stop - 1) / 2 if o else np.nan for o in _tn_obj])
_tn_nzp  = np.array([o[0].stop - o[0].start if o else 0 for o in _tn_obj])
_tn_area = np.zeros(_tn_n + 1)
for _z in range(_tn_lab.shape[0]):
    np.maximum(_tn_area, np.bincount(_tn_lab[_z].ravel(), minlength=_tn_n + 1), out=_tn_area)
_tn_dxy  = 2 * np.sqrt(_tn_area[1:] * xy_pixel_um ** 2 / np.pi)

# ── contrast: cell mean / median of a ring around the cell (photon-sum reference) ──
_tn_fp   = _tn_disk(max(1, int(round(tn_qc_ring_um / xy_pixel_um))))
_tn_ring = np.zeros_like(_tn_lab)
for _z in range(_tn_lab.shape[0]):
    _tn_ring[_z] = np.where((_tn_lab[_z] == 0) & (tn_ntp[_z] > 0),
                            _tn_ndi.grey_dilation(_tn_lab[_z], footprint=_tn_fp), 0)
_tn_nring = np.bincount(_tn_ring.ravel(), minlength=_tn_n + 1)[1:]
with _tn_w.catch_warnings(), np.errstate(all='ignore'):
    _tn_w.simplefilter('ignore')
    _tn_con = (np.asarray(_tn_ndi.mean(tn_ref, _tn_lab, _tn_ids), float)
               / np.maximum(np.asarray(_tn_ndi.median(tn_ref, _tn_ring, _tn_ids), float), 1e-9))
_tn_con[(_tn_nring == 0) | ~_tn_has] = np.nan            # no ring (surrounded by cells / edge): not judged

# ── photons per cell and timepoint, and the recorded fraction of the cell (as Step 17 counts them) ──
_tn_ph  = np.zeros((_tn_T, _tn_n))
_tn_fov = np.zeros((_tn_T, _tn_n))
for _t in range(_tn_T):
    _cv = np.zeros(_tn_can, bool)
    for _p in range(tile_int_t.shape[1]):
        _z0, _y0, _x0 = _tn_pos[_t, _p]
        _sl = (slice(_z0, _z0 + _tn_shp[0]), slice(_y0, _y0 + _tn_shp[1]), slice(_x0, _x0 + _tn_shp[2]))
        _tn_ph[_t] += np.bincount(_tn_lab[_sl].ravel(), weights=tile_int_t[_t, _p].ravel(),   # overlaps summed
                                  minlength=_tn_n + 1)[1:]                                     #  (Step 0d TCSPC)
        _cv[_sl] |= _tn_ok[_t, _p][:, None, None]
    _tn_fov[_t] = np.bincount(_tn_flat, weights=_cv.ravel(), minlength=_tn_n + 1)[1:] / np.maximum(_tn_vox, 1)
_tn_infov = _tn_fov >= tn_qc_min_fov
_tn_okt   = _tn_infov & (_tn_ph >= tn_qc_min_photons)
_tn_phin  = np.where(_tn_infov, _tn_ph, np.nan)
with _tn_w.catch_warnings():
    _tn_w.simplefilter('ignore')
    _tn_phmed, _tn_phmin = np.nanmedian(_tn_phin, axis=0), np.nanmin(_tn_phin, axis=0)

# ── table, kept cells ────────────────────────────────────────────────────────
tn_qc = pd.DataFrame(dict(cell_id=_tn_ids, z_centre=_tn_zc, n_planes=_tn_nzp, d_xy_um=_tn_dxy.round(1),
                          contrast=_tn_con.round(3), tp_in_fov=_tn_infov.sum(0), tp_usable=_tn_okt.sum(0),
                          photons_median=_tn_phmed.round(0), photons_min=_tn_phmin.round(0)))[_tn_has]
tn_qc = tn_qc.reset_index(drop=True)
tn_qc['pass_contrast'] = ~(tn_qc['contrast'] < tn_qc_min_contrast)   # NaN (no ring) passes
tn_qc['pass_fov']      = tn_qc['tp_in_fov'] >= _tn_need
tn_qc['pass_photons']  = tn_qc['tp_usable'] >= _tn_need               # in the FOV and >= min photons
tn_qc['keep']          = tn_qc['pass_contrast'] & tn_qc['pass_photons']
tn_keep  = tn_qc.loc[tn_qc['keep'], 'cell_id'].to_numpy()
_tn_lut  = np.zeros(_tn_n + 1, np.uint16 if tn_keep.size < 65535 else np.uint32)
_tn_lut[tn_keep] = np.arange(1, tn_keep.size + 1)
tn_cp_qc = _tn_lut[_tn_lab]                                           # kept cells, relabelled 1..n
tn_qc['qc_id'] = _tn_lut[tn_qc['cell_id'].to_numpy()]

_tn_qcp = os.path.join(tn_dir, f'{FName}_tune_cell_qc')
tn_qc.to_csv(_tn_qcp + '.csv', index=False)
with open(_tn_qcp + '.json', 'w') as _f:
    json.dump(dict(labels=_tn_labn, min_contrast=tn_qc_min_contrast, ring_um=tn_qc_ring_um, min_fov=tn_qc_min_fov,
                   min_photons=tn_qc_min_photons, min_tp=_tn_need, n_cells=int(_tn_has.sum()),
                   n_kept=int(tn_keep.size)), _f, indent=1)
tifffile.imwrite(os.path.join(tn_dir, f'{FName}_tune_cellpose_masks_qc.tif'), tn_cp_qc)

_tn_N = len(tn_qc)
_pc = lambda k: f'{k:4d} ({100 * k / max(_tn_N, 1):3.0f}%)'
print(f'Step 2t-4 on {_tn_labn}: {_tn_N} cells')
print(f'  contrast >= {tn_qc_min_contrast:g} (ring {tn_qc_ring_um:g} µm)             : {_pc(tn_qc.pass_contrast.sum())}')
print(f'  in the FOV (>= {tn_qc_min_fov:.0%} of the cell) at >= {_tn_need}/{_tn_T} timepoints: {_pc(tn_qc.pass_fov.sum())}')
print(f'    ... with >= {tn_qc_min_photons} photons at those          : {_pc(tn_qc.pass_photons.sum())}')
print(f'  kept (contrast and photons)                        : {_pc(tn_keep.size)}'
      f'   -> tn_cp_qc, {os.path.basename(_tn_qcp)}.csv')
print('Cells meeting both at each timepoint: ' + '  '.join(f't{_t:02d} {int(_tn_okt[_t, _tn_has].sum())}'
                                                            for _t in range(_tn_T)))
_tn_byz = tn_qc.assign(z=np.floor(tn_qc['z_centre']).astype(int)).groupby('z')
display(pd.DataFrame(dict(cells=_tn_byz.size(), pass_contrast=_tn_byz['pass_contrast'].sum(),
                          pass_fov=_tn_byz['pass_fov'].sum(), pass_photons=_tn_byz['pass_photons'].sum(),
                          kept=_tn_byz['keep'].sum())).rename_axis('centre plane z').T)

# ── figure: outlines by verdict, contrast histogram, photons per timepoint ──
_tn_cat = np.zeros(_tn_n + 1, np.uint8)
_tn_cat[tn_qc['cell_id'].to_numpy()] = np.select([tn_qc['keep'], ~tn_qc['pass_contrast']], [1, 2], 3)
_tn_ccol = {1: (1.0, 0.25, 0.1), 2: (1.0, 0.9, 0.0), 3: (0.25, 0.6, 1.0)}


def _tn_qc_rgb(im, lab):
    rgb, b, c = _tn_rgb(im), _tn_fb(lab, mode='inner'), _tn_cat[lab]
    for k, v in _tn_ccol.items():
        rgb[b & (c == k)] = v
    return rgb


fig, axs = plt.subplots(1, 4, figsize=(25, 6.6), gridspec_kw=dict(width_ratios=[1, 1, 0.8, 0.8]))
axs[0].imshow(_tn_qc_rgb(tn_ref[tn_zv], _tn_lab[tn_zv]), interpolation='nearest')
axs[0].add_patch(_tn_Rect((tn_crop[1].start, tn_crop[0].start), tn_crop[1].stop - tn_crop[1].start,
                          tn_crop[0].stop - tn_crop[0].start, fill=False, ec='w', lw=1, ls='--'))
axs[0].set_title(f'z={tn_zv}: red kept, yellow contrast < {tn_qc_min_contrast:g}, '
                 f'blue < {_tn_need} timepoints usable', fontsize=9)
axs[1].imshow(_tn_qc_rgb(tn_ref[tn_zv][tn_crop], _tn_lab[tn_zv][tn_crop]), interpolation='nearest')
axs[1].set_title('crop', fontsize=9)
for _a in axs[:2]:
    _a.axis('off')
axs[2].hist(np.clip(tn_qc['contrast'].dropna(), 0, 4), bins=60, color='0.5')
axs[2].axvline(tn_qc_min_contrast, color='r', ls='--')
axs[2].set_xlabel('cell mean / ring median (clipped at 4)'); axs[2].set_ylabel('cells')
axs[2].set_title(f'contrast: {int((~tn_qc.pass_contrast).sum())} cells below {tn_qc_min_contrast:g}', fontsize=9)
with _tn_w.catch_warnings():
    _tn_w.simplefilter('ignore')
    _tn_q = np.nanpercentile(_tn_phin[:, _tn_has], [10, 50, 90], axis=1)
axs[3].fill_between(np.arange(_tn_T), _tn_q[0], _tn_q[2], color='C0', alpha=0.25, label='10-90 %')
axs[3].plot(np.arange(_tn_T), _tn_q[1], 'o-', color='C0', label='median')
axs[3].axhline(tn_qc_min_photons, color='r', ls='--', label=f'min {tn_qc_min_photons}')
axs[3].set_yscale('log'); axs[3].set_xlabel('timepoint'); axs[3].set_ylabel('photons per cell')
axs[3].set_title('photons per cell and timepoint (cells in the FOV)', fontsize=9); axs[3].legend(fontsize=8)
plt.suptitle(f'{FName}  —  Step 2t-4: per-cell quality filter  ({_tn_labn})', fontsize=9)
plt.tight_layout()
plt.show()
"""

MD = r"""## Step 2t: segmentation tuning (sandbox)
Try a new segmentation reference, a gentler mask pre-processing and Cellpose settings **without touching**
`vol_sum3d`, `mask3d`, `cp_masks`: Steps 3–17 run as before. Files go to `<newFolder>/segmentation_tuning/`.
Run after Step 2; the cells can be re-run on their own (2t-1 first). Once the settings look right they get
merged into Steps 0d / 2.
- **2t-1 reference** `tn_ref`: Σ photons of all timepoints ÷ Σ brightness *b* of the timepoints (× tiles) covering
  each voxel. Where every timepoint covers a voxel this is the photon sum ÷ a constant (Poisson-optimal: brighter
  timepoints weigh more, e.g. Peredox brightening on NADH binding); at the drift edges it keeps the same level, where
  a plain sum would dim. *b* = mean photons on the tissue all timepoints share. Step 2 instead averages the
  timepoints after normalising each to its own 1–99.5 percentile.
- **2t-2 mask** `tn_norm`, `tn_mask`: optional light Gaussian → NLM (`'2d'` per plane or `'3d'`, strength
  h = `tn_nlm_hfac` × noise σ) → background subtraction → renormalise → threshold. Background options:
  `'ellipsoid'` = rolling ball scaled to the data range (removes slow background only); `'ball'` = Step 2 as is (a
  radius-30 px ball on 0..1 data is so curved that it fits inside 8 µm cells and hollows them); `'gauss'` = minus a
  wide Gaussian; `None`. The table scores the masks against Cellpose cells: share of cell voxels in the mask vs share
  of non-cell voxels (the latter also counts cells Cellpose missed, so it overstates false positives).
- **2t-3 Cellpose grid** on the crop: diameter (rows) × cellprob × flow threshold (columns), outlines on plane
  `tn_zv`. The flow threshold matters most here: Cellpose drops every mask whose flow error exceeds it (any value
  ≥ 0; higher = more lenient, `0` = check off). On Slice3 every mask has an error of 0.75–0.92, so the default 0.4
  (Step 2) throws away nearly all somata, 0.1 leaves none (NaN in the table) and ≥ 1 equals 0: the check does not
  separate good from bad cells in this tissue. Stitching joins masks on neighbouring planes when their
  intersection / union ≥ `stitch_threshold`, each mask to its best match only.
  `tn_run_full = True` runs `tn_full_settings` on the whole volume → `tn_cp_full`; re-run 2t-2 to score the mask
  against these cells. `tn_cp_input = 'tuned'` feeds the 2t-2 output to Cellpose instead of the smoothed intensity.
- **2t-4 quality filter** on `tn_cp_full` (else `cp_masks`): per cell, the contrast (mean of `tn_ref` / median of a
  ring around the cell) and, per timepoint, the recorded fraction of the cell and its photons (as Step 17 counts
  them). Kept: contrast ≥ `tn_qc_min_contrast` and ≥ `tn_qc_min_photons` photons with ≥ `tn_qc_min_fov` of the
  cell recorded at `tn_qc_min_tp` timepoints → `tn_qc` (table, `*_tune_cell_qc.csv`), `tn_cp_qc` (kept cells,
  relabelled). Outlines: red kept, yellow low contrast, blue too few usable timepoints.
"""

T1 = r"""# ── Step 2t-1: photon-sum segmentation reference (sandbox: vol_sum3d / mask3d / cp_masks untouched) ──
tn_crop_um = 200.0      # crop for the views and the Cellpose grid (µm, square)
tn_crop_yx = 'auto'     # top-left (y, x) in px, or 'auto' = the crop with the most cell-scale structure
tn_z       = None       # plane shown; None = the plane picked with the crop ('auto'), else mid_idx

import time as _time
from skimage.restoration import estimate_sigma as _tn_esig
from skimage.segmentation import find_boundaries as _tn_fb
tn_dir = os.path.join(newFolder, 'segmentation_tuning')
os.makedirs(tn_dir, exist_ok=True)
_tn_sp = (z_step_um, xy_pixel_um, xy_pixel_um)


def _tn_n01(v, sel=None, pct=(1, 99)):
    lo, hi = np.percentile(v if sel is None else v[sel], pct)
    return np.clip((v - lo) / max(hi - lo, 1e-9), 0, 1).astype(np.float32)


def _tn_rgb(im, labels=None, mask=None):   # grey image (1-99.7 %), mask tinted cyan, label outlines red
    rgb = np.dstack([_tn_n01(im, pct=(1, 99.7))] * 3)
    if mask is not None:
        rgb[mask] = 0.55 * rgb[mask] + 0.45 * np.array([0.1, 0.8, 0.9])
    if labels is not None:
        rgb[_tn_fb(labels, mode='inner')] = (1.0, 0.25, 0.1)
    return rgb


# ── tile origins of every timepoint on the canvas (as Step 0d) ────────────────
_tn_pos = np.asarray(pos_t if n_timepoints > 1 else [tile_reg['positions']], int)    # (time, tile, zyx)
_tn_can = tuple(int(v) for v in (canvas_t if n_timepoints > 1 else tile_reg['canvas_shape']))
_tn_shp = np.array(tile_int_t.shape[2:])
_tn_ok  = tile_int_t.reshape(*tile_int_t.shape[:3], -1).max(axis=3) > 0   # (time, tile, z): False = not recorded

# ── brightness b per timepoint x tile: mean photons on the tissue all timepoints of that tile share ──
tn_b = np.full(tile_int_t.shape[:2], np.nan)
for _tn_p in range(tile_int_t.shape[1]):
    _tn_s = _tn_pos[:, _tn_p].max(axis=0)                 # common box of the timepoints (z drift: partial overlap)
    _tn_e = _tn_pos[:, _tn_p].min(axis=0) + _tn_shp
    for _tn_t in range(tile_int_t.shape[0]):
        _tn_v = tile_int_t[_tn_t, _tn_p]
        _tn_a, _tn_c = _tn_s - _tn_pos[_tn_t, _tn_p], _tn_e - _tn_pos[_tn_t, _tn_p]
        _tn_r = (_tn_v[_tn_a[0]:_tn_c[0], _tn_a[1]:_tn_c[1], _tn_a[2]:_tn_c[2]][_tn_ok[_tn_t, _tn_p, _tn_a[0]:_tn_c[0]]]
                 if np.all(_tn_c > _tn_a) else _tn_v[_tn_ok[_tn_t, _tn_p]])
        if _tn_r.size:
            tn_b[_tn_t, _tn_p] = _tn_r.mean()
tn_b[~(tn_b > 0)] = np.nanmedian(tn_b[tn_b > 0])

# ── Σ photons / Σ b over the timepoints x tiles covering each voxel (tile overlaps feathered as Step 0d) ──
_tn_fe  = vt._feather(*_tn_shp[1:])
_tn_num = np.zeros(_tn_can, np.float32)
_tn_den = np.zeros(_tn_can, np.float32)
tn_ntp  = np.zeros(_tn_can, np.uint8)                     # timepoints x tiles per voxel
for _tn_t in range(tile_int_t.shape[0]):
    for _tn_p in range(tile_int_t.shape[1]):
        _tn_z0, _tn_y0, _tn_x0 = _tn_pos[_tn_t, _tn_p]
        _tn_sl = (slice(_tn_z0, _tn_z0 + _tn_shp[0]), slice(_tn_y0, _tn_y0 + _tn_shp[1]),
                  slice(_tn_x0, _tn_x0 + _tn_shp[2]))
        _tn_w = _tn_ok[_tn_t, _tn_p][:, None, None]
        _tn_num[_tn_sl] += _tn_fe * tile_int_t[_tn_t, _tn_p]
        _tn_den[_tn_sl] += _tn_fe * _tn_w * np.float32(tn_b[_tn_t, _tn_p])
        tn_ntp[_tn_sl]  += _tn_w.astype(np.uint8)
tn_ref  = np.divide(_tn_num, _tn_den, out=np.zeros(_tn_can, np.float32), where=_tn_den > 0)
_tn_cov = tn_ntp > 0
if tn_ref.shape != vol_sum3d.shape:
    raise ValueError(f'photon-sum reference {tn_ref.shape} != Step 2 intensity {vol_sum3d.shape}: re-run Step 0d')
_tn_ref_path = os.path.join(tn_dir, f'{FName}_photonsum_ref.tif')
if not (os.path.exists(_tn_ref_path) and np.array_equal(tifffile.imread(_tn_ref_path), tn_ref)):
    tifffile.imwrite(_tn_ref_path, tn_ref)   # rewritten only when it changes (2t-3 reuses its full run meanwhile)

# ── crop and plane for the views / the Cellpose grid ──────────────────────────
_tn_c = min(int(round(tn_crop_um / xy_pixel_um)), *_tn_can[1:])
if tn_crop_yx == 'auto':   # most cell-scale structure, on voxels most timepoints cover (edges are noisier)
    _tn_k, _ = md.training_crops(np.where(tn_ntp >= tn_ntp.max() / 2, tn_ref, 0), _tn_sp, n=1,
                                 crop_um=_tn_c * xy_pixel_um)
    _tn_zc, _tn_cy, _tn_cx = _tn_k[0] if _tn_k else (mid_idx, (_tn_can[1] - _tn_c) // 2, (_tn_can[2] - _tn_c) // 2)
else:
    (_tn_cy, _tn_cx), _tn_zc = tn_crop_yx, mid_idx
_tn_cy = int(np.clip(_tn_cy, 0, _tn_can[1] - _tn_c))
_tn_cx = int(np.clip(_tn_cx, 0, _tn_can[2] - _tn_c))
tn_crop = (slice(_tn_cy, _tn_cy + _tn_c), slice(_tn_cx, _tn_cx + _tn_c))
tn_zv   = int(tn_z if tn_z is not None else _tn_zc)


def _tn_cnr(v):   # contrast / noise in the crop: (p99.5 - p1) / wavelet noise sigma
    c = v[(slice(None),) + tn_crop]
    return (np.percentile(c, 99.5) - np.percentile(c, 1)) / max(float(_tn_esig(c, average_sigmas=True, channel_axis=None)), 1e-12)


_tn_rel = tn_b / tn_b.mean(axis=0, keepdims=True)
print('Brightness b (mean photons on the shared tissue) / mean over timepoints'
      + (' (tile 0)' if tile_int_t.shape[1] > 1 else '') + ':  '
      + '  '.join(f't{t:02d} {r:.2f}' for t, r in enumerate(_tn_rel[:, 0])))
print(f'Reference = Σ photons / Σ b of the timepoints x tiles covering each voxel  -> {_tn_ref_path}')
print(f'Contrast / noise in the crop:  Step 2 intensity {_tn_cnr(vol_sum3d):.1f}  |  photon sum {_tn_cnr(tn_ref):.1f}')
print(f'Crop: y {_tn_cy}-{_tn_cy + _tn_c}, x {_tn_cx}-{_tn_cx + _tn_c}  ({_tn_c} px = {_tn_c * xy_pixel_um:.0f} µm)'
      f'  |  plane z={tn_zv}')

fig, axs = plt.subplots(1, 4, figsize=(22, 5.8))
axs[0].imshow(_tn_rgb(tn_ref[tn_zv]), interpolation='nearest')
axs[0].add_patch(plt.Rectangle((_tn_cx - 0.5, _tn_cy - 0.5), _tn_c, _tn_c, fill=False, ec='yellow', lw=1.2))
axs[0].set_title(f'photon-sum reference  z={tn_zv}  (crop in yellow)', fontsize=10)
_im = axs[1].imshow(tn_ntp[tn_zv], cmap='viridis', interpolation='nearest', vmin=0, vmax=tn_ntp.max())
plt.colorbar(_im, ax=axs[1], fraction=0.046)
axs[1].set_title(f'timepoints x tiles per voxel  z={tn_zv}', fontsize=10)
axs[2].imshow(_tn_rgb(vol_sum3d[tn_zv][tn_crop]), interpolation='nearest')
axs[2].set_title('Step 2: mean of normalised timepoints (crop)', fontsize=10)
axs[3].imshow(_tn_rgb(tn_ref[tn_zv][tn_crop]), interpolation='nearest')
axs[3].set_title('photon sum / Σ b (crop)', fontsize=10)
for _a in axs:
    _a.axis('off')
plt.suptitle(f'{FName}  —  Step 2t-1: segmentation reference', fontsize=9)
plt.tight_layout()
plt.show()
"""

T2 = r"""# ── Step 2t-2: gentler pre-processing -> tn_norm, tn_mask (sandbox: Step 2 mask3d untouched) ──
tn_mask_src    = 'photonsum'        # 'photonsum' (2t-1) or 'step2' (vol_sum3d, the Step 2 input)
tn_smooth_um   = (0.0, 0.7, 0.7)    # Gaussian sigma (z, y, x) µm before NLM; (0, 0, 0) = none
tn_nlm         = '2d'               # None, '2d' (per plane) or '3d' (as Step 2)
tn_nlm_hfac    = 0.4                # NLM strength h = factor x noise sigma (Step 2: 0.8); lower = gentler
tn_nlm_patch   = 5                  # px (Step 2: 5)
tn_nlm_dist    = 6                  # px (Step 2: 6)
tn_bg          = 'ellipsoid'        # None, 'ellipsoid' (rolling ball scaled to the data), 'ball' (Step 2), 'gauss'
tn_bg_um       = 30.0               # background scale (µm): ball / ellipsoid radius, Gaussian sigma; >> cell size
tn_bg_height   = 1.0                # 'ellipsoid' height in normalised intensity (data 0..1); lower = harder
tn_thresh      = 0.55               # tn_mask = tn_norm >= tn_thresh
tn_thresh_scan = (0.35, 0.45, 0.55, 0.65, 0.75)   # thresholds scored in the table
# Step 2 as is: tn_mask_src='step2', tn_smooth_um=(0, 0, 0), tn_nlm='3d', tn_nlm_hfac=0.8, tn_bg='ball'

from skimage.restoration import (denoise_nl_means as _tn_nlm_fn, rolling_ball as _tn_rb,
                                 ellipsoid_kernel as _tn_ek)
_tn_t0 = _time.time()
_tn_v  = _tn_n01(tn_ref if tn_mask_src == 'photonsum' else vol_sum3d, _tn_cov)
if any(tn_smooth_um):
    _tn_v = gaussian_filter(_tn_v, [s / d for s, d in zip(tn_smooth_um, _tn_sp)])
if tn_nlm:
    _tn_h  = tn_nlm_hfac * float(_tn_esig(_tn_v, average_sigmas=True, channel_axis=None))
    _tn_kw = dict(h=_tn_h, patch_size=tn_nlm_patch, patch_distance=tn_nlm_dist, fast_mode=True, channel_axis=None)
    _tn_v  = (np.stack([_tn_nlm_fn(_p, **_tn_kw) for _p in _tn_v]) if tn_nlm == '2d'
              else _tn_nlm_fn(_tn_v, **_tn_kw)).astype(np.float32)
_tn_r = tn_bg_um / xy_pixel_um
if tn_bg == 'ellipsoid':
    _tn_k = _tn_ek((2 * int(round(_tn_r)) + 1,) * 2, tn_bg_height)
    _tn_v = np.stack([np.clip(_p - _tn_rb(_p, kernel=_tn_k, workers=-1), 0, None) for _p in _tn_v])
elif tn_bg == 'ball':
    _tn_v = np.stack([np.clip(_p - _tn_rb(_p, radius=_tn_r, workers=-1), 0, None) for _p in _tn_v])
elif tn_bg == 'gauss':
    _tn_v = np.clip(_tn_v - gaussian_filter(_tn_v, (0, _tn_r, _tn_r)), 0, None)
elif tn_bg is not None:
    raise ValueError(f"tn_bg must be None, 'ellipsoid', 'ball' or 'gauss', not {tn_bg!r}")
tn_norm = _tn_n01(_tn_v, _tn_cov)
tn_mask = (tn_norm >= tn_thresh) & _tn_cov
print(f'{tn_mask_src}  ->  smooth {tn_smooth_um} µm  ->  NLM {tn_nlm or "off"}'
      + (f' (h {tn_nlm_hfac} x sigma)' if tn_nlm else '') + f'  ->  background {tn_bg or "off"}'
      + (f' ({tn_bg_um:g} µm' + (f', height {tn_bg_height:g})' if tn_bg == 'ellipsoid' else ')') if tn_bg else '')
      + f'  ->  mask >= {tn_thresh}   ({_time.time() - _tn_t0:.0f} s)')
tifffile.imwrite(os.path.join(tn_dir, f'{FName}_tune_norm.tif'), tn_norm)
tifffile.imwrite(os.path.join(tn_dir, f'{FName}_tune_mask.tif'), tn_mask.astype(np.uint8))
with open(os.path.join(tn_dir, f'{FName}_tune_mask.json'), 'w') as _f:
    json.dump(dict(src=tn_mask_src, smooth_um=list(tn_smooth_um), nlm=tn_nlm, nlm_hfac=tn_nlm_hfac,
                   nlm_patch=tn_nlm_patch, nlm_dist=tn_nlm_dist, bg=tn_bg, bg_um=tn_bg_um,
                   bg_height=tn_bg_height, thresh=tn_thresh), _f, indent=1)

# ── score against Cellpose cells: the 2t-3 full run when present, else Step 2 cp_masks ──
_tn_cells, _tn_cells_name = globals().get('tn_cp_full'), '2t-3 full run (tn_cp_full)'
if _tn_cells is None or _tn_cells.shape != tn_norm.shape:
    _tn_cells, _tn_cells_name = cp_masks, 'Step 2 cp_masks'
if _tn_cells is not None and _tn_cells.max() > 0:
    _tn_in  = _tn_cells > 0
    _tn_ids = _tn_cells[_tn_in]
    _tn_n   = np.bincount(_tn_ids)[1:]
    _tn_bgv = ~_tn_in & _tn_cov

    def _tn_score(m):
        f = np.bincount(_tn_ids, weights=m[_tn_in].astype(np.float64))[1:][_tn_n > 0] / _tn_n[_tn_n > 0]
        return {'mask (% of volume)': 100 * m[_tn_cov].mean(), 'cell voxels in mask (%)': 100 * m[_tn_in].mean(),
                'per-cell median (%)': 100 * np.median(f), 'cells <50% in mask (%)': 100 * np.mean(f < 0.5),
                'non-cell voxels in mask (%)': 100 * m[_tn_bgv].mean()}

    _tn_rows = [dict(variant=f'Step 2 mask3d (thresh_norm {thresh_norm})', **_tn_score(mask3d))]
    _tn_rows += [dict(variant=f'tuned >= {t:g}' + ('  <- tn_thresh' if t == tn_thresh else ''),
                      **_tn_score((tn_norm >= t) & _tn_cov)) for t in sorted(set(tn_thresh_scan) | {tn_thresh})]
    print(f'Scored against {_tn_cells_name}: {int(_tn_cells.max())} cells')
    display(pd.DataFrame(_tn_rows).set_index('variant').round(1))
else:
    _tn_cells = None
    print('No Cellpose cells to score against (Step 2 skipped Cellpose, no 2t-3 full run)')

# ── views: crop at tn_zv ──────────────────────────────────────────────────────
_tn_srcv = (tn_ref if tn_mask_src == 'photonsum' else vol_sum3d)[tn_zv][tn_crop]
_tn_lab  = None if _tn_cells is None else _tn_cells[tn_zv][tn_crop]
fig, axs = plt.subplots(1, 4, figsize=(22, 5.8))
for _a, _img, _ttl in [(axs[0], _tn_rgb(_tn_srcv), f'input: {tn_mask_src}'),
                       (axs[1], _tn_rgb(tn_norm[tn_zv][tn_crop]), 'tuned pre-processing (tn_norm)'),
                       (axs[2], _tn_rgb(_tn_srcv, _tn_lab, tn_mask[tn_zv][tn_crop]), f'tn_mask >= {tn_thresh} (cyan)'),
                       (axs[3], _tn_rgb(_tn_srcv, _tn_lab, mask3d[tn_zv][tn_crop]), 'Step 2 mask3d (cyan)')]:
    _a.imshow(_img, interpolation='nearest')
    _a.set_title(_ttl, fontsize=10)
    _a.axis('off')
plt.suptitle(f'{FName}  —  Step 2t-2: mask pre-processing  z={tn_zv}'
             + (f'  (red: {_tn_cells_name})' if _tn_cells is not None else ''), fontsize=9)
plt.tight_layout()
plt.show()
"""

T3 = r"""# ── Step 2t-3: Cellpose grid on the crop (sandbox: cp_masks untouched) ───────
tn_cp_input  = 'photonsum'        # 'photonsum' (2t-1), 'step2' (the Step 2 Cellpose input), 'tuned' (tn_norm of 2t-2)
tn_cp_sigma  = (1.0, 0.7, 0.7)    # light Gaussian of the Cellpose input (z, y, x) µm (Step 2: (1.0, 0.7, 0.7))
tn_diameters = (8, 10, 12)       # µm: rows of the grid
tn_cellprobs = (-1.5, 0.0)        # columns (x flows): lower -> more / larger objects (Cellpose default 0)
tn_flows     = (0.4, 0.0)         # columns (x cellprobs): Cellpose drops every mask whose flow error > this value;
                                  #  any value >= 0: 0 = check off, higher = more lenient (Cellpose default 0.4).
                                  #  Slice3: all masks have errors 0.75-0.92 -> < 0.7 keeps ~none (NaN stats), >= 1 = off
tn_cp_fixed  = dict(mode='2d_stitch', stitch_threshold=0.25, min_volume_um3=None, model_path=None)
tn_run_grid  = True               # False: skip the grid (e.g. only the full run)
tn_run_full  = False              # True: tn_full_settings on the whole volume -> tn_cp_full (then re-run 2t-2 to score)
tn_full_settings = dict(diameter_um=10.0, cellprob_threshold=-1.5, flow_threshold=0.0)

_tn_cp_src = {'photonsum': tn_ref, 'step2': vol_seg3d, 'tuned': globals().get('tn_norm')}[tn_cp_input]
if _tn_cp_src is None:
    raise RuntimeError("tn_cp_input='tuned' needs Step 2t-2 first")
_tn_inp = md.seg_input(_tn_cp_src, _tn_sp, sigma_um=tn_cp_sigma)   # whole volume (no crop-edge effects), then cropped

if tn_run_grid:
    _tn_ci  = _tn_inp[(slice(None),) + tn_crop]
    _tn_cols = [(float(c), float(f)) for c in tn_cellprobs for f in tn_flows]
    _tn_set  = [dict(tn_cp_fixed, diameter_um=float(d), cellprob_threshold=c, flow_threshold=f)
                for d in tn_diameters for c, f in _tn_cols]
    print(f'Cellpose grid: {len(_tn_set)} settings on the crop {_tn_ci.shape} (input {tn_cp_input})', flush=True)
    tn_grid = md.cellpose_grid(_tn_ci, _tn_sp, _tn_set, gpu=cp_gpu)

    _nr, _nc = len(tn_diameters), len(_tn_cols) + 1
    fig, axs = plt.subplots(_nr, _nc, figsize=(3.8 * _nc, 3.9 * _nr), squeeze=False)
    for _a in axs.ravel():
        _a.axis('off')
    _tn_img = _tn_ci[tn_zv]
    axs[0, 0].imshow(_tn_rgb(_tn_img), interpolation='nearest')
    axs[0, 0].set_title(f'input: {tn_cp_input}  z={tn_zv}', fontsize=9)
    if cp_masks is not None and _nr > 1:
        _cm = cp_masks[(slice(None),) + tn_crop]
        axs[1, 0].imshow(_tn_rgb(_tn_img, _cm[tn_zv]), interpolation='nearest')
        axs[1, 0].set_title(f'Step 2 cp_masks: {len(np.unique(_cm[_cm > 0]))} cells in crop', fontsize=9)
    for _k, _r in enumerate(tn_grid):
        _i, _j = divmod(_k, len(_tn_cols))
        _s, _st = _r['settings'], _r['stats']
        _nz_k = len(np.unique(_r['masks'][tn_zv])) - 1
        axs[_i, _j + 1].imshow(_tn_rgb(_tn_img, _r['masks'][tn_zv]), interpolation='nearest')
        axs[_i, _j + 1].set_title(f'd {_s["diameter_um"]:g} µm  cellprob {_s["cellprob_threshold"]:g}  '
                                  f'flow {_s["flow_threshold"]:g}\n{_nz_k} on this plane, {_st["n"]} in crop  '
                                  f'(xy {_st["diam_xy_um"]:.1f} µm)', fontsize=9)
    plt.suptitle(f'{FName}  —  Step 2t-3: Cellpose on the crop  ({tn_cp_fixed["mode"]}, sigma {tn_cp_sigma} µm)',
                 fontsize=9)
    plt.tight_layout()
    plt.show()
    display(pd.DataFrame([dict(diameter_um=_r['settings']['diameter_um'], cellprob=_r['settings']['cellprob_threshold'],
                               flow=_r['settings']['flow_threshold'], cells_in_crop=_r['stats']['n'],
                               cells_on_plane=len(np.unique(_r['masks'][tn_zv])) - 1, diam_xy_um=round(_r['stats']['diam_xy_um'], 1),
                               z_extent_um=round(_r['stats']['z_extent_um'], 1), seconds=round(_r['seconds']))
                          for _r in tn_grid]))

if tn_run_full:   # whole volume -> tn_cp_full (reused while settings and input are unchanged)
    _tn_st   = dict(tn_cp_fixed, **tn_full_settings)
    _tn_fp   = os.path.join(tn_dir, f'{FName}_tune_cellpose_masks.tif')
    _tn_fj   = _tn_fp.replace('.tif', '.json')
    _tn_key  = dict(_tn_st, input=tn_cp_input, sigma_um=list(tn_cp_sigma))
    _tn_srcf = {'photonsum': _tn_ref_path, 'tuned': os.path.join(tn_dir, f'{FName}_tune_norm.tif'),
                'step2': stitch_oir if seg_source == 'olympus' else stitch['int_path']}[tn_cp_input]
    if (os.path.exists(_tn_fp) and os.path.exists(_tn_fj) and json.load(open(_tn_fj)) == _tn_key
            and os.path.getmtime(_tn_fp) > os.path.getmtime(_tn_srcf)):
        tn_cp_full = tifffile.imread(_tn_fp)
        print(f'Reloaded tn_cp_full <- {os.path.basename(_tn_fp)}')
    else:
        print(f'Cellpose on the whole volume {_tn_inp.shape}: {tn_full_settings} ...', flush=True)
        _tn_t0 = _time.time()
        tn_cp_full = md.segment_cellpose(_tn_inp, _tn_sp, gpu=cp_gpu, **_tn_st)
        tifffile.imwrite(_tn_fp, tn_cp_full)
        with open(_tn_fj, 'w') as _f:
            json.dump(_tn_key, _f, indent=1)
        print(f'Done in {_time.time() - _tn_t0:.0f} s  -> {_tn_fp}')
    _tn_s2 = md.mask_stats(tn_cp_full, _tn_sp)
    print(f'tn_cp_full: {_tn_s2["n"]} cells  |  median xy diameter {_tn_s2["diam_xy_um"]:.1f} µm  |  '
          f'median z extent {_tn_s2["z_extent_um"]:.1f} µm   (Step 2 cp_masks: {n_cells} cells)')
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    axs[0].imshow(_tn_rgb(md.seg_input(vol_seg3d[tn_zv:tn_zv + 1], _tn_sp)[0],
                          None if cp_masks is None else cp_masks[tn_zv]), interpolation='nearest')
    axs[0].set_title(f'Step 2: cp_masks ({n_cells} cells)  z={tn_zv}', fontsize=10)
    axs[1].imshow(_tn_rgb(_tn_inp[tn_zv], tn_cp_full[tn_zv]), interpolation='nearest')
    axs[1].set_title(f'tuned: {tn_cp_input}, d {_tn_st["diameter_um"]:g} µm, cellprob '
                     f'{_tn_st.get("cellprob_threshold", 0):g}, flow {_tn_st.get("flow_threshold", 0.4):g}  '
                     f'({_tn_s2["n"]} cells)  z={tn_zv}', fontsize=10)
    for _a in axs:
        _a.axis('off')
    plt.tight_layout()
    plt.show()
"""

NEW = [('md-2t', 'markdown', MD), ('code-2t-ref', 'code', T1), ('code-2t-mask', 'code', T2), ('code-2t-cp', 'code', T3),
       ('code-2t-qc', 'code', T4)]

# comment-only fixes in cells the user edits (values kept): (cell id, old text, new text)
COMMENT_FIXES = [('code-2t-cp',
                  "# columns (x cellprobs): flow-error QC, masks with a larger error are dropped;\n"
                  "                                  #  0 = QC off (Cellpose default 0.4 drops most cells here)",
                  "# columns (x cellprobs): Cellpose drops every mask whose flow error > this value;\n"
                  "                                  #  any value >= 0: 0 = check off, higher = more lenient (Cellpose default 0.4).\n"
                  "                                  #  Slice3: all masks have errors 0.75-0.92 -> < 0.7 keeps ~none (NaN stats), >= 1 = off")]


def make_cell(cid, kind, src):
    c = dict(cell_type=kind, id=cid, metadata={}, source=src.strip('\n').splitlines(keepends=True))
    if kind == 'code':
        c = dict(cell_type='code', execution_count=None, id=cid, metadata={}, outputs=[], source=c['source'])
    return c


def main(only=None, nb_path=NB):
    """(Re)place the NEW cells; `only` = list of cell ids to touch (others, e.g. the user's edited parameter
    cells, are left as they are).  A missing cell is inserted after its predecessor in NEW (or after AFTER).
    Then the COMMENT_FIXES are applied to cells that were not replaced."""
    nb = json.load(open(nb_path, encoding='utf-8'))
    cells = nb['cells']
    if AFTER not in [c.get('id') for c in cells]:
        sys.exit(f'cell {AFTER} not found')
    order = [n[0] for n in NEW]
    done = []
    for cid, kind, src in NEW:
        if only is not None and cid not in only:
            continue
        ids = [c.get('id') for c in cells]
        new = make_cell(cid, kind, src)
        if cid in ids:
            cells[ids.index(cid)] = new
            done.append(f'{cid} (replaced)')
        else:
            prev = next((p for p in reversed(order[:order.index(cid)]) if p in ids), AFTER)
            cells.insert(ids.index(prev) + 1, new)
            done.append(f'{cid} (inserted after {prev})')
    for cid, old, new in COMMENT_FIXES:
        hit = [c for c in cells if c.get('id') == cid]
        if not hit or (only is None or cid in only):
            continue
        src = ''.join(hit[0]['source'])
        if old in src:
            hit[0]['source'] = src.replace(old, new).splitlines(keepends=True)
            done.append(f'{cid} (comment updated, values kept)')
        elif new not in src:
            done.append(f'{cid} (comment NOT found, left as is)')
    nb['cells'] = cells
    with open(nb_path, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(nb, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print('\n'.join(done))


if __name__ == '__main__':
    # python patch_time_seg_tuning.py                    -> all NEW cells (overwrites edited parameters!)
    # python patch_time_seg_tuning.py code-2t-qc md-2t   -> only these
    main(sys.argv[1:] or None)
