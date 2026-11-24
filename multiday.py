"""
multiday.py
Multi-day cell tracking for tiled XYTZ FLIM sessions (AnalyzeTCSPC_XYTZTile.ipynb, MultiDay_Reference.ipynb).

Cells are segmented once per subject on a reference volume (dedicated intensity stack, or a session's
stitched intensity as interim reference) and keep a global ID.  Every session's stitched intensity is
registered to the reference in physical units (µm); the reference labels are then resampled into the
session grid (nearest neighbour), so the session's TCSPC data are used as acquired, never resampled.

Volumes are ZYX; spacing is (dz, dy, dx) in µm throughout.

Functions
---------
load_volume          -- ZYX float32 volume + spacing
preprocess_volume    -- Step 2 mask pre-processing: NLM (3D / 2D) + background (rolling ball / ellipsoid / Gaussian)
seg_input            -- light smoothing of the raw intensity for Cellpose (no NLM, no clipping)
segment_cellpose     -- Cellpose 4 (cpsam) in 3D or 2D+stitch, sizes given in µm
cell_quality         -- per-cell contrast to its surround + photons / recorded fraction per timepoint
mask_stats           -- count / size / z-extent summary of a label volume
cellpose_grid        -- run several Cellpose settings on a crop, return masks + stats (tuning)
plot_cellpose_grid   -- montage of the cellpose_grid results
training_crops / export_training_set / finetune_cellpose / eval_crops -- Cellpose fine-tuning on curated
                        2D crops (Cellpose_FineTune.ipynb); segment_cellpose(model_path=...) uses the result
reg_prep             -- registration contrast (smoothing, large-scale background removal, 0..1)
coarse_align         -- session -> reference start: MIP phase correlation + rotation sweep + depth search
register_session     -- SimpleITK rigid -> affine -> B-spline, session (fixed) to reference (moving)
save_transform / load_transform -- transform file + JSON summary
warp_to_session      -- resample a reference-space volume (intensity or labels) onto the session grid
refine_cells         -- per-cell local shift search (NCC) after label transfer
cell_qc              -- per-cell in-FOV / coverage / match flags
plot_registration_qc -- session vs warped reference: MIP overlay + checkerboard
subject_paths        -- file layout of a subject folder
save_reference_meta / load_reference -- reference volume source + spacing, curated labels
reference_cells_table-- cell_id, voxels, volume and centroid (µm) of a label volume
session_date         -- acquisition date from the path (YYYYMMDD) or the file time
update_sessions      -- add / replace a session entry in <subject>/sessions.json
collect_subject      -- per-cell results of all sessions -> long table (cells_long.csv)
plot_subject         -- per-cell values across sessions (trajectories + heatmap)

Registration convention: the session is the fixed image and the reference the moving one, so the
transform T maps session points to reference points (µm).  Resampling reference labels with T onto the
session grid is then direct (no inversion of the B-spline needed).
"""

import os
import json
import numpy as np
import tifffile
from scipy import ndimage as ndi


# ── volumes ───────────────────────────────────────────────────────────────────
def load_volume(path, spacing_zyx):
    """ZYX float32 volume from a TIFF + spacing (dz, dy, dx) in µm."""
    vol = tifffile.imread(path).astype(np.float32)
    vol = np.squeeze(vol)
    if vol.ndim != 3:
        raise ValueError(f'{os.path.basename(path)}: expected a ZYX volume, got shape {vol.shape}')
    return vol, tuple(float(s) for s in spacing_zyx)


def preprocess_volume(vol, nlm_h=None, patch_size=5, patch_distance=6, fast_mode=True, rb_radius=30,
                      verbose=True, nlm='3d', nlm_hfac=0.8, bg='ball', bg_height=1.0, smooth_px=None, valid=None):
    """
    Mask pre-processing: percentile normalisation (1-99 %) -> optional Gaussian -> non-local means ->
    per-plane background subtraction -> percentile normalisation clipped to [0, 1].  Returns vol_norm_3d
    (thresholded for mask3d).

    smooth_px : Gaussian sigma per axis (z, y, x) in px before NLM; None = none
    nlm       : '3d', '2d' (per plane) or None; strength h = nlm_h, else nlm_hfac x the estimated noise sigma
    bg        : 'ball'      skimage rolling ball of radius rb_radius px (on 0..1 data its height is also
                            rb_radius, so a radius-30 ball fits inside ~8 µm cells and hollows them);
                'ellipsoid' rolling ball of radius rb_radius px and height bg_height in data units (0..1):
                            removes the slow background only;
                'gauss'     minus a Gaussian of sigma rb_radius px;  None = no background subtraction
    valid     : boolean volume of recorded voxels: percentiles from these only, output 0 elsewhere
    The defaults reproduce the original Step 2 (3D NLM h 0.8 sigma, ball radius 30 px).
    """
    from skimage.restoration import denoise_nl_means, rolling_ball, estimate_sigma, ellipsoid_kernel
    sel = vol if valid is None else vol[valid]
    lo, hi = np.percentile(sel, 1), np.percentile(sel, 99)
    v = np.clip((vol - lo) / max(hi - lo, 1e-12), 0, 1).astype(np.float32)
    if smooth_px is not None and any(smooth_px):
        v = ndi.gaussian_filter(v, smooth_px)
    if nlm:
        sigma = float(estimate_sigma(v, average_sigmas=True, channel_axis=None))
        h = nlm_hfac * sigma if nlm_h is None else nlm_h
        kw = dict(h=h, patch_size=patch_size, patch_distance=patch_distance, fast_mode=fast_mode, channel_axis=None)
        if verbose:
            print(f'Noise sigma ~ {sigma:.5f}   h = {h:.5f}')
            print(f'Running {nlm.upper()} NLM (CPU)...', end=' ', flush=True)
        if nlm == '3d':
            v = denoise_nl_means(v, **kw).astype(np.float32)
        elif nlm == '2d':
            v = np.stack([denoise_nl_means(p, **kw) for p in v]).astype(np.float32)
        else:
            raise ValueError(f"nlm must be '3d', '2d' or None, not {nlm!r}")
        if verbose:
            print('done.')
    if bg is not None:
        if verbose:
            print(f'Background ({bg}, {rb_radius:.1f} px)...', end=' ', flush=True)
        if bg == 'ball':
            v = np.stack([np.clip(p - rolling_ball(p, radius=rb_radius, workers=-1), 0, None) for p in v])
        elif bg == 'ellipsoid':
            k = ellipsoid_kernel((2 * int(round(rb_radius)) + 1,) * 2, bg_height)
            v = np.stack([np.clip(p - rolling_ball(p, kernel=k, workers=-1), 0, None) for p in v])
        elif bg == 'gauss':
            v = np.clip(v - ndi.gaussian_filter(v, (0, rb_radius, rb_radius)), 0, None)
        else:
            raise ValueError(f"bg must be 'ball', 'ellipsoid', 'gauss' or None, not {bg!r}")
        if verbose:
            print('done.')
    sel = v if valid is None else v[valid]
    lo, hi = np.percentile(sel, 1), np.percentile(sel, 99)
    out = np.clip((v - lo) / max(hi - lo, 1e-12), 0, 1).astype(np.float32)
    if valid is not None:
        out[~valid] = 0
    return out


def seg_input(vol, spacing, sigma_um=(1.0, 0.7, 0.7), bg_um=None):
    """
    Cellpose input: the raw intensity with light Gaussian smoothing (sigma in µm per axis), optionally
    minus a large-scale per-plane background (Gaussian of bg_um).  No clipping: Cellpose normalises itself.
    """
    sig = [s / d for s, d in zip(sigma_um, spacing)]
    out = ndi.gaussian_filter(vol.astype(np.float32), sig)
    if bg_um:
        bg = ndi.gaussian_filter(out, (0, bg_um / spacing[1], bg_um / spacing[2]))
        out = out - bg
        out -= out.min()
    return out


# ── Cellpose ──────────────────────────────────────────────────────────────────
_CP_MODELS = {}


def _cp_model(gpu=True, model_path=None):
    """Cached CellposeModel: the generic cpsam, or a fine-tuned model file (reloaded when the file changes)."""
    key = (gpu, model_path, os.path.getmtime(model_path) if model_path else None)
    if key not in _CP_MODELS:
        from cellpose import models
        _CP_MODELS[key] = models.CellposeModel(gpu=gpu, pretrained_model=model_path or 'cpsam')
    return _CP_MODELS[key]


def segment_cellpose(vol, spacing, diameter_um, mode='3d', cellprob_threshold=0.0, flow_threshold=0.4,
                     min_volume_um3=None, stitch_threshold=0.25, flow3D_smooth=0, gpu=True, model_path=None):
    """
    Cellpose 4 on a ZYX volume with object sizes in µm: the generic cpsam, or a model fine-tuned with
    finetune_cellpose (model_path; trained at the scale diameter_um -> 30 px used here as well).

    mode '3d'        : do_3D with anisotropy = dz/dx (flow_threshold is not used by Cellpose in 3D)
    mode '2d_stitch' : 2D per plane, masks linked across z when IoU > stitch_threshold
    min_volume_um3   : smallest object kept; default = sphere of half the expected diameter.
    Returns uint16 label volume.
    """
    dz, dy, dx = spacing
    diam_px = diameter_um / dx
    if min_volume_um3 is None:
        min_volume_um3 = np.pi / 6 * (0.5 * diameter_um) ** 3
    min_size = max(1, int(round(min_volume_um3 / (dz * dy * dx))))
    kw = dict(diameter=diam_px, cellprob_threshold=cellprob_threshold, z_axis=0, channel_axis=None)
    model = _cp_model(gpu, model_path)
    if mode == '3d':
        masks = model.eval(vol, do_3D=True, anisotropy=dz / dx, flow3D_smooth=flow3D_smooth,
                           min_size=min_size, **kw)[0]
    elif mode == '2d_stitch':
        # per-plane min size: the cross-section of the smallest object
        min_area = max(1, int(round(np.pi / 4 * (0.5 * diameter_um) ** 2 / (dy * dx))))
        masks = model.eval(vol, do_3D=False, stitch_threshold=stitch_threshold,
                           flow_threshold=flow_threshold, min_size=min_area, **kw)[0]
        masks = _remove_small(masks, min_size)
    else:
        raise ValueError(f"mode must be '3d' or '2d_stitch', got {mode!r}")
    return masks.astype(np.uint16)


def _remove_small(masks, min_size):
    """Drop labels below min_size voxels and renumber 1..n."""
    counts = np.bincount(masks.ravel())
    keep = counts >= min_size
    keep[0] = False
    lut = np.zeros(counts.size, np.int64)
    lut[keep] = np.arange(1, keep.sum() + 1)
    return lut[masks]


# ── Cellpose fine-tuning (Cellpose_FineTune.ipynb) ────────────────────────────
# Training crops are 2D planes of the Cellpose input upscaled by 30 / diameter_px, i.e. at the scale
# segment_cellpose feeds the network (Cellpose rescales its input so the given diameter becomes 30 px).
def training_crops(vol, spacing, n=16, crop_um=128.0, min_dist_um=80.0, min_valid=0.95, z_margin=1):
    """
    (z, y0, x0) of n square crops (crop_um wide) with the most cell-scale structure, spread over the volume:
    band-pass energy per crop, picked greedily with at least min_dist_um between crop centres (z in µm too).
    Crops with less than min_valid of non-zero voxels (outside the stitched tiles) are skipped.
    """
    dz, dy, dx = spacing
    c = int(round(crop_um / dx))
    v = vol.astype(np.float32)
    band = np.abs(ndi.gaussian_filter(v, (0, 1.0 / dy, 1.0 / dx)) - ndi.gaussian_filter(v, (0, 6.0 / dy, 6.0 / dx)))
    step = max(1, c // 4)
    cand = []
    for z in range(z_margin, v.shape[0] - z_margin):
        for y0 in range(0, v.shape[1] - c + 1, step):
            for x0 in range(0, v.shape[2] - c + 1, step):
                if np.mean(v[z, y0:y0 + c, x0:x0 + c] > 0) < min_valid:
                    continue
                cand.append((float(band[z, y0:y0 + c, x0:x0 + c].mean()), z, y0, x0))
    cand.sort(reverse=True)
    picked = []
    for s, z, y0, x0 in cand:
        p = np.array([z * dz, (y0 + c / 2) * dy, (x0 + c / 2) * dx])
        if all(np.linalg.norm(p - q) >= min_dist_um for q in (pp for _, pp in picked)):
            picked.append(((z, y0, x0), p))
            if len(picked) == n:
                break
    return [k for k, _ in picked], c


def export_training_set(vol, spacing, crops, crop_px, diameter_um, cellprob_threshold=0.0, gpu=True,
                        model_path=None, sigma_um=(1.0, 0.7, 0.7)):
    """
    Training images + starting masks for curation: each crop of seg_input(vol) resized (linear) to
    out_px = 30 * crop width / diameter (so the expected diameter is 30 px, whatever the pixel size), then
    segmented by the current model at that scale.  Returns (images (n, out_px, out_px) float32,
    masks (n, out_px, out_px) uint16, scale = out_px / crop_px).
    """
    from skimage.transform import resize
    out_px = int(round(30.0 * crop_px * spacing[2] / diameter_um))
    inp = seg_input(vol, spacing, sigma_um=sigma_um)
    model = _cp_model(gpu, model_path)
    imgs, masks = [], []
    for z, y0, x0 in crops:
        im = resize(inp[z, y0:y0 + crop_px, x0:x0 + crop_px], (out_px, out_px), order=1,
                    preserve_range=True, anti_aliasing=False).astype(np.float32)
        imgs.append(im)
        masks.append(model.eval(im, diameter=30.0, cellprob_threshold=cellprob_threshold,
                                channel_axis=None)[0].astype(np.uint16))
    return np.stack(imgs), np.stack(masks), out_px / crop_px


def finetune_cellpose(imgs, masks, test_imgs, test_masks, save_dir, model_name, n_epochs=200, learning_rate=1e-5,
                      weight_decay=0.1, batch_size=4, base_model='cpsam', gpu=True):
    """Fine-tune Cellpose 4 on curated 2D crops (export_training_set scale).  The model is written to
    <save_dir>/models/<model_name>.  Returns dict(path, train_loss, test_loss)."""
    from cellpose import models, train
    os.makedirs(save_dir, exist_ok=True)          # train_seg only creates <save_dir>/models
    net = models.CellposeModel(gpu=gpu, pretrained_model=base_model).net
    path, tr, te = train.train_seg(net, train_data=list(imgs), train_labels=list(masks),
                                   test_data=list(test_imgs), test_labels=list(test_masks), channel_axis=None,
                                   n_epochs=n_epochs, learning_rate=learning_rate, weight_decay=weight_decay,
                                   batch_size=batch_size, save_path=save_dir, model_name=model_name,
                                   min_train_masks=1)
    return dict(path=str(path), train_loss=np.asarray(tr).tolist(), test_loss=np.asarray(te).tolist())


def eval_crops(imgs, model_path=None, cellprob_threshold=0.0, gpu=True):
    """Masks of each 2D crop (export_training_set scale) by the generic or a fine-tuned model."""
    model = _cp_model(gpu, model_path)
    return [model.eval(im, diameter=30.0, cellprob_threshold=cellprob_threshold, channel_axis=None)[0]
            for im in imgs]


def cell_quality(labels, ref, tile_int_t, positions, xy_um, recorded=None, ring_um=3.0, min_fov=0.9,
                 min_photons=500):
    """
    Per-cell quality of a label volume on the time-series canvas (AnalyzeTCSPC_XYTZTime Step 2).

    contrast       : mean of `ref` in the cell / median of `ref` in a ring of ring_um around it (in-plane; other
                     cells and unrecorded voxels left out); NaN when the ring is empty (cell surrounded by cells)
    photons / fov  : per timepoint, the photons in the cell (tile overlaps summed, as the Step 0d TCSPC stacks)
                     and the fraction of its voxels recorded (tile_int_t planes with photons)
    tp_in_fov      : timepoints with fov >= min_fov;  tp_usable: those with >= min_photons too

    labels     : (z, y, x) int label volume on the canvas;  ref: (z, y, x) intensity on the same canvas
    tile_int_t : (n_t, n_tiles, z, y, x) photons;  positions: (n_t, n_tiles, 3) canvas origins
    recorded   : boolean volume of recorded voxels (ring), default ref > 0
    Returns DataFrame (one row per label present: cell_id, z_centre, n_planes, d_xy_um, contrast, tp_in_fov,
    tp_usable, photons_median, photons_min over the in-FOV timepoints) and dict(photons, fov) of
    (n_t, n_labels) arrays indexed by label - 1.
    """
    import warnings
    import pandas as pd
    from skimage.morphology import disk
    n = int(labels.max())
    ids = np.arange(1, n + 1)
    flat = labels.ravel()
    vox = np.bincount(flat, minlength=n + 1)[1:]
    has = vox > 0
    rec = (ref > 0) if recorded is None else recorded

    objs = ndi.find_objects(labels, max_label=n)
    zc = np.array([(o[0].start + o[0].stop - 1) / 2 if o else np.nan for o in objs])
    nzp = np.array([o[0].stop - o[0].start if o else 0 for o in objs])
    area = np.zeros(n + 1)
    for z in range(labels.shape[0]):
        np.maximum(area, np.bincount(labels[z].ravel(), minlength=n + 1), out=area)
    dxy = 2 * np.sqrt(area[1:] * xy_um ** 2 / np.pi)

    fp = disk(max(1, int(round(ring_um / xy_um))))
    ring = np.zeros_like(labels)
    for z in range(labels.shape[0]):
        ring[z] = np.where((labels[z] == 0) & rec[z], ndi.grey_dilation(labels[z], footprint=fp), 0)
    nring = np.bincount(ring.ravel(), minlength=n + 1)[1:]
    with warnings.catch_warnings(), np.errstate(all='ignore'):
        warnings.simplefilter('ignore')
        con = (np.asarray(ndi.mean(ref, labels, ids), float)
               / np.maximum(np.asarray(ndi.median(ref, ring, ids), float), 1e-9))
    con[(nring == 0) | ~has] = np.nan

    tile_int_t = np.asarray(tile_int_t)
    pos = np.asarray(positions, int).reshape(tile_int_t.shape[0], tile_int_t.shape[1], 3)
    shp = np.array(tile_int_t.shape[2:])
    ok = tile_int_t.reshape(*tile_int_t.shape[:3], -1).max(axis=3) > 0
    n_t = tile_int_t.shape[0]
    ph = np.zeros((n_t, n))
    fov = np.zeros((n_t, n))
    for t in range(n_t):
        cv = np.zeros(labels.shape, bool)
        for p in range(tile_int_t.shape[1]):
            z0, y0, x0 = pos[t, p]
            sl = (slice(z0, z0 + shp[0]), slice(y0, y0 + shp[1]), slice(x0, x0 + shp[2]))
            ph[t] += np.bincount(labels[sl].ravel(), weights=tile_int_t[t, p].ravel(), minlength=n + 1)[1:]
            cv[sl] |= ok[t, p][:, None, None]
        fov[t] = np.bincount(flat, weights=cv.ravel(), minlength=n + 1)[1:] / np.maximum(vox, 1)
    infov = fov >= min_fov
    usable = infov & (ph >= min_photons)
    phin = np.where(infov, ph, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        phmed, phmin = np.nanmedian(phin, axis=0), np.nanmin(phin, axis=0)
    df = pd.DataFrame(dict(cell_id=ids, z_centre=zc, n_planes=nzp, d_xy_um=dxy.round(1), contrast=con.round(3),
                           tp_in_fov=infov.sum(0), tp_usable=usable.sum(0), photons_median=phmed.round(0),
                           photons_min=phmin.round(0)))[has].reset_index(drop=True)
    return df, dict(photons=ph, fov=fov)


def mask_stats(masks, spacing):
    """Count, median equivalent xy diameter (µm), median z extent (µm) and volume (µm³) of the labels."""
    dz, dy, dx = spacing
    objs = [s for s in ndi.find_objects(masks) if s is not None]
    if not objs:
        return dict(n=0, diam_xy_um=np.nan, z_extent_um=np.nan, volume_um3=np.nan)
    vox = np.bincount(masks.ravel())[1:]
    vox = vox[vox > 0]
    zext = np.array([s[0].stop - s[0].start for s in objs])
    # equivalent xy diameter from the largest cross-section of each object
    xy_area = np.zeros(masks.max() + 1)
    for z in range(masks.shape[0]):
        np.maximum(xy_area, np.bincount(masks[z].ravel(), minlength=xy_area.size), out=xy_area)
    xy_area = xy_area[1:][xy_area[1:] > 0]
    return dict(n=len(objs),
                diam_xy_um=float(np.median(2 * np.sqrt(xy_area * dy * dx / np.pi))),
                z_extent_um=float(np.median(zext) * dz),
                volume_um3=float(np.median(vox) * dz * dy * dx))


def cellpose_grid(vol, spacing, settings, gpu=True, verbose=True):
    """
    Run segment_cellpose for each settings dict on `vol` (a crop); returns a list of
    dict(settings=..., masks=..., stats=..., seconds=...).
    """
    import time
    out = []
    for k, st in enumerate(settings):
        t0 = time.perf_counter()
        m = segment_cellpose(vol, spacing, gpu=gpu, **st)
        sec = time.perf_counter() - t0
        stats = mask_stats(m, spacing)
        out.append(dict(settings=st, masks=m, stats=stats, seconds=sec))
        if verbose:
            print(f'[{k + 1}/{len(settings)}] {st}  ->  {stats["n"]} cells, xy {stats["diam_xy_um"]:.1f} µm, '
                  f'z {stats["z_extent_um"]:.1f} µm  ({sec:.0f} s)')
    return out


def plot_cellpose_grid(img, results, z_list, title=''):
    """Montage: rows = settings, columns = z planes; outlines over the (percentile-scaled) image."""
    import matplotlib.pyplot as plt
    from skimage.segmentation import find_boundaries
    nr, nc = len(results), len(z_list)
    fig, axs = plt.subplots(nr, nc, figsize=(3.6 * nc, 3.7 * nr), squeeze=False)
    for i, r in enumerate(results):
        for j, z in enumerate(z_list):
            im = img[z]
            im = np.clip((im - np.percentile(im, 1)) / (np.percentile(im, 99.7) - np.percentile(im, 1)), 0, 1)
            rgb = np.dstack([im, im, im])
            rgb[find_boundaries(r['masks'][z], mode='inner')] = (1.0, 0.25, 0.1)
            axs[i, j].imshow(rgb, interpolation='nearest')
            axs[i, j].set_xticks([]); axs[i, j].set_yticks([])
            if j == 0:
                s, st = r['settings'], r['stats']
                axs[i, j].set_ylabel(f"{s.get('mode', '3d')} d={s['diameter_um']:g}µm cp={s.get('cellprob_threshold', 0):g}\n"
                                     f"n={st['n']} xy={st['diam_xy_um']:.0f} z={st['z_extent_um']:.0f}µm",
                                     fontsize=8)
            if i == 0:
                axs[i, j].set_title(f'z={z}', fontsize=9)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig


# ── registration: contrast, coarse start ──────────────────────────────────────
def reg_prep(vol, spacing, smooth_um=1.5, bg_um=40.0, pct=(1, 99.5)):
    """Registration contrast: Gaussian smoothing (µm), minus large-scale per-plane background, -> 0..1."""
    v = ndi.gaussian_filter(vol.astype(np.float32), [smooth_um / s for s in spacing])
    if bg_um:
        v = v - ndi.gaussian_filter(v, (0, bg_um / spacing[1], bg_um / spacing[2]))
    lo, hi = np.percentile(v, pct)
    return np.clip((v - lo) / max(hi - lo, 1e-12), 0, 1).astype(np.float32)


def _ncc(a, b):
    a = a - a.mean()
    b = b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else 0.0


def _rot2(theta):
    """Rotation matrix acting on (y, x) coordinates."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def _bin_xy(vol, f):
    """Block-mean binning of the y/x axes by integer factor f (trailing pixels dropped)."""
    if f <= 1:
        return vol.astype(np.float32)
    nz, ny, nx = vol.shape
    v = vol[:, :ny // f * f, :nx // f * f].astype(np.float32)
    return v.reshape(nz, ny // f, f, nx // f, f).mean(axis=(2, 4))


def coarse_align(ref, ref_spacing, mov, mov_spacing, mov_mask=None, max_rot_deg=10.0, rot_step_deg=1.0,
                 px_um=4.0, overlap_ratio=0.3, verbose=True):
    """
    Start transform session -> reference, p_ref = M @ p_mov + t (µm, ZYX), from
      xy : masked phase correlation of max projections at px_um, for every rotation about z in the sweep;
      z  : correlation of session planes with reference planes (warped into the session xy frame) vs depth offset.
    `ref` / `mov` should be reg_prep outputs.  Returns dict(M, t, theta_deg, score_xy, score_z).
    """
    from skimage.registration import phase_cross_correlation
    fr = max(1, int(round(px_um / ref_spacing[1])))
    fm = max(1, int(round(px_um / mov_spacing[1])))
    pr, pm = ref_spacing[1] * fr, mov_spacing[1] * fm            # actual coarse pixel sizes
    Rz = _bin_xy(ref, fr)
    Sz = _bin_xy(mov, fm)
    Smz = np.ones(Sz.shape, bool) if mov_mask is None else _bin_xy(mov_mask.astype(np.float32), fm) > 0.5
    if abs(pm - pr) > 1e-6:                                         # session to the reference pixel size
        Sz = ndi.zoom(Sz, (1, pm / pr, pm / pr), order=1)
        Smz = ndi.zoom(Smz.astype(np.uint8), (1, pm / pr, pm / pr), order=0).astype(bool)
    R, S, Sm = Rz.max(axis=0), Sz.max(axis=0), Smz.any(axis=0)
    h, w = S.shape
    D = int(np.ceil(np.hypot(h, w)))
    cS, cO = np.array([(h - 1) / 2, (w - 1) / 2]), np.array([(D - 1) / 2, (D - 1) / 2])
    P = (max(R.shape[0], D), max(R.shape[1], D))
    Rp = np.zeros(P, np.float32); Rp[:R.shape[0], :R.shape[1]] = R
    Rmp = np.zeros(P, bool);      Rmp[:R.shape[0], :R.shape[1]] = True

    best = None
    for th_deg in np.arange(-max_rot_deg, max_rot_deg + 1e-9, rot_step_deg):
        Rot = _rot2(np.deg2rad(th_deg))
        Minv = Rot.T                                               # out(u) = S(cS + Rot^T (u - cO))
        off = cS - Minv @ cO
        St = ndi.affine_transform(S, Minv, offset=off, output_shape=(D, D), order=1)
        Smt = ndi.affine_transform(Sm.astype(np.float32), Minv, offset=off, output_shape=(D, D), order=0) > 0.5
        Sp = np.zeros(P, np.float32); Sp[:D, :D] = St
        Smp = np.zeros(P, bool);      Smp[:D, :D] = Smt
        shift = phase_cross_correlation(Rp, Sp, reference_mask=Rmp, moving_mask=Smp,
                                        overlap_ratio=overlap_ratio)
        if isinstance(shift, tuple):
            shift = shift[0]
        # score: NCC of the overlap at that shift
        Ss = ndi.shift(Sp, shift, order=1)
        Ms = ndi.shift(Smp.astype(np.float32), shift, order=0) > 0.5
        ov = Ms & Rmp
        sc = _ncc(Rp[ov], Ss[ov]) if ov.sum() > 100 else -1.0
        if best is None or sc > best[0]:
            best = (sc, th_deg, Rot, shift)
    sc_xy, th_deg, Rot, shift = best
    b = cO - Rot @ cS + shift                                       # x_R_px = Rot @ x_S_px + b
    t_yx = b * pr

    # depth: warp each reference plane into the session xy frame, then scan the z offset
    W = np.stack([ndi.affine_transform(Rz[k], Rot, offset=b, output_shape=Sz.shape[1:], order=1)
                  for k in range(Rz.shape[0])])
    Wv = ndi.affine_transform(np.ones(Rz.shape[1:], np.float32), Rot, offset=b,
                              output_shape=Sz.shape[1:], order=0) > 0.5
    dzr, dzm = ref_spacing[0], mov_spacing[0]
    zm = np.arange(Sz.shape[0]) * dzm
    step = min(dzr, dzm) / 2
    tz_grid = np.arange(-zm[-1] * 0.7, (Rz.shape[0] - 1) * dzr - zm[-1] * 0.3 + 1e-9, step)
    best_z = (-np.inf, 0.0)
    for tz in tz_grid:
        f = (zm + tz) / dzr
        ok = (f >= 0) & (f <= Rz.shape[0] - 1)
        if ok.sum() < max(2, 0.3 * len(zm)):
            continue
        k0 = np.floor(f[ok]).astype(int)
        k1 = np.minimum(k0 + 1, Rz.shape[0] - 1)
        a = (f[ok] - k0)[:, None, None]
        Wi = (1 - a) * W[k0] + a * W[k1]
        m = Smz[ok] & Wv[None]
        sc = _ncc(Sz[ok][m], Wi[m]) if m.sum() > 100 else -1.0
        if sc > best_z[0]:
            best_z = (sc, tz)
    M = np.eye(3)
    M[1:, 1:] = Rot
    # bin-centre origins: coarse pixel k of an f-binned image sits at (k + (f-1)/2) native pixels
    t = np.array([best_z[1], *(t_yx + (fr - 1) / 2 * ref_spacing[1]
                                - Rot @ np.full(2, (fm - 1) / 2 * mov_spacing[1]))])
    if verbose:
        print(f'coarse: rotation {th_deg:+.1f}°, shift (z, y, x) = ({t[0]:.1f}, {t[1]:.1f}, {t[2]:.1f}) µm, '
              f'NCC xy {sc_xy:.3f}, z {best_z[0]:.3f}')
    return dict(M=M, t=t, theta_deg=float(th_deg), score_xy=float(sc_xy), score_z=float(best_z[0]))


# ── registration: SimpleITK ───────────────────────────────────────────────────
def _to_sitk(vol, spacing, origin=(0.0, 0.0, 0.0)):
    import SimpleITK as sitk
    img = sitk.GetImageFromArray(np.ascontiguousarray(vol))
    img.SetSpacing((float(spacing[2]), float(spacing[1]), float(spacing[0])))
    img.SetOrigin((float(origin[2]), float(origin[1]), float(origin[0])))
    return img


def _working(vol, spacing, xy_um):
    """Volume binned in xy to ~xy_um; returns (array, spacing, origin) with the bin-centre origin."""
    f = max(1, int(round(xy_um / spacing[1])))
    sp = (spacing[0], spacing[1] * f, spacing[2] * f)
    org = (0.0, (f - 1) / 2 * spacing[1], (f - 1) / 2 * spacing[2])
    return _bin_xy(vol, f), sp, org


_P = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], float)   # ZYX <-> XYZ


def _sitk_linear(kind, M_zyx, t_zyx, center_zyx):
    """Euler3D / Affine transform (sitk, XYZ) for p_ref = M p_mov + t (ZYX), rotation centre center_zyx."""
    import SimpleITK as sitk
    M = _P @ np.asarray(M_zyx, float) @ _P
    c = np.asarray(center_zyx, float)[::-1]
    t = np.asarray(t_zyx, float)[::-1]
    T = sitk.Euler3DTransform() if kind == 'rigid' else sitk.AffineTransform(3)
    T.SetCenter(c.tolist())
    T.SetMatrix(M.ravel().tolist())
    T.SetTranslation((t + M @ c - c).tolist())
    return T


def _reg_method(metric, fixed_mask, sampling):
    import SimpleITK as sitk
    R = sitk.ImageRegistrationMethod()
    if metric == 'correlation':
        R.SetMetricAsCorrelation()
    elif metric == 'mattes':
        R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    else:
        raise ValueError(f"metric must be 'correlation' or 'mattes', got {metric!r}")
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(sampling, seed=1)
    if fixed_mask is not None:
        R.SetMetricFixedMask(fixed_mask)
    R.SetInterpolator(sitk.sitkLinear)
    R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    return R


def _ncc_after(fixed, moving, T, mask_np):
    import SimpleITK as sitk
    w = sitk.GetArrayFromImage(sitk.Resample(moving, fixed, T, sitk.sitkLinear, 0.0, sitk.sitkFloat32))
    f = sitk.GetArrayFromImage(fixed)
    m = mask_np & (w > 0)
    return _ncc(f[m], w[m]) if m.sum() > 100 else float('nan')


def register_session(ref, ref_spacing, mov, mov_spacing, mov_mask=None,
                     stages=('rigid',), work_xy_um=2.0, bspline_ctrl_um=100.0, bspline_max_um=3.0,
                     max_rot_deg=10.0, metric='correlation', coarse=None, min_ncc=0.3, verbose=True):
    """
    Register a session volume (fixed) to the reference (moving); both should be reg_prep outputs.
    mov_mask: session voxels with data (coverage > 0).  Returns dict(transform (sitk, session µm -> reference µm),
    coarse, ncc per stage, stages).  `coarse` (a coarse_align result) skips the coarse search.
    A final NCC below min_ncc raises a RuntimeWarning (session probably outside the reference: the 10-29 vs
    11-02 sessions of subject 336 give 0.07; matching volumes give > 0.9).

    stages: any of 'rigid', 'affine', 'bspline' (in that order).  Rigid is the default: on the synthetic test
    (rotation + tilt + 2 µm random warp) it gives median/p95 errors 0.5/1.2 µm (xy) and 0.6/1.4 µm (z);
    residual local misalignment is left to refine_cells.  The B-spline is bounded (each control-point
    displacement within ±bspline_max_um): it lowers the median xy error (0.33 µm) but raises z p95 (3 µm);
    unbounded it fits noise (z errors of tens of µm).
    """
    import SimpleITK as sitk
    if coarse is None:
        coarse = coarse_align(ref, ref_spacing, mov, mov_spacing, mov_mask, max_rot_deg=max_rot_deg,
                              verbose=verbose)
    fx_np, fx_sp, fx_org = _working(mov, mov_spacing, work_xy_um)
    mv_np, mv_sp, mv_org = _working(ref, ref_spacing, work_xy_um)
    fixed, moving = _to_sitk(fx_np, fx_sp, fx_org), _to_sitk(mv_np, mv_sp, mv_org)
    if mov_mask is not None:
        fm_np = _bin_xy(mov_mask.astype(np.float32), int(round(fx_sp[1] / mov_spacing[1]))) > 0.99
    else:
        fm_np = np.ones(fx_np.shape, bool)
    fm_np &= fx_np > 0
    fmask = _to_sitk(fm_np.astype(np.uint8), fx_sp, fx_org)
    center = (np.array(fx_np.shape) - 1) / 2 * np.array(fx_sp) + np.array(fx_org)

    T = _sitk_linear('rigid', coarse['M'], coarse['t'], center)
    ncc = {'coarse': _ncc_after(fixed, moving, T, fm_np)}
    if verbose:
        print(f'NCC after coarse: {ncc["coarse"]:.3f}')

    def _linear_stage(T0, lr, iters):
        R = _reg_method(metric, fmask, 0.05)
        R.SetOptimizerAsRegularStepGradientDescent(learningRate=lr, minStep=1e-4, numberOfIterations=iters,
                                                   relaxationFactor=0.5, gradientMagnitudeTolerance=1e-8)
        R.SetOptimizerScalesFromPhysicalShift()
        R.SetShrinkFactorsPerLevel([4, 2, 1])
        R.SetSmoothingSigmasPerLevel([4.0, 2.0, 0.0])
        R.SetInitialTransform(T0, inPlace=True)
        R.Execute(fixed, moving)
        return T0

    if 'rigid' in stages:
        T = _linear_stage(T, 1.0, 200)
        ncc['rigid'] = _ncc_after(fixed, moving, T, fm_np)
        if verbose:
            print(f'NCC after rigid:  {ncc["rigid"]:.3f}')
    if 'affine' in stages:
        A = sitk.AffineTransform(3)
        A.SetCenter(T.GetCenter()); A.SetMatrix(T.GetMatrix()); A.SetTranslation(T.GetTranslation())
        T = _linear_stage(A, 0.5, 200)
        ncc['affine'] = _ncc_after(fixed, moving, T, fm_np)
        if verbose:
            print(f'NCC after affine: {ncc["affine"]:.3f}')
    if 'bspline' in stages:
        mesh = [max(1, int(round(sz * sp / bspline_ctrl_um)))
                for sz, sp in zip(fixed.GetSize(), fixed.GetSpacing())]
        B = sitk.BSplineTransformInitializer(fixed, mesh, order=3)
        R = _reg_method(metric, fmask, 0.05)
        R.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5, numberOfIterations=100,
                               maximumNumberOfCorrections=5, maximumNumberOfFunctionEvaluations=1000,
                               costFunctionConvergenceFactor=1e7,
                               lowerBound=-bspline_max_um, upperBound=bspline_max_um)
        R.SetShrinkFactorsPerLevel([2, 1])
        R.SetSmoothingSigmasPerLevel([2.0, 0.0])
        R.SetMovingInitialTransform(T)
        R.SetInitialTransform(B, inPlace=True)
        R.Execute(fixed, moving)
        T = sitk.CompositeTransform([T, B])             # p_ref = linear(bspline(p_session))
        ncc['bspline'] = _ncc_after(fixed, moving, T, fm_np)
        if verbose:
            print(f'NCC after B-spline (mesh {mesh}): {ncc["bspline"]:.3f}')
    final = list(ncc.values())[-1]
    if not final >= min_ncc:
        import warnings
        warnings.warn(f'registration NCC {final:.3f} < {min_ncc}: the session may not overlap the reference '
                      '(different FOV?) -- check the overlay before using the transferred cells',
                      RuntimeWarning, stacklevel=2)
    return dict(transform=T, coarse=coarse, ncc=ncc, stages=list(stages))


def save_transform(path, reg, **extra):
    """Transform file (.h5) + JSON summary (coarse result, NCC per stage, extra fields)."""
    import SimpleITK as sitk
    sitk.WriteTransform(reg['transform'], path)
    c = reg['coarse']
    summary = dict(transform_file=os.path.basename(path), stages=reg['stages'], ncc=reg['ncc'],
                   coarse=dict(theta_deg=c['theta_deg'], t_zyx_um=np.asarray(c['t']).tolist(),
                               score_xy=c['score_xy'], score_z=c['score_z']), **extra)
    with open(os.path.splitext(path)[0] + '.json', 'w') as f:
        json.dump(summary, f, indent=1)


def load_transform(path):
    """(transform, summary dict) saved by save_transform."""
    import SimpleITK as sitk
    T = sitk.ReadTransform(path)
    with open(os.path.splitext(path)[0] + '.json') as f:
        return T, json.load(f)


# ── label transfer ────────────────────────────────────────────────────────────
def warp_to_session(vol, ref_spacing, T, session_shape, session_spacing, labels=False):
    """Resample a reference-space volume onto the session grid with T (session µm -> reference µm)."""
    import SimpleITK as sitk
    if labels:
        mv = _to_sitk(vol.astype(np.uint16), ref_spacing)
        interp, ptype = sitk.sitkNearestNeighbor, sitk.sitkUInt16
    else:
        mv = _to_sitk(vol.astype(np.float32), ref_spacing)
        interp, ptype = sitk.sitkLinear, sitk.sitkFloat32
    grid = sitk.Image([int(s) for s in session_shape[::-1]], sitk.sitkUInt8)
    grid.SetSpacing([float(s) for s in session_spacing[::-1]])
    return sitk.GetArrayFromImage(sitk.Resample(mv, grid, T, interp, 0, ptype))


def refine_cells(labels, session_img, ref_warped, spacing, max_shift_um=(3.0, 3.0, 3.0), margin_um=5.0,
                 min_gain=0.05, min_ncc=0.3, verbose=True):
    """
    Per-cell local refinement after label transfer.  For each label, integer shifts within max_shift_um are
    scored by NCC between the session image at the shifted position and the warped reference around the
    cell (cell dilated by margin_um).  A shift is applied only if it raises the NCC by >= min_gain and the
    result is >= min_ncc.  Cells are written in order of their final NCC; voxels already taken are kept.
    Returns (refined labels, DataFrame cell_id / ncc0 / ncc / shift_z,y,x (µm) / moved).
    """
    import pandas as pd
    dz, dy, dx = spacing
    rz, ry, rx = (int(round(m / s)) for m, s in zip(max_shift_um, spacing))
    mz, my, mx = (max(1, int(round(margin_um / s))) for s in spacing)
    shp = labels.shape
    disk = ndi.generate_binary_structure(2, 1)[None]          # xy-only dilation element
    rows, placed = [], []
    for i, sl in enumerate(ndi.find_objects(labels)):
        if sl is None:
            continue
        cid = i + 1
        lo = [max(0, sl[0].start - mz), max(0, sl[1].start - my), max(0, sl[2].start - mx)]
        hi = [min(shp[0], sl[0].stop + mz), min(shp[1], sl[1].stop + my), min(shp[2], sl[2].stop + mx)]
        box = tuple(slice(a, b) for a, b in zip(lo, hi))
        cell = labels[box] == cid
        reg = ndi.binary_dilation(cell, structure=disk, iterations=my)
        zz, yy, xx = np.nonzero(reg)
        wv = ref_warped[box][zz, yy, xx]
        best, ncc0 = (-np.inf, (0, 0, 0)), np.nan
        for sz in range(-rz, rz + 1):
            for sy in range(-ry, ry + 1):
                for sx in range(-rx, rx + 1):
                    z2, y2, x2 = zz + lo[0] + sz, yy + lo[1] + sy, xx + lo[2] + sx
                    ok = (z2 >= 0) & (z2 < shp[0]) & (y2 >= 0) & (y2 < shp[1]) & (x2 >= 0) & (x2 < shp[2])
                    if ok.sum() < 0.8 * ok.size:
                        continue
                    sc = _ncc(session_img[z2[ok], y2[ok], x2[ok]], wv[ok])
                    if (sz, sy, sx) == (0, 0, 0):
                        ncc0 = sc
                    if sc > best[0]:
                        best = (sc, (sz, sy, sx))
        sc, d = best
        moved = bool(np.isfinite(ncc0) and sc - ncc0 >= min_gain and sc >= min_ncc)
        if not moved:
            sc, d = ncc0, (0, 0, 0)
        rows.append(dict(cell_id=cid, ncc0=ncc0, ncc=sc, shift_z=d[0] * dz, shift_y=d[1] * dy,
                         shift_x=d[2] * dx, moved=moved))
        cz, cy, cx = np.nonzero(cell)
        placed.append((sc if np.isfinite(sc) else -np.inf, cid,
                       cz + lo[0] + d[0], cy + lo[1] + d[1], cx + lo[2] + d[2]))
    out = np.zeros_like(labels)
    for sc, cid, z, y, x in sorted(placed, key=lambda p: -p[0]):
        ok = (z >= 0) & (z < shp[0]) & (y >= 0) & (y < shp[1]) & (x >= 0) & (x < shp[2])
        z, y, x = z[ok], y[ok], x[ok]
        free = out[z, y, x] == 0
        out[z[free], y[free], x[free]] = cid
    df = pd.DataFrame(rows)
    if verbose and len(df):
        print(f'refine_cells: {len(df)} cells, {int(df.moved.sum())} shifted;  NCC median '
              f'{df.ncc0.median():.3f} -> {df.ncc.median():.3f}')
    return out, df


def cell_qc(ref_labels, ref_spacing, session_labels, session_spacing, coverage=None, refine_df=None,
            min_ncc=0.3, min_in_fov=0.5):
    """
    Per reference cell: voxels expected in the session grid vs found, coverage fraction, NCC, flag
    ('ok', 'partial', 'low_match', 'outside').  Cells are kept and flagged, not dropped.
    """
    import pandas as pd
    n = int(max(ref_labels.max(), session_labels.max()))
    ref_vox = np.bincount(ref_labels.ravel(), minlength=n + 1)[1:]
    ses_vox = np.bincount(session_labels.ravel(), minlength=n + 1)[1:]
    expected = ref_vox * np.prod(ref_spacing) / np.prod(session_spacing)
    in_fov = np.where(expected > 0, ses_vox / np.maximum(expected, 1e-9), 0.0)
    if coverage is not None:
        cov_vox = np.bincount(session_labels.ravel(), weights=(coverage > 0).ravel().astype(float),
                              minlength=n + 1)[1:]
        cov_frac = np.where(ses_vox > 0, cov_vox / np.maximum(ses_vox, 1), 0.0)
    else:
        cov_frac = np.where(ses_vox > 0, 1.0, 0.0)
    df = pd.DataFrame(dict(cell_id=np.arange(1, n + 1), ref_voxels=ref_vox, session_voxels=ses_vox,
                           in_fov=in_fov, coverage=cov_frac))
    df = df[df.ref_voxels > 0]
    if refine_df is not None and len(refine_df):
        df = df.merge(refine_df, on='cell_id', how='left')
    else:
        df['ncc'] = np.nan
    flag = np.full(len(df), 'ok', object)
    if df.ncc.notna().any():                       # NaN NCC (no valid local comparison) counts as low_match
        flag[~(df.ncc >= min_ncc).to_numpy()] = 'low_match'
    flag[((df.in_fov < min_in_fov) | (df.coverage < 0.9)).to_numpy()] = 'partial'
    flag[(df.session_voxels == 0).to_numpy()] = 'outside'
    df['flag'] = flag
    return df.reset_index(drop=True)


def plot_registration_qc(ses, ref_w, spacing, mask=None, labels=None, z=None, title=''):
    """
    Session (green) vs warped reference (magenta), both 0..1: max projections, a checkerboard of the
    projections, and one z plane with label outlines.  Grey/white = agreement.
    """
    import matplotlib.pyplot as plt
    from skimage.segmentation import find_boundaries
    if mask is not None:                    # same voxels on both sides, so both are scaled alike
        ses = np.where(mask, ses, 0)
        ref_w = np.where(mask, ref_w, 0)
    z = ses.shape[0] // 2 if z is None else z

    def _n(a):
        lo, hi = np.percentile(a, 1), np.percentile(a, 99.7)
        return np.clip((a - lo) / max(hi - lo, 1e-9), 0, 1)

    a, b = _n(ses.max(axis=0)), _n(ref_w.max(axis=0))
    ov = np.dstack([b, a, b])
    k = max(8, min(a.shape) // 8)
    cb = ((np.arange(a.shape[0])[:, None] // k + np.arange(a.shape[1])[None] // k) % 2).astype(bool)
    chk = np.where(cb, a, b)
    za, zb = _n(ses[z]), _n(ref_w[z])
    zov = np.dstack([zb, za, zb])
    if labels is not None:
        zov[find_boundaries(labels[z], mode='inner')] = (1.0, 0.8, 0.0)
    aspect = spacing[1] / spacing[2]
    fig, axs = plt.subplots(1, 3, figsize=(18, 7 * a.shape[0] / a.shape[1] + 1))
    for ax, img, ttl in ((axs[0], ov, 'MIP: session (green) / reference (magenta)'),
                         (axs[1], chk, 'MIP checkerboard (session / reference)'),
                         (axs[2], zov, f'z={z}' + ('  + cell outlines (yellow)' if labels is not None else ''))):
        ax.imshow(img, cmap=None if img.ndim == 3 else 'gray', aspect=aspect, interpolation='nearest')
        ax.set_title(ttl, fontsize=10)
        ax.axis('off')
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig


# ── subject: reference, session registry, collection ─────────────────────────
def subject_paths(subject_dir):
    """File layout of a subject folder."""
    ref_dir = os.path.join(subject_dir, 'reference')
    return dict(subject_dir=subject_dir, ref_dir=ref_dir,
                ref_meta=os.path.join(ref_dir, 'reference_meta.json'),
                ref_labels_auto=os.path.join(ref_dir, 'reference_labels_auto.tif'),
                ref_labels=os.path.join(ref_dir, 'reference_labels.tif'),
                ref_cells=os.path.join(ref_dir, 'reference_cells.csv'),
                sessions=os.path.join(subject_dir, 'sessions.json'),
                cells_long=os.path.join(subject_dir, 'cells_long.csv'))


def save_reference_meta(subject_dir, source, spacing, **extra):
    """Record the reference volume (path to a ZYX TIFF, not copied) and its spacing (dz, dy, dx) µm."""
    P = subject_paths(subject_dir)
    os.makedirs(P['ref_dir'], exist_ok=True)
    meta = dict(source=source, spacing_zyx_um=[float(s) for s in spacing], **extra)
    with open(P['ref_meta'], 'w') as f:
        json.dump(meta, f, indent=1)
    return meta


def load_reference(subject_dir, labels=True):
    """dict(vol, spacing, labels (curated reference_labels.tif, or None), meta) of a subject."""
    P = subject_paths(subject_dir)
    if not os.path.exists(P['ref_meta']):
        raise FileNotFoundError(f'{P["ref_meta"]} not found -- set the reference in MultiDay_Reference.ipynb')
    with open(P['ref_meta']) as f:
        meta = json.load(f)
    vol, sp = load_volume(meta['source'], meta['spacing_zyx_um'])
    lab = None
    if labels and os.path.exists(P['ref_labels']):
        lab = tifffile.imread(P['ref_labels']).astype(np.uint16)
        if lab.shape != vol.shape:
            raise ValueError(f'reference labels {lab.shape} != reference volume {vol.shape}')
    return dict(vol=vol, spacing=sp, labels=lab, meta=meta)


def reference_cells_table(labels, spacing):
    """cell_id, voxels, volume_um3 and centroid (µm, ZYX) of every label."""
    import pandas as pd
    counts = np.bincount(labels.ravel())
    ids = np.flatnonzero(counts)
    ids = ids[ids > 0]
    com = np.array(ndi.center_of_mass(np.ones(labels.shape, np.uint8), labels, ids)).reshape(-1, 3)
    com = com * np.asarray(spacing)
    return pd.DataFrame(dict(cell_id=ids, voxels=counts[ids], volume_um3=counts[ids] * np.prod(spacing),
                             centroid_z_um=com[:, 0].round(2), centroid_y_um=com[:, 1].round(2),
                             centroid_x_um=com[:, 2].round(2)))


def session_date(path, fallback=None):
    """YYYY-MM-DD from a YYYYMMDD token in the path, else from the file's modification time, else `fallback`
    (e.g. the .fbs.xml acquisition date when the raw file was deleted after its uint8 conversion)."""
    import re
    import datetime as _dt
    m = re.search(r'(?<!\d)(20\d{2})(\d{2})(\d{2})(?!\d)', path)
    if m:
        return f'{m.group(1)}-{m.group(2)}-{m.group(3)}'
    if os.path.exists(path):
        return _dt.datetime.fromtimestamp(os.path.getmtime(path)).strftime('%Y-%m-%d')
    return fallback


def update_sessions(subject_dir, **entry):
    """Add or replace (same 'session') an entry of <subject>/sessions.json; returns the list."""
    P = subject_paths(subject_dir)
    sessions = []
    if os.path.exists(P['sessions']):
        with open(P['sessions']) as f:
            sessions = json.load(f)
    sessions = [s for s in sessions if s.get('session') != entry['session']] + [entry]
    sessions.sort(key=lambda s: (s.get('date') or '', s['session']))
    with open(P['sessions'], 'w') as f:
        json.dump(sessions, f, indent=1)
    return sessions


def collect_subject(subject_dir, save=True, verbose=True):
    """
    Long table of all sessions in sessions.json: per-cell lifetimes (Step 14/15), tav statistics (Step 16)
    and registration QC (Step 2b), one row per (session, cell_id).  Saved as <subject>/cells_long.csv.
    """
    import pandas as pd
    P = subject_paths(subject_dir)
    with open(P['sessions']) as f:
        sessions = json.load(f)
    frames = []
    for s in sessions:
        d, base, tag = s['analysis_dir'], s['session'], s.get('cell_tag', '')
        f_lt = os.path.join(d, f'{base}{tag}_per_cell_lifetimes.csv')
        f_tav = os.path.join(d, f'{base}{tag}_per_cell_tav.csv')
        f_qc = s.get('qc') or os.path.join(d, f'{base}{tag}_cell_qc.csv')
        if not os.path.exists(f_lt):
            if verbose:
                print(f'  {s.get("date")}  {base}: no per-cell lifetimes yet (run Steps 14-16)')
            continue
        df = pd.read_csv(f_lt)
        if os.path.exists(f_tav):
            df = df.merge(pd.read_csv(f_tav), on='cell_id', how='left', suffixes=('', '_tav'))
        if os.path.exists(f_qc):
            q = pd.read_csv(f_qc)
            df = df.merge(q[[c for c in ('cell_id', 'flag', 'ncc', 'in_fov', 'coverage', 'moved')
                             if c in q.columns]], on='cell_id', how='left')
        df.insert(0, 'session', base)
        df.insert(1, 'date', s.get('date'))
        frames.append(df)
        if verbose:
            n_ok = int((df.flag == 'ok').sum()) if 'flag' in df else len(df)
            print(f'  {s.get("date")}  {base}: {len(df)} cells, {n_ok} flagged ok')
    long = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if save and len(long):
        long.to_csv(P['cells_long'], index=False)
        if verbose:
            print(f'Saved {len(long)} rows -> {P["cells_long"]}')
    return long


def plot_subject(long, value='aw_LT', flags=('ok',), min_sessions=2, title=''):
    """Per-cell `value` across sessions: trajectories (left) and cells x sessions heatmap (right)."""
    import matplotlib.pyplot as plt
    df = long
    if 'flag' in df and flags:
        df = df[df.flag.isin(flags)]
    piv = df.pivot_table(index='cell_id', columns='date', values=value, aggfunc='first')
    piv = piv[piv.notna().sum(axis=1) >= min_sessions]
    fig, axs = plt.subplots(1, 2, figsize=(15, 5.5), gridspec_kw=dict(width_ratios=[1.2, 1]))
    if piv.empty:
        for ax in axs:
            ax.text(0.5, 0.5, f'no cell with {value} in >= {min_sessions} sessions', ha='center', transform=ax.transAxes)
        return fig
    x = np.arange(piv.shape[1])
    for _, row in piv.iterrows():
        axs[0].plot(x, row.values, '-', color='0.6', lw=0.6, alpha=0.5)
    axs[0].plot(x, piv.median(axis=0).values, 'o-', color='C3', lw=2, label='median')
    axs[0].set_xticks(x, piv.columns, rotation=30)
    axs[0].set_ylabel(value)
    axs[0].set_title(f'{len(piv)} cells in >= {min_sessions} sessions (flags {", ".join(flags)})', fontsize=10)
    axs[0].legend()
    order = piv.mean(axis=1).sort_values().index
    lo, hi = np.nanpercentile(piv.values, [5, 95])
    im = axs[1].imshow(piv.loc[order].values, aspect='auto', cmap='viridis', vmin=lo, vmax=hi,
                       interpolation='nearest')
    axs[1].set_xticks(x, piv.columns, rotation=30)
    axs[1].set_ylabel('cells (sorted by mean)')
    plt.colorbar(im, ax=axs[1], label=value)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig
