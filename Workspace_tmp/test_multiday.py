"""Checks for multiday.py registration + label transfer on a synthetic session with a known transform.

Reference = the 11-02 stitched intensity.  Session = the reference resampled through a known
session -> reference transform (rotation about z + small tilt + translation + smooth B-spline warp),
with a smooth gain change, noise and a missing corner (coverage mask).
"""
import os
import sys
import time
import glob
import numpy as np
import tifffile
import SimpleITK as sitk
sys.path.insert(0, r'X:\FLIM_2026')
import multiday as md

D = sys.argv[1] if len(sys.argv) > 1 else \
    r'D:\Peredox_FLIM\XYZT1_336_X+0.50_Y+0.80_Decay128_512x512_Z186_Ch1_T1_P1_S1'
ref_sp = (3.0, 0.99, 0.99)
ref, _ = md.load_volume(glob.glob(os.path.join(D, '*_stitched_intensity.tif'))[0], ref_sp)
labels = tifffile.imread(glob.glob(os.path.join(D, 'stitched', '*_FLIM_Analysis', '*_cellpose_masks.tif'))[0])
rng = np.random.default_rng(1)

# ── ground-truth session -> reference transform ──────────────────────────────
ses_sp, ses_shape = (3.0, 0.99, 0.99), (22, 800, 600)
ses_center = (np.array(ses_shape) - 1) / 2 * np.array(ses_sp)         # ZYX µm
target = np.array([52.0, 700.0, 480.0])                                # where the centre lands in the ref
E = sitk.Euler3DTransform()
E.SetCenter(ses_center[::-1].tolist())
E.SetRotation(np.deg2rad(1.0), np.deg2rad(-0.5), np.deg2rad(4.0))     # about x, y, z
E.SetTranslation((target - ses_center)[::-1].tolist())
grid = sitk.Image([int(s) for s in ses_shape[::-1]], sitk.sitkUInt8)
grid.SetSpacing([float(s) for s in ses_sp[::-1]])
B = sitk.BSplineTransformInitializer(grid, [4, 4, 1], order=3)
p = np.array(B.GetParameters())
p = rng.normal(0, 2.0, p.size)                                         # ~2 µm control-point displacements
p[2 * p.size // 3:] *= 0.5                                             # weaker in z
B.SetParameters(p.tolist())
T_true = sitk.CompositeTransform([E, B])

ses = md.warp_to_session(ref, ref_sp, T_true, ses_shape, ses_sp)
zz, yy, xx = np.meshgrid(*[np.linspace(-1, 1, n) for n in ses_shape], indexing='ij')
ses = ses * (1.0 + 0.35 * yy - 0.25 * xx + 0.15 * zz)                  # smooth gain change
ses = np.clip(ses + rng.normal(0, 0.08, ses.shape), 0, None).astype(np.float32)
cov = np.ones(ses_shape, np.uint8)
cov[14:, 500:, 400:] = 0                                               # missing tile corner in deep planes
ses[cov == 0] = 0

# ── registration ──────────────────────────────────────────────────────────────
t0 = time.time()
ref_p = md.reg_prep(ref, ref_sp)
ses_p = md.reg_prep(ses, ses_sp)
reg = md.register_session(ref_p, ref_sp, ses_p, ses_sp, mov_mask=cov > 0)
print(f'registration: {time.time() - t0:.0f} s')

# displacement error on session voxels with data
idx = np.argwhere(cov > 0)
idx = idx[rng.choice(len(idx), 4000, replace=False)]
pts = idx * np.array(ses_sp)
err = np.array([np.subtract(reg['transform'].TransformPoint(tuple(q[::-1])),
                            T_true.TransformPoint(tuple(q[::-1])))[::-1] for q in pts])
exy, ez = np.hypot(err[:, 1], err[:, 2]), np.abs(err[:, 0])
print(f'error xy: median {np.median(exy):.2f} µm, p95 {np.percentile(exy, 95):.2f} µm  |  '
      f'z: median {np.median(ez):.2f} µm, p95 {np.percentile(ez, 95):.2f} µm')

# ── label transfer vs ground truth ────────────────────────────────────────────
lab_true = md.warp_to_session(labels, ref_sp, T_true, ses_shape, ses_sp, labels=True)
lab_est = md.warp_to_session(labels, ref_sp, reg['transform'], ses_shape, ses_sp, labels=True)
ref_w = md.warp_to_session(ref_p, ref_sp, reg['transform'], ses_shape, ses_sp)
lab_ref, rdf = md.refine_cells(lab_est, ses_p, ref_w, ses_sp)


def iou_per_cell(a, b, min_vox=20):
    n = int(max(a.max(), b.max())) + 1
    inter = np.bincount(a.ravel()[(a.ravel() == b.ravel()) & (a.ravel() > 0)], minlength=n)
    na, nb = np.bincount(a.ravel(), minlength=n), np.bincount(b.ravel(), minlength=n)
    ok = na >= min_vox
    ok[0] = False
    return inter[ok] / (na[ok] + nb[ok] - inter[ok])


def centroid_err(a, b, sp, min_vox=20):
    """Distance (µm) between the centroids of the same label in a and b."""
    from scipy import ndimage as ndi
    n = int(max(a.max(), b.max()))
    na, nb = np.bincount(a.ravel(), minlength=n + 1), np.bincount(b.ravel(), minlength=n + 1)
    ids = np.flatnonzero((na >= min_vox) & (nb >= min_vox))
    ids = ids[ids > 0]
    ca = np.array(ndi.center_of_mass(np.ones_like(a), a, ids)) * sp
    cb = np.array(ndi.center_of_mass(np.ones_like(b), b, ids)) * sp
    return np.linalg.norm(ca - cb, axis=1)


iou = iou_per_cell(lab_true, lab_est)
iou_r = iou_per_cell(lab_true, lab_ref)
ce, ce_r = centroid_err(lab_true, lab_est, ses_sp), centroid_err(lab_true, lab_ref, ses_sp)
print(f'labels vs truth ({iou.size} cells): IoU median {np.median(iou):.3f}, centroid error median '
      f'{np.median(ce):.2f} µm (p95 {np.percentile(ce, 95):.2f})  |  after refine_cells: IoU {np.median(iou_r):.3f}, '
      f'centroid {np.median(ce_r):.2f} µm (p95 {np.percentile(ce_r, 95):.2f})')
qc = md.cell_qc(labels, ref_sp, lab_ref, ses_sp, coverage=cov, refine_df=rdf)
print('QC flags:', qc.flag.value_counts().to_dict())

# ── identity: reference registered to itself ─────────────────────────────────
reg_i = md.register_session(ref_p, ref_sp, ref_p, ref_sp, stages=('rigid',), verbose=False)
lab_i = md.warp_to_session(labels, ref_sp, reg_i['transform'], labels.shape, ref_sp, labels=True)
same = np.mean(lab_i[labels > 0] == labels[labels > 0])
print(f'identity: coarse {reg_i["coarse"]["theta_deg"]:+.1f}° {np.round(reg_i["coarse"]["t"], 2)} µm, '
      f'labels unchanged on {100 * same:.1f}% of cell voxels')

# ── transform file round trip ─────────────────────────────────────────────────
tmp = os.path.join(os.environ.get('TEMP', '.'), 'test_multiday_tf.h5')
md.save_transform(tmp, reg, session='synthetic')
T2, summ = md.load_transform(tmp)
q = tuple(pts[0][::-1])
print('transform round trip OK:', np.allclose(T2.TransformPoint(q), reg['transform'].TransformPoint(q)))

ok = (np.median(exy) < 1.0 and np.percentile(exy, 95) < 2.0 and np.median(ez) < 1.0
      and np.percentile(ez, 95) < 2.0 and np.median(ce_r) < 1.5 and same > 0.99)
print('ALL OK' if ok else 'CHECK FAILED')
