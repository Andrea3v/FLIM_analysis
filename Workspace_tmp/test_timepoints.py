"""register_timepoints / time_positions / fuse_tiles on synthetic drift.
Timepoints are crops of one larger random-structure volume, so the true drift is known exactly."""
import os, sys, tempfile
import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import vista_tiles as vt

rng = np.random.default_rng(1)
nz, ny, nx, nbin, M = 8, 128, 128, 4, 12
world = gaussian_filter(rng.random((nz + 2 * M, ny + 2 * M, nx + 2 * M)), (1.0, 1.5, 1.5)) ** 4
world /= world.max()
drift = [(0, 0, 0), (1, -7, 4), (-2, 10, -3), (3, 2, 11)]      # origin of timepoint t in timepoint 0 coords


def crop(d):
    z, y, x = M + np.array(d)
    return world[z:z + nz, y:y + ny, x:x + nx]


tile_int_t = np.stack([crop(d)[np.newaxis] for d in drift]) * 40 + rng.poisson(2, (len(drift), 1, nz, ny, nx))
res = vt.register_timepoints(tile_int_t.astype(np.float32), ref=0, max_dz=4)
assert [tuple(s) for s in res['shifts']] == drift, res['shifts']
assert all(n > 0.5 for n in res['ncc']), res['ncc']
print('drift recovered:', res['shifts'])

# reference = last timepoint: shifts relative to it
res3 = vt.register_timepoints(tile_int_t.astype(np.float32), ref=3, max_dz=6, verbose=False)
assert [tuple(s) for s in res3['shifts']] == [tuple(np.subtract(d, drift[3])) for d in drift], res3['shifts']
print('ref=3 OK')

# common canvas: every timepoint's content lands at the same canvas voxels
reg = dict(positions=[[0, 0, 0]], tile_shape=[nz, ny, nx], canvas_shape=[nz, ny, nx],
           norm_lo=[0.0], norm_hi=[1.0])
pos, canvas = vt.time_positions(reg, res['shifts'])
assert canvas == [nz + 5, ny + 17, nx + 14], canvas
for t, d in enumerate(drift):
    w = np.zeros(canvas); z, y, x = pos[t, 0]
    w[z:z + nz, y:y + ny, x:x + nx] = crop(d)
    if t == 0:
        w0 = w
    else:
        both = (w > 0) & (w0 > 0)
        assert np.allclose(w[both], w0[both]), t
print('time_positions OK:', pos[:, 0].tolist(), canvas)

# fuse: time-sum photons = sum of all timepoints; per-timepoint stack = its own photons
with tempfile.TemporaryDirectory() as td:
    paths = []
    for t, d in enumerate(drift):
        dec = rng.poisson(np.repeat(crop(d)[:, np.newaxis] * 3, nbin, axis=1)).astype(np.uint8)
        paths.append(os.path.join(td, f't{t}.tif'))
        tifffile.imwrite(paths[-1], dec, imagej=True, metadata={'axes': 'ZCYX'})
    ti = np.stack([tifffile.imread(p).sum(axis=1) for p in paths]).astype(np.float32)
    reg_all = dict(reg, positions=pos.reshape(-1, 3).tolist(), canvas_shape=canvas,
                   norm_lo=[0.0] * len(drift), norm_hi=[1.0] * len(drift))
    st = vt.fuse_tiles(paths, reg_all, ti, td, 'sum', verbose=False)
    assert st['photons_fused'] == st['photons_tiles'], (st['photons_fused'], st['photons_tiles'])
    cov = tifffile.imread(st['cov_path'])
    assert cov.max() == len(drift) and cov.shape == tuple(canvas)
    st1 = vt.fuse_tiles(paths[1:2], dict(reg, positions=pos[1].tolist(), canvas_shape=canvas), ti[1:2], td,
                        'sum_T01', stitched_name='stitched_T01', verbose=False)
    assert st1['stitched_dir'].endswith('stitched_T01') and st1['photons_fused'] == ti[1].sum()
    assert len(st1['plane_paths']) == canvas[0]
print('fuse_tiles time-sum / per-timepoint OK')

# convert_to_uint8 file names for a time series (no conversion: names only)
print('ALL OK')
