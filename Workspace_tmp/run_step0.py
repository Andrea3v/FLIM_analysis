"""Step 0 (convert -> register -> fuse) outside the notebook, with QC figure + checks."""
import os, sys, time
import numpy as np
import tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, r'X:\FLIM_2026')
import vista_tiles as vt

RAW, QC_PNG = sys.argv[1], sys.argv[2]
raw_dir = os.path.dirname(RAW)
info = vt.parse_vista_name(RAW)
grid = (2, 3)

t = time.time()
conv = vt.convert_to_uint8(RAW, os.path.join(raw_dir, 'tiles_uint8'), n_pos=grid[0] * grid[1])
print(f'convert: {time.time() - t:.0f} s')

# uint8 tiles vs float file, a few blocks
rep = conv['report']
with open(RAW, 'rb') as fh:
    lay = vt.vista_layout(RAW)
    for p, z in ((0, 0), (1, 15), (3, 7), (5, 15)):
        tile = tifffile.memmap(conv['tile_paths'][p], mode='r')
        k0 = (p * rep['n_z'] + z) * rep['nbin']
        ref = sum(vt.read_plane(fh, lay, k0 + t_).astype(np.float64).sum() for t_ in range(rep['nbin']))
        print(f'  tile {p} z {z}: uint8 sum {int(tile[z].sum(dtype=np.int64)):,}  float sum {ref:,.0f}')
        del tile

t = time.time()
reg = vt.register_tiles(conv['tile_int'], grid, overlap=0.20, layout='auto', fallback='snake_tl', max_dz=2)
print(f'register: {time.time() - t:.0f} s')
vt.save_tile_positions(os.path.join(raw_dir, f'{info["base"]}_tile_positions.json'), reg, raw_path=RAW)

t = time.time()
fus = vt.fuse_tiles(conv['tile_paths'], reg, conv['tile_int'], raw_dir, info['base'], overwrite=True)
print(f'fuse: {time.time() - t:.0f} s')
p0 = tifffile.imread(fus['plane_paths'][0])
print('stitched plane shape/dtype', p0.shape, p0.dtype)

# QC: fused intensity max projection + tile outlines, coverage
fi = tifffile.imread(fus['int_path'])
cov = tifffile.imread(fus['cov_path'])
mp = fi.max(axis=0)
fig, ax = plt.subplots(1, 2, figsize=(18, 7))
ax[0].imshow(mp, cmap='gray', vmin=0, vmax=np.percentile(mp, 99.5))
n_z, ny, nx = reg['tile_shape']
for p, (z0, y0, x0) in enumerate(reg['positions']):
    ax[0].add_patch(plt.Rectangle((x0, y0), nx, ny, fill=False, ec=f'C{p}', lw=1))
    ax[0].text(x0 + 8, y0 + 30, f'{p}', color=f'C{p}', fontsize=14)
ax[0].set_title(f'fused intensity (max over z), layout {reg["layout"]}')
ax[1].imshow(cov.max(axis=0)); ax[1].set_title('coverage (max over z)')
plt.tight_layout(); plt.savefig(QC_PNG, dpi=70)
print('QC ->', QC_PNG)
