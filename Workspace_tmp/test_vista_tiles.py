"""Checks for vista_tiles.py: byte layout on the real file + synthetic tile registration."""
import sys
import numpy as np
sys.path.insert(0, r'X:\FLIM_2026')
import vista_tiles as vt

RAW = sys.argv[1]

# ── 1. layout / reader ────────────────────────────────────────────────────────
lay = vt.vista_layout(RAW)
print('layout', lay.summary())
rng = np.random.default_rng(0)
ks = sorted(set(rng.integers(0, lay.n_planes, 300).tolist()) | {0, 1, lay.k0 - 1, lay.k0, lay.k0 + 1, lay.n_planes - 1})
n_junk_in, n_junk_out, n_corr = 0, 0, 0
with open(RAW, 'rb') as fh:
    for k in ks:
        fh.seek(lay.plane_offset(k))
        a = np.frombuffer(fh.read(lay.plane_bytes), '<f4')
        with np.errstate(invalid='ignore'):
            junk = ~(np.isfinite(a) & (a >= 0) & (a == np.round(a)) & (a < 256))
        corr = np.zeros(a.size, bool)
        corr[lay.corrupted_pixels(k)] = True
        n_junk_in += int((junk & corr).sum())
        n_junk_out += int((junk & ~corr).sum())
        n_corr += int(corr.sum())
print(f'{len(ks)} planes: junk inside stray spans {n_junk_in}, junk OUTSIDE spans {n_junk_out}, '
      f'span pixels {n_corr}')
# 2 known extra junk pixels: 8 bytes at file offset 512 (plane 0) — a stray IFD pointer; zeroed by
# the residual-junk check in convert_to_uint8
assert n_junk_out <= 2, 'junk pixels not explained by stray IFD spans'

# ── 2. synthetic registration (2x3 snake_tl, 20% overlap, known offsets) ─────
nz, ny, nx = 16, 128, 128
step = int(ny * 0.8)
big = vt.gaussian_filter(rng.poisson(1.0, (nz + 4, 2 * ny, 3 * nx)).astype(np.float32), (1, 3, 3))
big += vt.gaussian_filter(rng.random(big.shape).astype(np.float32), (2, 8, 8)) * 5
rc = vt.tile_grid_positions('snake_tl', (2, 3))
jit = rng.integers(-4, 5, (6, 2)); dzs = np.array([0, 1, 0, -1, 1, 0]) + 2   # |relative dz| <= 2
truth = []
tiles = np.zeros((6, nz, ny, nx), np.float32)
for p, (r, c) in enumerate(rc):
    y0, x0 = r * step + jit[p, 0] + 8, c * step + jit[p, 1] + 8
    tiles[p] = big[dzs[p]:dzs[p] + nz, y0:y0 + ny, x0:x0 + nx]
    truth.append((dzs[p], y0, x0))
truth = np.array(truth); truth -= truth.min(axis=0)
reg = vt.register_tiles(tiles, (2, 3), overlap=0.2, layout='auto', verbose=False)
print('synthetic layout', reg['layout'], 'positions match truth:',
      np.array_equal(np.array(reg['positions']), truth))
print(' scores', {k: round(v, 3) for k, v in reg['layout_scores'].items()})
assert reg['layout'] == 'snake_tl' and np.array_equal(np.array(reg['positions']), truth)
print('ALL OK')
