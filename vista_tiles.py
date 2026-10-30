"""
vista_tiles.py
Readers and tile tools for ISS Vista multi-page XYTZ TIFFs (AnalyzeTCSPC_XYTZTile.ipynb).

Vista writes classic (32-bit offset) TIFFs even above 4 GB.  Layout of the 2026-10 exports:
    pages 0 .. K0-1   [plane data][IFD]     stride = plane_bytes + ifd_bytes
    pages K0 .. N-1   [plane data]          stride = plane_bytes
K0 = first page whose IFD would start beyond 4 GB.  The IFDs of pages >= K0 are written at
(true offset mod 2**32), i.e. on top of plane data in the first 4 GB.  Those spans are computed
exactly by `VistaLayout` and zeroed on read (the photons there are lost).
Plane order in the file: [tile position][z][decay bin].

Functions
---------
parse_vista_name     -- nbin / ny / nx / n_blocks / channel from the Vista file name
vista_layout         -- byte layout of a Vista TIFF (plane offsets + stray-IFD spans)
read_plane           -- one decay-bin plane as float32 photon counts (stray-IFD pixels zeroed)
convert_to_uint8     -- raw Vista TIFF -> one uint8 ZCYX hyperstack per tile + tile intensity
tile_grid_positions  -- (row, col) of every tile for a named acquisition layout
register_tiles       -- pairwise 3D phase correlation + global least squares -> integer offsets
save_tile_positions / load_tile_positions -- JSON round trip of the registration result
fuse_tiles           -- stitched per-z TYX TCSPC TIFFs (overlaps summed) + segmentation intensity
"""

import os
import re
import glob
import json
import struct
import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter
from skimage.registration import phase_cross_correlation

_FOUR_GB = 2 ** 32
LAYOUTS  = [f'{order}_{corner}' for order in ('snake', 'raster') for corner in ('tl', 'tr', 'bl', 'br')]


# ── file name / byte layout ───────────────────────────────────────────────────
def parse_vista_name(path):
    """Acquisition dimensions encoded in a Vista file name, e.g.
    'XYZT1_336__Decay128_512x512_Z96_Ch1_T1_P1_S1.tif' -> nbin=128, nx=ny=512, n_blocks=96."""
    name  = os.path.basename(path)
    m_bin = re.search(r'_Decay(\d+)', name)
    m_xy  = re.search(r'_(\d+)x(\d+)_', name)
    m_z   = re.search(r'_Z(\d+)_', name)
    m_ch  = re.search(r'_Ch(\d+)', name)
    if not (m_bin and m_xy and m_z):
        raise ValueError(f'Cannot parse _Decay<n>, _<nx>x<ny>_ and _Z<n>_ from "{name}"')
    return dict(nbin=int(m_bin[1]), nx=int(m_xy[1]), ny=int(m_xy[2]),
                n_blocks=int(m_z[1]), channel=int(m_ch[1]) if m_ch else None,
                base=os.path.splitext(name)[0])


def _read_ifd(fh, offset):
    fh.seek(offset)
    n = struct.unpack('<H', fh.read(2))[0]
    tags = {}
    for _ in range(n):
        code, typ, cnt, val = struct.unpack('<HHII', fh.read(12))
        tags[code] = (typ, cnt, val)
    return tags, struct.unpack('<I', fh.read(4))[0]


def _tag_first(fh, tag):
    """First value of a SHORT/LONG tag, inline or via pointer."""
    typ, cnt, val = tag
    size = 2 if typ == 3 else 4
    if cnt * size <= 4:
        return val & 0xFFFF if typ == 3 else val
    fh.seek(val)
    return struct.unpack('<H' if typ == 3 else '<I', fh.read(size))[0]


class VistaLayout:
    """Byte layout of a Vista float32 TIFF; see module docstring."""

    def __init__(self, path, nbin, n_blocks, ny, nx):
        with open(path, 'rb') as fh:
            hdr = fh.read(8)
            if hdr[:4] != b'II*\x00':
                raise ValueError(f'{path}: not a little-endian classic TIFF')
            ifd0 = struct.unpack('<I', hdr[4:])[0]
            t0, ifd1 = _read_ifd(fh, ifd0)
            width, length = _tag_first(fh, t0[256]), _tag_first(fh, t0[257])
            bps   = _tag_first(fh, t0[258])
            fmt   = _tag_first(fh, t0[339]) if 339 in t0 else 1
            data0 = _tag_first(fh, t0[273])
            t1, _ = _read_ifd(fh, ifd1)
            data1 = _tag_first(fh, t1[273])
        if (width, length) != (nx, ny):
            raise ValueError(f'TIFF is {width}x{length}, file name says {nx}x{ny}')
        if (bps, fmt) != (32, 3):
            raise ValueError(f'expected float32 planes, got {bps} bit / SampleFormat {fmt}')

        self.path        = path
        self.ny, self.nx = ny, nx
        self.n_planes    = nbin * n_blocks
        self.plane_bytes = ny * nx * 4
        self.data0       = data0
        self.stride      = data1 - data0
        self.ifd_bytes   = self.stride - self.plane_bytes
        if ifd0 != data0 + self.plane_bytes or self.ifd_bytes <= 0:
            raise ValueError('unexpected Vista page layout (IFD not right after plane data)')

        # pages whose IFD starts below 4 GB keep [data][IFD]; later pages are data only
        self.k0 = min(self.n_planes,
                      (_FOUR_GB - 1 - data0 - self.plane_bytes) // self.stride + 1)
        n_tail = self.n_planes - self.k0
        expected = data0 + self.k0 * self.stride + n_tail * self.plane_bytes
        size = os.path.getsize(path)
        if n_tail and size != expected:
            raise ValueError(f'file size {size:,} B != expected {expected:,} B '
                             f'({self.n_planes} planes, K0={self.k0}) — layout not recognised')
        if not n_tail and size < data0 + (self.n_planes - 1) * self.stride + self.plane_bytes:
            raise ValueError(f'file too short for {self.n_planes} planes')

        # stray IFDs of pages >= K0, written at (true offset mod 2**32)
        k = np.arange(self.k0, self.n_planes, dtype=np.int64)
        true_ifd = data0 + self.k0 * self.stride + (k - self.k0 + 1) * self.plane_bytes
        self.stray_starts = np.unique(true_ifd % _FOUR_GB)

    def plane_offset(self, k):
        if k < self.k0:
            return self.data0 + k * self.stride
        return self.data0 + self.k0 * self.stride + (k - self.k0) * self.plane_bytes

    def corrupted_pixels(self, k):
        """Flat pixel indices of plane k overwritten by stray IFDs."""
        a = self.plane_offset(k)
        b = a + self.plane_bytes
        i0 = np.searchsorted(self.stray_starts, a - self.ifd_bytes, side='right')
        i1 = np.searchsorted(self.stray_starts, b, side='left')
        spans = [np.arange((max(s, a) - a) // 4, (min(s + self.ifd_bytes, b) - a + 3) // 4)
                 for s in self.stray_starts[i0:i1]]
        return np.concatenate(spans) if spans else np.zeros(0, dtype=np.int64)

    def summary(self):
        return dict(data0=int(self.data0), stride=int(self.stride), ifd_bytes=int(self.ifd_bytes),
                    k0=int(self.k0), n_planes=int(self.n_planes),
                    n_stray_ifds=int(self.stray_starts.size))


def vista_layout(path, nbin=None, n_blocks=None, ny=None, nx=None):
    """VistaLayout of `path`; dimensions default to those in the file name."""
    info = parse_vista_name(path)
    return VistaLayout(path, nbin or info['nbin'], n_blocks or info['n_blocks'],
                       ny or info['ny'], nx or info['nx'])


def read_plane(fh, layout, k):
    """Plane k as float32 (ny, nx) photon counts; stray-IFD pixels set to 0."""
    fh.seek(layout.plane_offset(k))
    a = np.frombuffer(fh.read(layout.plane_bytes), '<f4').copy()
    a[layout.corrupted_pixels(k)] = 0
    return a.reshape(layout.ny, layout.nx)


# ── uint8 conversion ──────────────────────────────────────────────────────────
def convert_to_uint8(raw_path, out_dir, n_pos, xy_pixel_um=None, z_step_um=None,
                     overwrite=False, verbose=True):
    """
    Convert a raw Vista XYTZ TIFF into one uint8 ImageJ hyperstack per tile.

    Each tile file is (n_z, nbin, ny, nx), axes ZCYX (decay bins as channels, grayscale mode),
    memory-mappable with tifffile.memmap.  Also writes the per-tile intensity (Σ bins) and a
    JSON report.  Skips the work when all outputs exist and overwrite is False.

    Returns dict: tile_paths, tile_int (n_pos, n_z, ny, nx) float32, int_path, report.
    """
    info = parse_vista_name(raw_path)
    nbin, ny, nx, n_blocks, base = info['nbin'], info['ny'], info['nx'], info['n_blocks'], info['base']
    if n_blocks % n_pos:
        raise ValueError(f'{n_blocks} z-blocks cannot be split into {n_pos} tiles')
    n_z = n_blocks // n_pos

    os.makedirs(out_dir, exist_ok=True)
    tile_paths = [os.path.join(out_dir, f'{base}_pos{p}.tif') for p in range(n_pos)]
    int_path   = os.path.join(out_dir, f'{base}_tiles_intensity.tif')
    rep_path   = os.path.join(out_dir, f'{base}_conversion_report.json')

    if not overwrite and all(os.path.exists(p) for p in tile_paths + [int_path, rep_path]):
        with open(rep_path) as f:
            report = json.load(f)
        tile_int = tifffile.imread(int_path).astype(np.float32).reshape(n_pos, n_z, ny, nx)
        if verbose:
            print(f'Reusing converted tiles in {out_dir}')
        return dict(tile_paths=tile_paths, tile_int=tile_int, int_path=int_path, report=report)

    layout = VistaLayout(raw_path, nbin, n_blocks, ny, nx)
    hs_kw = dict(imagej=True, metadata={'axes': 'ZCYX', 'mode': 'grayscale'})
    if z_step_um:
        hs_kw['metadata'].update(spacing=float(z_step_um), unit='um')
    if xy_pixel_um:
        hs_kw['resolution'] = (1.0 / xy_pixel_um, 1.0 / xy_pixel_um)
        hs_kw['metadata']['unit'] = 'um'

    tile_int    = np.zeros((n_pos, n_z, ny, nx), np.float32)
    block_decay = np.zeros((n_pos, n_z, nbin), np.int64)
    n_corrupt = n_residual = n_clipped = 0
    max_count = 0.0
    if verbose:
        print(f'Converting {layout.n_planes} planes  ({n_pos} tiles x {n_z} z x {nbin} bins)  '
              f'K0={layout.k0}, {layout.stray_starts.size} stray IFDs')

    with open(raw_path, 'rb') as fh:
        for p in range(n_pos):
            mm = tifffile.memmap(tile_paths[p], shape=(n_z, nbin, ny, nx), dtype=np.uint8, **hs_kw)
            for z in range(n_z):
                for t in range(nbin):
                    k = (p * n_z + z) * nbin + t
                    fh.seek(layout.plane_offset(k))
                    a = np.frombuffer(fh.read(layout.plane_bytes), '<f4').copy()
                    bad = layout.corrupted_pixels(k)
                    a[bad] = 0
                    n_corrupt += bad.size
                    with np.errstate(invalid='ignore'):
                        ok = np.isfinite(a) & (a >= 0) & (a == np.round(a))
                    n_residual += int(a.size - ok.sum())
                    a[~ok] = 0
                    max_count = max(max_count, float(a.max()))
                    over = a > 255
                    n_clipped += int(over.sum())
                    a[over] = 255
                    plane = a.astype(np.uint8).reshape(ny, nx)
                    mm[z, t] = plane
                    tile_int[p, z] += plane
                    block_decay[p, z, t] = int(plane.sum(dtype=np.int64))
            mm.flush()
            del mm
            if verbose:
                print(f'  tile {p + 1}/{n_pos}  photons={block_decay[p].sum():,}  -> '
                      f'{os.path.basename(tile_paths[p])}')

    tifffile.imwrite(int_path, tile_int, imagej=True, metadata={'axes': 'TZYX'})
    report = dict(raw_path=raw_path, base=base, nbin=nbin, ny=ny, nx=nx, n_blocks=n_blocks,
                  n_pos=n_pos, n_z=n_z, layout=layout.summary(),
                  corrupted_pixels=int(n_corrupt),
                  corrupted_fraction=n_corrupt / float(layout.n_planes * ny * nx),
                  residual_junk_pixels=int(n_residual), clipped_pixels=int(n_clipped),
                  max_count=max_count,
                  photons_per_block=block_decay.sum(axis=2).tolist(),
                  block_decay=block_decay.tolist())
    with open(rep_path, 'w') as f:
        json.dump(report, f)
    if verbose:
        print(f'Corrupted (stray IFD) pixels zeroed: {n_corrupt:,} '
              f'({100 * report["corrupted_fraction"]:.3f}% of all bins)  |  '
              f'residual junk: {n_residual}  |  clipped >255: {n_clipped}  |  max count {max_count:g}')
    return dict(tile_paths=tile_paths, tile_int=tile_int, int_path=int_path, report=report)


# ── tile registration ─────────────────────────────────────────────────────────
def tile_grid_positions(layout, grid):
    """(n_tiles, 2) array of (row, col) per acquisition index.
    layout = '<raster|snake>_<tl|tr|bl|br>' (row-major, starting corner)."""
    order, corner = layout.split('_')
    n_rows, n_cols = grid
    rc = []
    for p in range(n_rows * n_cols):
        r, c = divmod(p, n_cols)
        if order == 'snake' and r % 2:
            c = n_cols - 1 - c
        if corner[1] == 'r':
            c = n_cols - 1 - c
        if corner[0] == 'b':
            r = n_rows - 1 - r
        rc.append((r, c))
    return np.array(rc)


def _ncc(a, b):
    a = a - a.mean()
    b = b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else 0.0


def _overlap_ncc(A, B, d):
    """NCC of the region shared by A and B when B's origin sits at offset d in A coords."""
    sa, sb = [], []
    for ax in range(3):
        lo = max(0, d[ax])
        hi = min(A.shape[ax], d[ax] + B.shape[ax])
        if hi - lo < 2:
            return -1.0
        sa.append(slice(lo, hi))
        sb.append(slice(lo - d[ax], hi - d[ax]))
    return _ncc(A[tuple(sa)], B[tuple(sb)])


def _register_pair(A, B, axis, step, search_px, max_dz, refine_px=2):
    """Offset (dz, dy, dx) of tile B's origin in tile A coords, B adjacent to A along `axis`
    (1 = below, 2 = right).  3D masked normalised cross-correlation on the overlap strips
    (no apodisation window: a window shared by both strips biases the peak to zero shift),
    then an NCC search over dz in [-max_dz, max_dz] and +-refine_px in y/x."""
    L   = A.shape[axis]
    ov  = int(round(L - step))
    w   = min(L, ov + search_px)
    sl_a = [slice(None)] * 3
    sl_b = [slice(None)] * 3
    sl_a[axis] = slice(L - w, L)
    sl_b[axis] = slice(0, w)
    As, Bs = A[tuple(sl_a)], B[tuple(sl_b)]
    min_overlap = max(0.05, 0.04 * L / w)   # true overlap may be much smaller than the strip
    shift = phase_cross_correlation(As, Bs, reference_mask=np.ones(As.shape, bool),
                                    moving_mask=np.ones(Bs.shape, bool), overlap_ratio=min_overlap)
    if isinstance(shift, tuple):
        shift = shift[0]
    d0 = np.round(shift).astype(int)
    d0[axis] += L - w

    nominal = np.zeros(3, int)
    nominal[axis] = int(round(step))
    if np.any(np.abs(d0[1:] - nominal[1:]) > search_px):   # phase correlation lost -> start at nominal
        d0[1:] = nominal[1:]

    best, best_d = -np.inf, None
    for dz in range(-max_dz, max_dz + 1):
        for dy in range(-refine_px, refine_px + 1):
            for dx in range(-refine_px, refine_px + 1):
                d = (dz, d0[1] + dy, d0[2] + dx)
                s = _overlap_ncc(A, B, d)
                if s > best:
                    best, best_d = s, d
    return np.array(best_d, int), best, nominal


def register_tiles(tile_int, grid, overlap=0.20, layout='auto', fallback='snake_tl', max_dz=2,
                   search_frac=0.20, norm_pct=(1, 99.5), min_score=0.15, ambiguity=0.03,
                   smooth_sigma=1.5, try_transposed=True, verbose=True):
    """
    Integer tile offsets for a tiled z-stack.

    tile_int : (n_pos, n_z, ny, nx) intensity per tile (acquisition order)
    grid     : (n_rows, n_cols)
    overlap  : nominal fractional overlap between neighbours (Olympus default 0.20); the true
               step may differ by up to search_frac * tile size
    layout   : 'auto' or one of LAYOUTS; 'auto' scores all layouts (and, with try_transposed, the
               (n_cols, n_rows) grid too) by mean pair NCC.  `fallback` is used when nothing
               matches (best < min_score) or when it scores within `ambiguity` of the best.

    Returns dict: layout, grid, layout_scores, grid_rc, pairs, positions (n_pos, 3) int
    (z0, y0, x0), canvas_shape, norm_lo, norm_hi, overlap, step, tile_shape.
    """
    n_pos, n_z, ny, nx = tile_int.shape
    grid = tuple(int(g) for g in grid)
    if n_pos != grid[0] * grid[1]:
        raise ValueError(f'{n_pos} tiles do not fit a {grid[0]}x{grid[1]} grid')
    step = (ny * (1 - overlap), nx * (1 - overlap))
    search_px = int(round(search_frac * max(ny, nx)))

    norm_lo = np.array([np.percentile(t, norm_pct[0]) for t in tile_int], float)
    norm_hi = np.array([np.percentile(t, norm_pct[1]) for t in tile_int], float)
    tiles_n = [gaussian_filter((t - lo) / max(hi - lo, 1e-9), sigma=(0, smooth_sigma, smooth_sigma))
               for t, lo, hi in zip(tile_int, norm_lo, norm_hi)]

    cache = {}

    def pairs_for(g, lay):
        rc = tile_grid_positions(lay, g)
        where = {(int(r), int(c)): p for p, (r, c) in enumerate(rc)}
        out = []
        for (r, c), a in where.items():
            for axis, (rr, cc) in ((1, (r + 1, c)), (2, (r, c + 1))):
                b = where.get((rr, cc))
                if b is None:
                    continue
                key = (a, b, axis)
                if key not in cache:
                    cache[key] = _register_pair(tiles_n[a], tiles_n[b], axis, step[axis - 1],
                                                search_px, max_dz)
                d, score, nominal = cache[key]
                out.append(dict(a=a, b=b, axis=axis, offset=d.tolist(), score=score,
                                nominal=nominal.tolist()))
        return rc, out

    grids = [grid]
    if layout == 'auto' and try_transposed and grid[0] != grid[1]:
        grids.append(grid[::-1])
    candidates = [(g, lay) for g in grids for lay in (LAYOUTS if layout == 'auto' else [layout])]
    scores = {}
    for g, lay in candidates:
        _, prs = pairs_for(g, lay)
        scores[(g, lay)] = float(np.mean([p['score'] for p in prs]))
    chosen = max(scores, key=scores.get)
    if layout == 'auto':
        fb = (grid, fallback)
        if scores[chosen] < min_score:
            if verbose:
                print(f'Layout auto-detect: no layout matches (best {chosen[1]} {chosen[0]} = '
                      f'{scores[chosen]:.3f}) -> fallback {fallback} {grid}')
            chosen = fb
        elif fb != chosen and scores[fb] >= scores[chosen] - ambiguity:
            if verbose:
                print(f'Layout auto-detect: fallback {fallback} within {ambiguity} of best -> using it')
            chosen = fb
    grid, chosen_layout = chosen
    rc, pairs = pairs_for(grid, chosen_layout)

    # ── global weighted least squares, tile 0 fixed ──────────────────────────
    rows, rhs, wts = [], [], []
    for pr in pairs:
        good = pr['score'] >= min_score
        pr['used'] = 'measured' if good else 'nominal'
        d = np.array(pr['offset'] if good else pr['nominal'], float)
        row = np.zeros(n_pos)
        row[pr['b']], row[pr['a']] = 1, -1
        rows.append(row); rhs.append(d); wts.append(pr['score'] if good else 0.5 * min_score)
    anchor = np.zeros(n_pos); anchor[0] = 1
    rows.append(anchor); rhs.append(np.zeros(3)); wts.append(10.0)
    W = np.sqrt(np.array(wts))[:, np.newaxis]
    sol, *_ = np.linalg.lstsq(np.array(rows) * W, np.array(rhs) * W, rcond=None)
    positions = np.round(sol).astype(int)
    positions -= positions.min(axis=0)
    canvas = tuple(int(v) for v in (positions + np.array([n_z, ny, nx])).max(axis=0))
    for pr in pairs:   # disagreement between the pair measurement and the global solution
        pr['residual'] = (positions[pr['b']] - positions[pr['a']] - np.array(pr['offset'])).tolist()

    if verbose:
        if layout == 'auto':
            print('Layout scores (mean pair NCC):  ' +
                  '  '.join(f'{lay}{g[0]}x{g[1]}={v:.3f}'
                            for (g, lay), v in sorted(scores.items(), key=lambda kv: -kv[1])[:6]))
        print(f'Layout: {chosen_layout}  grid {grid[0]} rows x {grid[1]} cols   '
              f'nominal step = {step[0]:.1f} x {step[1]:.1f} px')
        for pr in pairs:
            print(f'  pair {pr["a"]}->{pr["b"]}  {"down " if pr["axis"] == 1 else "right"}  '
                  f'offset (dz,dy,dx)={tuple(int(v) for v in pr["offset"])}  NCC={pr["score"]:.3f}  '
                  f'[{pr["used"]}]  residual={tuple(pr["residual"])}')
        for p in range(n_pos):
            print(f'  tile {p}  grid (r,c)={tuple(int(v) for v in rc[p])}  '
                  f'origin (z0,y0,x0)={tuple(int(v) for v in positions[p])}')
        print(f'Canvas (z, y, x) = {canvas}')

    return dict(layout=chosen_layout, grid=list(grid),
                layout_scores={f'{lay}_{g[0]}x{g[1]}': v for (g, lay), v in scores.items()},
                grid_rc=rc.tolist(), pairs=pairs, positions=positions.tolist(),
                canvas_shape=list(canvas), norm_lo=norm_lo.tolist(), norm_hi=norm_hi.tolist(),
                overlap=overlap, step=list(step), tile_shape=[n_z, ny, nx])


def save_tile_positions(path, reg, **extra):
    with open(path, 'w') as f:
        json.dump(dict(reg, **extra), f, indent=1)


def load_tile_positions(path):
    with open(path) as f:
        return json.load(f)


# ── fusion ────────────────────────────────────────────────────────────────────
def fuse_tiles(tile_paths, reg, tile_int, out_dir, base, overwrite=False, verbose=True):
    """
    Stitch tiles into the canvas given by `reg` (register_tiles result).

    TCSPC: overlaps are summed (raw photon counts, no rescaling) and written one TYX TIFF per
    fused z to <out_dir>/stitched/<base>_stitched_Z###.tif (uint8, uint16 if a plane overflows).
    Segmentation intensity: per-tile normalised (reg norm_lo/hi), feather-blended in xy,
    written to <out_dir>/<base>_stitched_intensity.tif (z, y, x) float32, with
    <out_dir>/<base>_coverage.tif (number of tiles per voxel).

    Returns dict: stitched_dir, plane_paths, int_path, cov_path, photons_fused, photons_tiles.
    """
    stitched_dir = os.path.join(out_dir, 'stitched')
    int_path     = os.path.join(out_dir, f'{base}_stitched_intensity.tif')
    cov_path     = os.path.join(out_dir, f'{base}_coverage.tif')
    Zc, Yc, Xc   = reg['canvas_shape']
    plane_paths  = [os.path.join(stitched_dir, f'{base}_stitched_Z{z:03d}.tif') for z in range(Zc)]

    existing = set(glob.glob(os.path.join(stitched_dir, f'{base}_stitched_Z*.tif')))
    if (not overwrite and existing == set(plane_paths)
            and os.path.exists(int_path) and os.path.exists(cov_path)):
        if verbose:
            print(f'Reusing stitched planes in {stitched_dir}')
        return dict(stitched_dir=stitched_dir, plane_paths=plane_paths, int_path=int_path,
                    cov_path=cov_path, photons_fused=None,
                    photons_tiles=float(tile_int.sum(dtype=np.float64)))

    os.makedirs(stitched_dir, exist_ok=True)
    for old in existing:
        os.remove(old)   # stale planes from a previous canvas size

    tiles = [tifffile.memmap(p, mode='r') for p in tile_paths]
    n_z, nbin, ny, nx = tiles[0].shape
    pos  = np.array(reg['positions'])
    lo, hi = np.array(reg['norm_lo']), np.array(reg['norm_hi'])
    yy, xx = np.mgrid[0:ny, 0:nx]
    feather = np.minimum(np.minimum(yy + 1, ny - yy), np.minimum(xx + 1, nx - xx)).astype(np.float32)

    fused_int = np.zeros((Zc, Yc, Xc), np.float32)
    coverage  = np.zeros((Zc, Yc, Xc), np.uint8)
    photons, n_uint16 = 0, 0
    for z in range(Zc):
        acc = np.zeros((nbin, Yc, Xc), np.uint16)
        num = np.zeros((Yc, Xc), np.float32)
        den = np.zeros((Yc, Xc), np.float32)
        for p, tile in enumerate(tiles):
            tz = z - pos[p, 0]
            if not 0 <= tz < n_z:
                continue
            y0, x0 = pos[p, 1], pos[p, 2]
            acc[:, y0:y0 + ny, x0:x0 + nx] += tile[tz]
            norm = (tile_int[p, tz] - lo[p]) / max(hi[p] - lo[p], 1e-9)
            num[y0:y0 + ny, x0:x0 + nx] += feather * norm
            den[y0:y0 + ny, x0:x0 + nx] += feather
            coverage[z, y0:y0 + ny, x0:x0 + nx] += 1
        np.divide(num, den, out=fused_int[z], where=den > 0)
        photons += int(acc.sum(dtype=np.int64))
        if acc.max() > 255:
            n_uint16 += 1
            out = acc
        else:
            out = acc.astype(np.uint8)
        tifffile.imwrite(plane_paths[z], out, metadata={'axes': 'TYX'})
        if verbose:
            print(f'  z={z:03d}/{Zc - 1}  tiles={int(coverage[z].max())}  -> '
                  f'{os.path.basename(plane_paths[z])}', end='\r', flush=True)
    del tiles

    tifffile.imwrite(int_path, fused_int)
    tifffile.imwrite(cov_path, coverage)
    photons_tiles = float(tile_int.sum(dtype=np.float64))
    if verbose:
        print(f'\nStitched {Zc} planes  ({Zc} x {nbin} x {Yc} x {Xc})  -> {stitched_dir}')
        print(f'Photons: fused {photons:,}  vs tiles {photons_tiles:,.0f}  '
              f'({"OK" if photons == photons_tiles else "MISMATCH"})'
              + (f'  |  {n_uint16} plane(s) stored as uint16' if n_uint16 else ''))
    return dict(stitched_dir=stitched_dir, plane_paths=plane_paths, int_path=int_path,
                cov_path=cov_path, photons_fused=photons, photons_tiles=photons_tiles)
