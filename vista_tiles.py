"""
vista_tiles.py
Readers and tile tools for ISS Vista multi-page XYTZ TIFFs (AnalyzeTCSPC_XYTZTile.ipynb,
AnalyzeTCSPC_XYTZTime.ipynb for time series).

Vista writes classic (32-bit offset) TIFFs even above 4 GB.  Layout of the 2026-10 exports:
    pages 0 .. K0-1   [plane data][IFD]     stride = plane_bytes + ifd_bytes
    pages K0 .. N-1   [plane data]          stride = plane_bytes
K0 = first page whose IFD would start beyond 4 GB.  The IFDs of pages >= K0 are written at
(true offset mod 2**32), i.e. on top of plane data in the first 4 GB.  Those spans are computed
exactly by `VistaLayout` and zeroed on read (the photons there are lost).
Plane order in the file: [timepoint][tile position][z][decay bin].

Functions
---------
parse_vista_name     -- nbin / ny / nx / n_blocks / channel from the Vista file name
vista_layout         -- byte layout of a Vista TIFF (plane offsets + stray-IFD spans)
read_plane           -- one decay-bin plane as float32 photon counts (stray-IFD pixels zeroed)
convert_to_uint8     -- raw Vista TIFF -> one uint8 ZCYX hyperstack per tile + tile intensity
tile_grid_positions  -- (row, col) of every tile for a named acquisition layout
register_tiles       -- pairwise 3D phase correlation + global least squares -> integer offsets
register_timepoints  -- integer (dz, dy, dx) drift of every timepoint onto a reference timepoint
time_positions       -- tile origins of every timepoint on one common canvas
save_tile_positions / load_tile_positions -- JSON round trip of the registration result
fuse_tiles           -- stitched per-z TYX TCSPC TIFFs (overlaps summed) + segmentation intensity
photon_sum_reference -- segmentation reference: Σ photons / Σ brightness of the timepoints x tiles per voxel
move_legacy_outputs  -- move Step-0 outputs of the old flat layout (loose in the raw folder) into analysis/
channel_offset       -- sub-pixel (dy, dx) between two detectors that recorded the same tiles
fuse_intensity       -- stitch another intensity channel of the same tiles with the same positions
align_blocks         -- true tile / z of every recorded z-block when Vista dropped planes, from a
                        complete reference recorded in parallel (e.g. the Olympus PMT tiles)
find_fbs_xml         -- the Vista acquisition settings (<name>.fbs.xml) of a raw export
find_raw             -- the raw Vista TIFF of an acquisition folder (or its name, once deleted after conversion)
delete_raw_if_lossless -- delete the float32 raw TIFF once its uint8 tiles are verified to hold all its photons
"""

import os
import re
import glob
import json
import struct
import warnings
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


def find_fbs_xml(raw_path):
    """Vista acquisition settings of a raw export: '<name>.fbs.xml' in the raw file's folder or its parent,
    where <name> is a prefix of the raw file name (longest wins).  None if absent."""
    name = os.path.basename(raw_path)
    folder = os.path.dirname(os.path.abspath(raw_path))
    for d in (folder, os.path.dirname(folder)):
        hits = [p for p in glob.glob(os.path.join(glob.escape(d), '*.fbs.xml'))
                if name.startswith(os.path.basename(p)[:-len('.fbs.xml')])]
        if hits:
            return max(hits, key=len)
    return None


_VISTA_STEM = re.compile(r'_Decay\d+_\d+x\d+_Z\d+_Ch\d+_T\d+_P\d+_S\d+$')
_REPORT_DIRS = (os.path.join('analysis', 'tiles_uint8'), 'tiles_uint8')   # XYTZTime / XYTZTile layout


def find_raw(path, report_dirs=_REPORT_DIRS):
    """
    The raw Vista XYTZ TIFF of an acquisition: `path` is the acquisition folder (or the raw file itself).

    In the folder, the raw file is the .tif whose name ends in the Vista suffix
    (_Decay<n>_<nx>x<ny>_Z<n>_Ch<c>_T<t>_P<p>_S<s>); outputs named after it (..._stitched_intensity.tif) are not
    candidates. Without one (the raw file was deleted after its uint8 conversion, delete_raw_if_lossless), the name
    comes from the conversion report below the folder (<report_dir>/<base>_conversion_report.json).
    Returns (raw_path, exists): where the raw file is or was. Its name carries the acquisition dimensions
    (parse_vista_name) and locates the .fbs.xml (find_fbs_xml), so both still work after the deletion.
    FileNotFoundError when neither is found; ValueError when the folder holds several acquisitions.
    """
    if os.path.isfile(path):
        return path, True
    if path.lower().endswith(('.tif', '.tiff')):          # a raw file that is gone: its folder's report
        folder, want = os.path.dirname(path), os.path.splitext(os.path.basename(path))[0]
    else:
        folder, want = path, None
    if not os.path.isdir(folder):
        raise FileNotFoundError(f'{path} does not exist')
    raws = sorted(p for ext in ('*.tif', '*.tiff') for p in glob.glob(os.path.join(glob.escape(folder), ext))
                  if _VISTA_STEM.search(os.path.splitext(os.path.basename(p))[0]))
    reports = {}
    for d in report_dirs:
        for r in glob.glob(os.path.join(glob.escape(os.path.join(folder, d)), '*_conversion_report.json')):
            reports.setdefault(os.path.basename(r)[:-len('_conversion_report.json')], r)
    if want is not None:
        raws = [p for p in raws if os.path.splitext(os.path.basename(p))[0] == want]
        reports = {b: r for b, r in reports.items() if b == want}
    if len(raws) == 1:
        return raws[0], True
    if len(raws) > 1:
        raise ValueError(f'{len(raws)} raw Vista TIFFs in {folder}: '
                         + ', '.join(os.path.basename(p) for p in raws) + ' - select the file instead')
    if len(reports) == 1:
        base, rep_path = next(iter(reports.items()))
        with open(rep_path) as f:
            ext = os.path.splitext(json.load(f).get('raw_path') or '.tif')[1] or '.tif'
        return os.path.join(folder, base + ext), False
    if len(reports) > 1:
        raise ValueError(f'{len(reports)} converted acquisitions below {folder}, no raw file: '
                         + ', '.join(sorted(reports)) + ' - give the raw file path (it may be deleted)')
    raise FileNotFoundError(f'No raw Vista TIFF (<name>_Decay<n>_<nx>x<ny>_Z<n>_Ch1_T1_P1_S1.tif) and no uint8 '
                            f'conversion report ({" or ".join(report_dirs)}) in {folder}')


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
                     overwrite=False, verbose=True, n_t=1):
    """
    Convert a raw Vista XYTZ TIFF into one uint8 ImageJ hyperstack per tile (and timepoint).

    Each tile file is (n_z, nbin, ny, nx), axes ZCYX (decay bins as channels, grayscale mode),
    memory-mappable with tifffile.memmap.  Also writes the per-tile intensity (Σ bins) and a
    JSON report.  Skips the work when all outputs exist and overwrite is False.
    n_t: timepoints (repeated z-stacks of all tiles); blocks are ordered [timepoint][tile][z] and the
    files are <base>_pos<p>.tif (n_t = 1) or <base>_t<tt>_pos<p>.tif, in that order.

    Returns dict: tile_paths (n_t * n_pos), tile_int (n_t * n_pos, n_z, ny, nx) float32, int_path, report.
    """
    info = parse_vista_name(raw_path)
    nbin, ny, nx, n_blocks, base = info['nbin'], info['ny'], info['nx'], info['n_blocks'], info['base']
    n_files = n_t * n_pos
    if n_blocks % n_files:
        raise ValueError(f'{n_blocks} z-blocks cannot be split into {n_t} timepoints x {n_pos} tiles')
    n_z = n_blocks // n_files

    os.makedirs(out_dir, exist_ok=True)
    tile_paths = ([os.path.join(out_dir, f'{base}_pos{p}.tif') for p in range(n_pos)] if n_t == 1 else
                  [os.path.join(out_dir, f'{base}_t{t:02d}_pos{p}.tif') for t in range(n_t) for p in range(n_pos)])
    int_path   = os.path.join(out_dir, f'{base}_tiles_intensity.tif')
    rep_path   = os.path.join(out_dir, f'{base}_conversion_report.json')

    if not overwrite and all(os.path.exists(p) for p in tile_paths + [int_path, rep_path]):
        with open(rep_path) as f:
            report = json.load(f)
        if report.get('n_z') == n_z and report.get('n_t', 1) == n_t:
            tile_int = tifffile.imread(int_path).astype(np.float32).reshape(n_files, n_z, ny, nx)
            if verbose:
                print(f'Reusing converted tiles in {out_dir}')
            return dict(tile_paths=tile_paths, tile_int=tile_int, int_path=int_path, report=report)

    if not os.path.exists(raw_path):
        _have = all(os.path.exists(p) for p in tile_paths + [int_path, rep_path])
        raise FileNotFoundError(f'{raw_path} is gone (deleted after an earlier conversion?) and the uint8 tiles in '
                                f'{out_dir} are ' + ('for another number of timepoints / tiles, or overwrite=True'
                                                     if _have else 'incomplete') + ': nothing to convert from')
    layout = VistaLayout(raw_path, nbin, n_blocks, ny, nx)
    hs_kw = dict(imagej=True, metadata={'axes': 'ZCYX', 'mode': 'grayscale'})
    if z_step_um:
        hs_kw['metadata'].update(spacing=float(z_step_um), unit='um')
    if xy_pixel_um:
        hs_kw['resolution'] = (1.0 / xy_pixel_um, 1.0 / xy_pixel_um)
        hs_kw['metadata']['unit'] = 'um'

    tile_int    = np.zeros((n_files, n_z, ny, nx), np.float32)
    block_decay = np.zeros((n_files, n_z, nbin), np.int64)
    n_corrupt = n_residual = n_clipped = 0
    max_count = 0.0
    if verbose:
        print(f'Converting {layout.n_planes} planes  ({f"{n_t} timepoints x " if n_t > 1 else ""}'
              f'{n_pos} tiles x {n_z} z x {nbin} bins)  K0={layout.k0}, {layout.stray_starts.size} stray IFDs')

    with open(raw_path, 'rb') as fh:
        for p in range(n_files):
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
                print(f'  {f"t{p // n_pos:02d} " if n_t > 1 else ""}tile {p % n_pos + 1}/{n_pos}  '
                      f'photons={block_decay[p].sum():,}  -> {os.path.basename(tile_paths[p])}')

    tifffile.imwrite(int_path, tile_int, imagej=True, metadata={'axes': 'TZYX'})
    report = dict(raw_path=raw_path, raw_bytes=os.path.getsize(raw_path), base=base, nbin=nbin, ny=ny, nx=nx, n_blocks=n_blocks,
                  n_pos=n_pos, n_t=n_t, n_z=n_z, layout=layout.summary(),
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


def delete_raw_if_lossless(raw_path, conv, max_junk=100, verbose=True):
    """
    Delete the float32 Vista TIFF once its uint8 tiles (convert_to_uint8 result `conv`) hold all of its photons.

    Deleted only when
      - no count was clipped at 255 (report clipped_pixels == 0);
      - at most max_junk bins held a value that is not a photon count (negative, non-integer, not finite; set to 0
        by the conversion). Every Vista file converted up to 2026-11 has 2; a wrong byte layout gives millions.
        Bins overwritten by Vista's stray IFDs (corrupted_pixels) are no reason to keep it: the raw file holds the
        IFD bytes there too;
      - the raw file is not newer than the conversion report and, for reports written since 2026-11-24, has the
        size recorded at conversion;
      - every uint8 tile file, read back, gives the photon count per z-block and bin recorded during the conversion.
    The .fbs.xml and every other file stay. Returns (deleted, message).
    """
    rep = conv['report']
    rep_path = os.path.join(os.path.dirname(conv['int_path']), f'{rep["base"]}_conversion_report.json')
    if not os.path.isfile(raw_path):
        return False, f'{raw_path} not found'
    if os.path.splitext(os.path.basename(raw_path))[0] != rep['base']:
        return False, f'the tiles are of {rep["base"]}, not of {os.path.basename(raw_path)}'
    if rep.get('clipped_pixels', 1):
        return False, f'{rep.get("clipped_pixels")} bins clipped at 255 (counts > 255 per bin and pixel)'
    if rep.get('residual_junk_pixels', max_junk + 1) > max_junk:
        return False, f'{rep.get("residual_junk_pixels")} bins with values that are not photon counts (> {max_junk})'
    if os.path.getmtime(raw_path) > os.path.getmtime(rep_path):
        return False, 'the raw file is newer than its conversion: convert again (rerun_convert = True)'
    size = os.path.getsize(raw_path)
    if rep.get('raw_bytes') not in (None, size):
        return False, f'raw file has {size:,} B, {rep["raw_bytes"]:,} B at conversion'
    if 'block_decay' not in rep:
        return False, 'the conversion report has no per-block counts to verify the tiles: rerun_convert = True'
    expect = np.asarray(rep['block_decay'], dtype=np.int64)          # (files, z, bin)
    if expect.shape[0] != len(conv['tile_paths']):
        return False, f'{len(conv["tile_paths"])} tile files, {expect.shape[0]} in the report'
    if verbose:
        print(f'Verifying {len(conv["tile_paths"])} uint8 tile file(s) against the conversion report...', flush=True)
    for p, path in enumerate(conv['tile_paths']):
        a = tifffile.memmap(path, mode='r')
        got = np.stack([a[z].reshape(a.shape[1], -1).sum(axis=1, dtype=np.int64) for z in range(a.shape[0])])
        del a
        if got.shape != expect[p].shape or not np.array_equal(got, expect[p]):
            return False, f'{os.path.basename(path)} does not hold the photons recorded at conversion: rerun_convert = True'
    try:
        os.remove(raw_path)
    except OSError as e:
        return False, f'could not delete it ({e}); is it open in another program?'
    return True, (f'{os.path.basename(raw_path)} ({size / 1e9:.1f} GB freed): no clipped counts, '
                  f'{rep.get("residual_junk_pixels", 0)} non-count bins, every tile file verified')


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
    if n_pos == 1:   # single position: nothing to register, the tile is the canvas
        if verbose:
            print(f'Single tile: no registration  |  canvas (z, y, x) = {(n_z, ny, nx)}')
        return dict(layout='single', grid=list(grid), layout_scores={}, grid_rc=[[0, 0]], pairs=[],
                    positions=[[0, 0, 0]], canvas_shape=[n_z, ny, nx], norm_lo=norm_lo.tolist(),
                    norm_hi=norm_hi.tolist(), overlap=overlap, step=list(step), tile_shape=[n_z, ny, nx])
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
    at_bound = [f'{pr["a"]}->{pr["b"]}' for pr in pairs if pr['used'] == 'measured' and abs(pr['offset'][0]) >= max_dz]
    if at_bound:
        warnings.warn(f'tile pair(s) {", ".join(at_bound)} at the z search limit (|dz| = max_dz = {max_dz}): '
                      f'the true offset may be larger, increase max_dz', RuntimeWarning)

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


def register_timepoints(tile_int_t, ref=0, max_dz=4, refine_px=2, norm_pct=(1, 99.5), smooth_sigma=1.5,
                        min_score=0.15, verbose=True):
    """
    Integer drift of repeated z-stacks of the same tiles (time-lapse) relative to timepoint `ref`.

    tile_int_t : (n_t, n_pos, n_z, ny, nx) intensity per timepoint and tile
    Per tile: xy from masked phase correlation of the z max-projections, then an NCC search over
    dz in [-max_dz, max_dz] (at that xy), +-refine_px in y/x (at the best dz) and dz +-1 again.
    The shift of a timepoint is the median over the tiles that match (score >= min_score): the
    sample moves as a whole.  shift = origin of timepoint t in the coordinates of timepoint `ref`
    (the convention of register_tiles pair offsets), so tile p of timepoint t sits at
    positions[p] + shift[t] (time_positions).

    Returns dict: ref, shifts (n_t, 3) int, ncc (n_t,) at the shift, ncc0 (n_t,) without correction,
    per_tile (n_t, n_pos, 3), per_tile_ncc (n_t, n_pos), max_dz.
    """
    n_t, n_pos = tile_int_t.shape[:2]

    def prep(v):
        lo, hi = np.percentile(v, norm_pct)
        return gaussian_filter((v - lo) / max(hi - lo, 1e-9), sigma=(0, smooth_sigma, smooth_sigma))

    def best_of(A, B, cands, best):
        for d in cands:
            s = _overlap_ncc(A, B, d)
            if s > best[0]:
                best = (s, d)
        return best

    refs     = [prep(tile_int_t[ref, p]) for p in range(n_pos)]
    per_tile = np.zeros((n_t, n_pos, 3), int)
    per_ncc  = np.zeros((n_t, n_pos))
    shifts   = np.zeros((n_t, 3), int)
    ncc, ncc0 = np.zeros(n_t), np.zeros(n_t)
    for t in range(n_t):
        movs = refs if t == ref else [prep(tile_int_t[t, p]) for p in range(n_pos)]
        ncc0[t] = np.mean([_overlap_ncc(A, B, (0, 0, 0)) for A, B in zip(refs, movs)])
        if t == ref:
            per_ncc[t] = ncc[t] = ncc0[t]
            continue
        for p, (A, B) in enumerate(zip(refs, movs)):
            Am, Bm = A.max(axis=0), B.max(axis=0)
            sh = phase_cross_correlation(Am, Bm, reference_mask=np.ones(Am.shape, bool),
                                         moving_mask=np.ones(Bm.shape, bool), overlap_ratio=0.3)
            dy, dx = np.round(sh[0] if isinstance(sh, tuple) else sh).astype(int)
            best = best_of(A, B, [(dz, dy, dx) for dz in range(-max_dz, max_dz + 1)], (-np.inf, None))
            dz = best[1][0]
            best = best_of(A, B, [(dz, dy + a, dx + b) for a in range(-refine_px, refine_px + 1)
                                  for b in range(-refine_px, refine_px + 1)], best)
            dz, dy, dx = best[1]
            best = best_of(A, B, [(dz + c, dy, dx) for c in (-1, 1) if abs(dz + c) <= max_dz], best)
            per_ncc[t, p], per_tile[t, p] = best[0], best[1]
        good = per_ncc[t] >= min_score
        shifts[t] = np.round(np.median(per_tile[t][good if good.any() else slice(None)], axis=0)).astype(int)
        ncc[t] = np.mean([_overlap_ncc(A, B, shifts[t]) for A, B in zip(refs, movs)])
    weak = [t for t in range(n_t) if t != ref and ncc[t] < min_score]
    if weak:
        warnings.warn(f'timepoint(s) {weak} match timepoint {ref} poorly (NCC < {min_score}) after drift correction: '
                      f'check the drift QC', RuntimeWarning)
    at_bound = [t for t in range(n_t) if abs(shifts[t, 0]) >= max_dz and t != ref]
    if at_bound:
        warnings.warn(f'timepoint(s) {at_bound} at the z drift limit (|dz| = {max_dz}): the true drift may be '
                      f'larger, increase the max dz', RuntimeWarning)
    if verbose:
        for t in range(n_t):
            print(f'  t{t:02d}  drift (dz,dy,dx)={tuple(int(v) for v in shifts[t])}  '
                  f'NCC {ncc0[t]:.3f} -> {ncc[t]:.3f}' + ('  [reference]' if t == ref else ''))
    return dict(ref=int(ref), shifts=shifts.tolist(), ncc=ncc.tolist(), ncc0=ncc0.tolist(),
                per_tile=per_tile.tolist(), per_tile_ncc=per_ncc.tolist(), max_dz=int(max_dz))


def time_positions(reg, shifts):
    """Tile origins of every timepoint on one canvas that holds them all.
    reg: register_tiles result; shifts: (n_t, 3) register_timepoints drift.
    Returns (positions (n_t, n_pos, 3) int, canvas_shape [z, y, x])."""
    pos = np.array(reg['positions'], int)[np.newaxis] + np.asarray(shifts, int)[:, np.newaxis]
    pos -= pos.reshape(-1, 3).min(axis=0)
    canvas = (pos.reshape(-1, 3) + np.array(reg['tile_shape'])).max(axis=0)
    return pos, [int(v) for v in canvas]


def save_tile_positions(path, reg, **extra):
    with open(path, 'w') as f:
        json.dump(dict(reg, **extra), f, indent=1)


def load_tile_positions(path):
    with open(path) as f:
        return json.load(f)


# ── fusion ────────────────────────────────────────────────────────────────────
def _feather(ny, nx):
    """Blending weight: distance (px) to the nearest tile edge."""
    yy, xx = np.mgrid[0:ny, 0:nx]
    return np.minimum(np.minimum(yy + 1, ny - yy), np.minimum(xx + 1, nx - xx)).astype(np.float32)


def photon_sum_reference(tile_int_t, positions, canvas_shape):
    """
    Segmentation reference of a time series (or one acquisition, n_t = 1) on the common canvas:
    Σ photons / Σ b over the timepoints x tiles covering each voxel, tile overlaps feather-blended.

    b = mean photons of a timepoint x tile on the tissue all timepoints of that tile share (their common box,
    so z drift does not bias it; planes never recorded are left out).  Where every timepoint covers a voxel the
    reference is the photon sum / a constant: Poisson-optimal, brighter timepoints weigh more (e.g. Peredox
    brightening on NADH binding), nothing is rescaled per timepoint.  At the drift edges, covered by fewer
    timepoints, it keeps the same level where a plain sum would dim.  Planes never recorded get no weight.

    tile_int_t   : (n_t, n_tiles, z, y, x) photons summed over the decay bins
    positions    : (n_t, n_tiles, 3) canvas origin (z, y, x) of every timepoint x tile
    canvas_shape : (z, y, x)
    Returns ref (float32, canvas; 0 where nothing was recorded), ntp (uint8: timepoints x tiles recorded per
    voxel), b (n_t, n_tiles).
    """
    tile_int_t = np.asarray(tile_int_t)
    pos = np.asarray(positions, int).reshape(tile_int_t.shape[0], tile_int_t.shape[1], 3)
    can = tuple(int(v) for v in canvas_shape)
    shp = np.array(tile_int_t.shape[2:])
    ok = tile_int_t.reshape(*tile_int_t.shape[:3], -1).max(axis=3) > 0      # (time, tile, z): plane recorded
    n_t, n_p = tile_int_t.shape[:2]

    b = np.full((n_t, n_p), np.nan)
    for p in range(n_p):
        s, e = pos[:, p].max(axis=0), pos[:, p].min(axis=0) + shp           # common box of the timepoints
        for t in range(n_t):
            v = tile_int_t[t, p]
            a, c = s - pos[t, p], e - pos[t, p]
            r = (v[a[0]:c[0], a[1]:c[1], a[2]:c[2]][ok[t, p, a[0]:c[0]]] if np.all(c > a) else v[ok[t, p]])
            if r.size:
                b[t, p] = r.mean()
    b[~(b > 0)] = np.nanmedian(b[b > 0]) if np.any(b > 0) else 1.0

    fe = _feather(*shp[1:])
    num = np.zeros(can, np.float32)
    den = np.zeros(can, np.float32)
    ntp = np.zeros(can, np.uint8)
    for t in range(n_t):
        for p in range(n_p):
            z0, y0, x0 = pos[t, p]
            sl = (slice(z0, z0 + shp[0]), slice(y0, y0 + shp[1]), slice(x0, x0 + shp[2]))
            w = ok[t, p][:, None, None]
            num[sl] += fe * tile_int_t[t, p]
            den[sl] += fe * w * np.float32(b[t, p])
            ntp[sl] += w.astype(np.uint8)
    ref = np.divide(num, den, out=np.zeros(can, np.float32), where=den > 0)
    return ref, ntp, b


def fuse_tiles(tile_paths, reg, tile_int, out_dir, base, overwrite=False, verbose=True, block_of_slot=None,
               stitched_name='stitched', int_dir=None):
    """
    Stitch tiles into the canvas given by `reg` (register_tiles result).

    TCSPC: overlaps are summed (raw photon counts, no rescaling) and written one TYX TIFF per
    fused z to <out_dir>/<stitched_name>/<base>_stitched_Z###.tif (uint8, uint16 if a plane overflows);
    stitched_name may be a relative path (e.g. 'stitched_timepoints/T00') or an absolute folder.
    Several timepoints of a time-lapse are summed the same way: pass all their tiles, with
    positions from time_positions, flattened (time-major).
    Segmentation intensity: per-tile normalised (reg norm_lo/hi), feather-blended in xy,
    written to <int_dir>/<base>_stitched_intensity.tif (z, y, x) float32, with
    <int_dir>/<base>_coverage.tif (number of tiles per voxel); int_dir None = out_dir.
    block_of_slot: (n_tiles * n_z,) file block shown at each true slot, -1 = plane never recorded
    (align_blocks); None = file layout.  tile_int must then be in the true layout as well.

    Returns dict: stitched_dir, plane_paths, int_path, cov_path, photons_fused, photons_tiles.
    """
    stitched_dir = os.path.join(out_dir, stitched_name)
    int_dir      = int_dir or out_dir
    int_path     = os.path.join(int_dir, f'{base}_stitched_intensity.tif')
    cov_path     = os.path.join(int_dir, f'{base}_coverage.tif')
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
    os.makedirs(int_dir, exist_ok=True)
    for old in existing:
        os.remove(old)   # stale planes from a previous canvas size

    tiles = [tifffile.memmap(p, mode='r') for p in tile_paths]
    n_z, nbin, ny, nx = tiles[0].shape
    pos  = np.array(reg['positions'])
    lo, hi = np.array(reg['norm_lo']), np.array(reg['norm_hi'])
    feather = _feather(ny, nx)

    fused_int = np.zeros((Zc, Yc, Xc), np.float32)
    coverage  = np.zeros((Zc, Yc, Xc), np.uint8)
    photons, n_uint16 = 0, 0
    for z in range(Zc):
        acc = np.zeros((nbin, Yc, Xc), np.uint16)
        num = np.zeros((Yc, Xc), np.float32)
        den = np.zeros((Yc, Xc), np.float32)
        for p in range(len(tiles)):
            tz = z - pos[p, 0]
            if not 0 <= tz < n_z:
                continue
            src = p * n_z + tz if block_of_slot is None else int(block_of_slot[p * n_z + tz])
            if src < 0:
                continue                     # plane never recorded: no photons, no coverage
            y0, x0 = pos[p, 1], pos[p, 2]
            acc[:, y0:y0 + ny, x0:x0 + nx] += tiles[src // n_z][src % n_z]
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


def move_legacy_outputs(raw_dir, base, out_dir, int_dir, tp_dir, verbose=True):
    """
    Move the Step-0 outputs that older versions of the time-lapse notebook wrote loose into the raw
    folder into the grouped layout (AnalyzeTCSPC_XYTZTime.ipynb):
        tiles_uint8/<base>_*                                  -> <out_dir>/tiles_uint8/
        <base>_tile_positions.json, <base>_block_map.json     -> <out_dir>/
        <base>[_T<tt>]_stitched_intensity.tif, <base>[_T<tt>]_coverage.tif,
        <base>_stitched_<CH>.tif / .json (Olympus channel)   -> <int_dir>/
        stitched/   (time-sum planes + the *_FLIM_Analysis folders inside it)  -> <out_dir>/stitched/
        stitched_T<tt>/                                       -> <tp_dir>/T<tt>/
    Renames within one drive keep the modification times, so every cache stays valid.  Only files
    named after `base` are touched, and nothing is overwritten: an item that already exists in the
    new layout is left where it was (listed in the return value).

    Returns dict: moved [(src, dst)], kept [(src, dst)] (destination already present).
    """
    def _move(src, dst, moved, kept):
        if os.path.exists(dst):
            kept.append((src, dst))
            return
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        try:
            os.replace(src, dst)
        except OSError as e:
            raise RuntimeError(f'Could not move {src} -> {dst} ({e}). Close the files in other programs '
                               f'(Fiji, napari, another kernel) and run again: the move resumes where it stopped.') from e
        moved.append((src, dst))

    moved, kept = [], []
    b = re.escape(base)
    if os.path.normcase(os.path.abspath(out_dir)) == os.path.normcase(os.path.abspath(raw_dir)):
        return dict(moved=moved, kept=kept)
    # tiles_uint8/<base>_*   (only this acquisition's files; the folder is removed once empty)
    old_tiles = os.path.join(raw_dir, 'tiles_uint8')
    if os.path.isdir(old_tiles):
        for f in sorted(os.listdir(old_tiles)):
            if f.startswith(base + '_'):
                _move(os.path.join(old_tiles, f), os.path.join(out_dir, 'tiles_uint8', f), moved, kept)
        if not os.listdir(old_tiles):
            os.rmdir(old_tiles)
    # loose files next to the raw file
    rx_top = re.compile(rf'^{b}_(tile_positions|block_map)\.json$')
    rx_int = re.compile(rf'^{b}(_T\d+)?_(stitched_intensity|coverage)\.tif$|^{b}_stitched_[^.\\/]+\.(tif|json)$')
    for f in sorted(os.listdir(raw_dir)):
        src = os.path.join(raw_dir, f)
        if not os.path.isfile(src):
            continue
        if rx_top.match(f):
            _move(src, os.path.join(out_dir, f), moved, kept)
        elif rx_int.match(f):
            _move(src, os.path.join(int_dir, f), moved, kept)
    # stitched planes: the time sum (with its analysis folders) and one folder per timepoint
    for d in sorted(os.listdir(raw_dir)):
        src = os.path.join(raw_dir, d)
        if not os.path.isdir(src):
            continue
        if d == 'stitched' and glob.glob(os.path.join(glob.escape(src), f'{glob.escape(base)}_stitched_Z*.tif')):
            _move(src, os.path.join(out_dir, 'stitched'), moved, kept)
        elif re.fullmatch(r'stitched_T\d+', d) and glob.glob(
                os.path.join(glob.escape(src), f'{glob.escape(base)}_{d[9:]}_stitched_Z*.tif')):
            _move(src, os.path.join(tp_dir, d[9:]), moved, kept)
    if verbose and (moved or kept):
        rel = lambda p: os.path.relpath(p, raw_dir)
        print(f'Moved {len(moved)} item(s) of the old layout (outputs loose in the raw folder) into '
              f'{rel(out_dir)}{os.sep}')
        for s, d in kept:
            print(f'  NOT moved (already in the new layout, delete the old copy by hand): {rel(s)}  [{rel(d)}]')
    return dict(moved=moved, kept=kept)


# ── a second intensity channel of the same tiles ──────────────────────────────
def channel_offset(ref_tiles, mov_tiles, upsample=10, smooth_sigma=1.0, top_frac=0.5):
    """
    Sub-pixel (dy, dx) that moves `mov_tiles` onto `ref_tiles`: the same tiles recorded by two
    detectors at once (e.g. an Olympus PMT and the TCSPC detector), so z is shared and only the
    pixel sampling may differ.  Phase correlation per tile and z plane on the `top_frac` planes with
    the most structure; the offset is the median over all of them.

    Returns dict: offset (dy, dx), spread (median absolute deviation over planes, px),
    per_tile (n_pos, 2) medians, ncc_before / ncc_after per tile.
    """
    from scipy.ndimage import shift as nd_shift
    n_pos, n_z = ref_tiles.shape[:2]
    sm = (0, smooth_sigma, smooth_sigma)
    refs = [gaussian_filter(t.astype(np.float32), sm) for t in ref_tiles]
    movs = [gaussian_filter(t.astype(np.float32), sm) for t in mov_tiles]
    shifts, per_tile = [], []
    for a, b in zip(refs, movs):
        score = a.reshape(n_z, -1).std(axis=1) * b.reshape(n_z, -1).std(axis=1)
        zs = np.argsort(score)[::-1][:max(1, int(round(top_frac * n_z)))]
        sh = np.array([phase_cross_correlation(a[z], b[z], upsample_factor=upsample,
                                               normalization=None)[0] for z in zs])
        shifts.append(sh)
        per_tile.append(np.median(sh, axis=0))
    allsh = np.concatenate(shifts)
    off = np.median(allsh, axis=0)
    has = [a.reshape(n_z, -1).max(axis=1) > 0 for a in refs]          # planes with data (not dropped)
    ncc_before = [_ncc(a[h], b[h]) for a, b, h in zip(refs, movs, has)]
    ncc_after = [_ncc(a[h], nd_shift(b, (0, off[0], off[1]), order=1, mode='nearest')[h])
                 for a, b, h in zip(refs, movs, has)]
    return dict(offset=off.tolist(), spread=np.median(np.abs(allsh - off), axis=0).tolist(),
                per_tile=np.array(per_tile).tolist(), ncc_before=ncc_before, ncc_after=ncc_after)


def fuse_intensity(tile_vols, reg, out_path=None, shift_yx=None, norm_pct=(1, 99.5)):
    """
    Stitch another intensity channel of the same tiles at the positions in `reg` (register_tiles
    result): each tile optionally shifted by `shift_yx` (sub-pixel, linear interpolation; see
    channel_offset), normalised to its own percentiles, then feather-blended like the segmentation
    intensity of fuse_tiles.  Returns the (z, y, x) float32 volume, also written to out_path if given.
    """
    from scipy.ndimage import shift as nd_shift
    n_pos, n_z, ny, nx = tile_vols.shape
    if [n_z, ny, nx] != list(reg['tile_shape']):
        raise ValueError(f'tiles are {(n_z, ny, nx)}, registration expects {tuple(reg["tile_shape"])}')
    Zc, Yc, Xc = reg['canvas_shape']
    pos = np.array(reg['positions'])
    feather = _feather(ny, nx)
    num = np.zeros((Zc, Yc, Xc), np.float32)
    den = np.zeros((Zc, Yc, Xc), np.float32)
    for p in range(n_pos):
        t = tile_vols[p].astype(np.float32)
        if shift_yx is not None and np.any(np.abs(shift_yx) > 1e-3):
            t = nd_shift(t, (0, shift_yx[0], shift_yx[1]), order=1, mode='nearest')
        lo, hi = np.percentile(t, norm_pct)
        z0, y0, x0 = pos[p]
        num[z0:z0 + n_z, y0:y0 + ny, x0:x0 + nx] += feather * (t - lo) / max(hi - lo, 1e-9)
        den[z0:z0 + n_z, y0:y0 + ny, x0:x0 + nx] += feather
    out = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    if out_path:
        tifffile.imwrite(out_path, out)
    return out


# ── dropped planes ────────────────────────────────────────────────────────────
def _block_features(v, bin_px=4, sigmas=(1.0, 8.0)):
    """Binned, band-passed (cells, not the depth profile), unit-norm rows for NCC by dot product."""
    n, ny, nx = v.shape
    v = v[:, :ny - ny % bin_px, :nx - nx % bin_px].astype(np.float32)
    v = v.reshape(n, ny // bin_px, bin_px, nx // bin_px, bin_px).mean(axis=(2, 4))
    v = gaussian_filter(v, (0, sigmas[0], sigmas[0])) - gaussian_filter(v, (0, sigmas[1], sigmas[1]))
    v = v.reshape(n, -1)
    v -= v.mean(axis=1, keepdims=True)
    nrm = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.where(nrm > 0, nrm, 1)


def align_blocks(block_int, ref_frames, min_ncc=0.3, mixed_ratio=0.8, gap_penalty=0.02):
    """
    True acquisition slot (tile * n_z + z) of every z-block of a Vista file, from reference frames of
    the same scan recorded by another detector (e.g. the Olympus PMT tiles, complete and indexed).

    Vista can drop planes: the file layout then puts every later block in a slot too early and leaves
    the last blocks empty.  Each non-empty block is compared with every reference frame (NCC of the
    band-passed images); the assignment is the best monotone path (blocks keep their order, each slot is
    used once, `gap_penalty` per skipped slot).  A block is discarded when its match is poor
    (NCC < min_ncc), or when it sits next to a gap in the path and matches worse than mixed_ratio x the
    median of its two neighbours on each side: Vista sums several frames per block, so a block at a gap
    can hold frames from both sides of it.

    block_int  : (n_blocks, ny, nx) photons per block in file order (sum over bins)
    ref_frames : (n_slots, ny, nx) reference frames in acquisition order (tile-major, z-minor)
    Returns dict: block_of_slot (n_slots,) source block or -1 (never recorded / discarded),
    slot_of_block (n_blocks,) assigned slot or -1, ncc (n_blocks,) at the assigned slot, ncc_layout
    (n_blocks,) at the file slot, missing (slots without data), empty (blocks without photons),
    discarded (blocks dropped), shifted (any block away from its file slot), ncc_matrix.
    """
    n_blocks, n_slots = block_int.shape[0], ref_frames.shape[0]
    S = _block_features(block_int) @ _block_features(ref_frames).T
    rec = np.flatnonzero(block_int.reshape(n_blocks, -1).max(axis=1) > 0)
    empty = np.setdiff1d(np.arange(n_blocks), rec)

    # best monotone path: score[i, s] = S[block i, slot s] + max_{s' < s} (score[i-1, s'] - gap * (s - s' - 1))
    idx = np.arange(n_slots)
    score = S[rec[0]] - gap_penalty * idx
    back = np.zeros((rec.size, n_slots), int)
    for i in range(1, rec.size):
        g = score + gap_penalty * idx                       # prev score + gap * s'
        best_prev = np.full(n_slots, -np.inf)
        arg = np.zeros(n_slots, int)
        run, run_arg = -np.inf, 0
        for s in range(1, n_slots):
            if g[s - 1] > run:
                run, run_arg = g[s - 1], s - 1
            best_prev[s], arg[s] = run, run_arg
        score = S[rec[i]] + best_prev - gap_penalty * (idx - 1)
        back[i] = arg
    path = np.zeros(rec.size, int)
    path[-1] = int(np.argmax(score - gap_penalty * (n_slots - 1 - idx)))     # trailing skips cost the same
    for i in range(rec.size - 1, 0, -1):
        path[i - 1] = back[i, path[i]]

    slot_of_block = np.full(n_blocks, -1)
    slot_of_block[rec] = path
    ncc = np.full(n_blocks, np.nan)
    ncc[rec] = S[rec, path]
    discarded = []
    for j, (b, s) in enumerate(zip(rec, path)):
        at_gap = (j > 0 and s - path[j - 1] > 1) or (j < rec.size - 1 and path[j + 1] - s > 1)
        nbr = [ncc[rec[i]] for i in range(max(0, j - 2), min(rec.size, j + 3)) if i != j]
        if S[b, s] < min_ncc or (at_gap and nbr and S[b, s] < mixed_ratio * np.median(nbr)):
            discarded.append(int(b))
            slot_of_block[b] = -1
    block_of_slot = np.full(n_slots, -1)
    ok = slot_of_block >= 0
    block_of_slot[slot_of_block[ok]] = np.flatnonzero(ok)
    n_lay = min(n_blocks, n_slots)
    return dict(block_of_slot=block_of_slot, slot_of_block=slot_of_block, ncc=ncc,
                ncc_layout=np.array([S[b, b] if b < n_slots else np.nan for b in range(n_blocks)]),
                missing=np.flatnonzero(block_of_slot < 0), empty=empty, discarded=np.array(discarded, int),
                shifted=bool(np.any(slot_of_block[rec] != rec)) or n_blocks != n_slots, ncc_matrix=S)
