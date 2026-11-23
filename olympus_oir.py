"""
olympus_oir.py
Olympus FluoView (FV31S, FVMPE-RS) multi-area outputs for AnalyzeTCSPC_XYTZTile.ipynb (and the
time-series AnalyzeTCSPC_XYTZTime.ipynb).

    <session>/<date>_..._Olympus/<name>_Cycle/
        matl.omp2info                     multi-area (MATL) protocol: grid, overlap, area order
                                          (also found as matl.omp2info.xml)
        <name>_A01_G001_0001.oir ...      one .oir per area (tile): all channels, all z

OIR files (reverse-engineered on FV31S 2.4 files): 'OLYMPUSRAWFORMAT' header, then blocks.  Pixel data
are stored in strips of a few lines, one block per strip:
    <u32 byte offset in the image> <u32 n bytes> <u32 len(uid)> <uid> <u32 n bytes> <u32 4> <pixels, <u2>
    uid = 'z<ZZZ>_0_<t>_<channel uuid>_<strip>'   ('REF_LSM0_<uuid>_<strip>' = reference image, ignored)
Metadata are XML blocks (<u32 length> before '<?xml'); the last 'lsmimage:imageProperties' block holds
channels (name, emission filter), pixel size and the z axis.

Functions
---------
find_matl_dir     -- folder holding matl.omp2info below a session folder
read_matl_info    -- grid (rows, cols), overlap, tile files in acquisition order, layout name
oir_metadata      -- channels, pixel size, z step, n_z, image size of one .oir
read_oir_channel  -- (n_z, ny, nx) uint16 of one channel
pick_channel      -- channel by name, or 'auto' = longest-wavelength emission filter (else brightest)
"""

import os
import re
import glob
import struct
import xml.etree.ElementTree as ET
import numpy as np

_UUID = rb'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'


def _local(tag):
    return tag.rsplit('}', 1)[-1]


def _children(el, name):
    return [c for c in el if _local(c.tag) == name]


def _child_text(el, name, cast=str, default=None):
    """Text of the first descendant (not `el` itself) named `name`."""
    for c in el.iter():
        if c is not el and _local(c.tag) == name and c.text is not None:
            try:
                return cast(c.text.strip())
            except ValueError:
                return default
    return default


# ── multi-area layout (matl.omp2info) ─────────────────────────────────────────
MATL_NAMES = ('matl.omp2info', 'matl.omp2info.xml')   # FV31S writes the first; copied/exported folders
                                                       #  can carry the .xml extension (2026-11-18 slices)


def _matl_file(matl_dir):
    """The MATL protocol file in `matl_dir` (either name of MATL_NAMES), or None."""
    return next((p for p in (os.path.join(matl_dir, n) for n in MATL_NAMES) if os.path.isfile(p)), None)


def find_matl_dir(root, max_depth=3):
    """Folder holding matl.omp2info (or matl.omp2info.xml) at most `max_depth` levels below `root`
    (None if absent).  Several hits: the one whose folder name shares the longest prefix with `root`'s name."""
    hits = []
    for d in range(max_depth + 1):
        for name in MATL_NAMES:
            hits += glob.glob(os.path.join(glob.escape(root), *(['*'] * d), name))
    if not hits:
        return None
    name = os.path.basename(os.path.normpath(root)).replace('_', '').lower()
    key = lambda p: len(os.path.commonprefix([name, os.path.basename(os.path.dirname(p)).replace('_', '').lower()]))
    return os.path.dirname(max(hits, key=key))


def read_matl_info(matl_dir):
    """
    Multi-area protocol of an Olympus MATL acquisition.

    Returns dict: grid (n_rows, n_cols) = (max yIndex + 1, max xIndex + 1), overlap (fraction),
    files (tile .oir paths in acquisition order, None where missing), grid_rc ((row, col) per tile),
    layout (one of vista_tiles.LAYOUTS matching the order, or None), path.
    """
    from vista_tiles import LAYOUTS, tile_grid_positions
    path = _matl_file(matl_dir)
    if path is None:
        raise FileNotFoundError(f'no {" / ".join(MATL_NAMES)} in {matl_dir}')
    root = ET.parse(path).getroot()
    overlap = None
    for el in root.iter():
        if _local(el.tag) == 'overlap' and el.text:
            overlap = float(el.text) / 100.0
    areas = []
    for el in root.iter():
        if _local(el.tag) == 'area':
            img = _child_text(el, 'image')
            x, y = _child_text(el, 'xIndex', int), _child_text(el, 'yIndex', int)
            if img is not None and x is not None and y is not None:
                areas.append((img, y, x))
    if not areas:
        raise ValueError(f'{path}: no <matl:area> entries')
    # acquisition order = running number at the end of the file name (..._0001.oir)
    num = lambda a: int(m[1]) if (m := re.search(r'_(\d+)\.oir$', a[0])) else 0
    areas.sort(key=num)
    rc = np.array([(r, c) for _, r, c in areas])
    grid = (int(rc[:, 0].max()) + 1, int(rc[:, 1].max()) + 1)
    layout = None
    if len(areas) == grid[0] * grid[1]:
        layout = next((lay for lay in LAYOUTS if np.array_equal(tile_grid_positions(lay, grid), rc)), None)
    files = [os.path.join(matl_dir, a[0]) for a in areas]
    files = [f if os.path.exists(f) else None for f in files]
    return dict(path=path, grid=grid, overlap=overlap, files=files, grid_rc=rc.tolist(), layout=layout)


# ── OIR files ─────────────────────────────────────────────────────────────────
def _xml_blocks(b):
    for m in re.finditer(rb'<\?xml', b):
        s = m.start()
        n = struct.unpack('<I', b[s - 4:s])[0]
        if 0 < n <= len(b) - s:
            yield b[s:s + n]


def oir_metadata(path):
    """
    Acquisition metadata of one .oir file (last imageProperties block).

    Returns dict: channels [dict(id, name, order, filter, emission_nm (lo, hi) or None, voltage)],
    pixel_um (y, x), z_step_um, n_z, z_start, z_end, n_t (TIMELAPSE axis, 1 without), ny, nx, bit_depth.
    """
    with open(path, 'rb') as fh:
        b = fh.read()
    props = [x for x in _xml_blocks(b) if b'imageProperties' in x[:400]]
    if not props:
        raise ValueError(f'{path}: no imageProperties XML block')
    root = ET.fromstring(props[-1].decode('utf-8', 'replace'))
    info = next((e for e in root.iter() if _local(e.tag) == 'imageInfo'), None)
    if info is None:
        raise ValueError(f'{path}: no imageInfo')

    volts = {}
    for e in root.iter():
        if _local(e.tag) == 'pmt' and e.get('channelId'):
            volts[e.get('channelId')] = _child_text(e, 'voltage', float)
    channels, px, bits = [], None, None
    for e in info.iter():
        if _local(e.tag) != 'channel' or not e.get('id'):
            continue
        filt = _child_text(e, 'filterName')
        m = re.search(r'(\d{3})\D+(\d{3})', filt or '')
        channels.append(dict(id=e.get('id'), name=_child_text(e, 'name'), order=int(e.get('order', 0)),
                             filter=filt, emission_nm=(int(m[1]), int(m[2])) if m else None,
                             voltage=volts.get(e.get('id'))))
        for ln in _children(e, 'length'):
            px = (_child_text(ln, 'y', float), _child_text(ln, 'x', float))
        bits = bits or _child_text(e, 'bitCounts', int)
    channels.sort(key=lambda c: c['order'])

    z_step, n_z, z0, z1, n_t = None, 1, None, None, 1
    for ax in _children(info, 'axis'):
        if _child_text(ax, 'axis') == 'ZSTACK':
            z_step = _child_text(ax, 'step', float)
            n_z = _child_text(ax, 'maxSize', int, 1)
            z0, z1 = _child_text(ax, 'startPosition', float), _child_text(ax, 'endPosition', float)
        elif _child_text(ax, 'axis') == 'TIMELAPSE':   # not yet seen in a file: checked against the Vista block count
            n_t = _child_text(ax, 'maxSize', int, 1)
    return dict(channels=channels, pixel_um=px, z_step_um=z_step, n_z=n_z, z_start=z0, z_end=z1, n_t=n_t,
                ny=_child_text(info, 'height', int), nx=_child_text(info, 'width', int), bit_depth=bits)


def read_oir_channel(path, channel_id, n_z=None, ny=None, nx=None):
    """(n_z, ny, nx) uint16 image of one channel (uuid).  Dimensions default to oir_metadata(path)."""
    if None in (n_z, ny, nx):
        meta = oir_metadata(path)
        n_z, ny, nx = n_z or meta['n_z'], ny or meta['ny'], nx or meta['nx']
    with open(path, 'rb') as fh:
        b = fh.read()
    pat = re.compile(rb'([\x10-\x7f])\x00\x00\x00(z(\d+)_\d+_\d+_' + re.escape(channel_id.encode()) + rb'_\d+)')
    plane_bytes = ny * nx * 2
    buf = np.zeros(n_z * plane_bytes, np.uint8)
    filled = np.zeros(n_z, np.int64)
    for m in pat.finditer(b):
        if m.group(1)[0] != len(m.group(2)):
            continue
        s, e = m.start(), m.end()
        off, cnt = struct.unpack('<II', b[s - 8:s])
        cnt2, flag = struct.unpack('<II', b[e:e + 8])
        z = int(m.group(3)) - 1
        if flag != 4 or cnt != cnt2 or not 0 <= z < n_z or off + cnt > plane_bytes:
            continue          # index entries at the end of the file reuse the uid strings
        buf[z * plane_bytes + off:z * plane_bytes + off + cnt] = np.frombuffer(b, np.uint8, cnt, e + 8)
        filled[z] += cnt
    if np.any(filled != plane_bytes):
        bad = np.flatnonzero(filled != plane_bytes)
        raise ValueError(f'{os.path.basename(path)}: incomplete planes for channel {channel_id[:8]} '
                         f'(z = {bad.tolist()[:10]}, bytes {filled[bad].tolist()[:10]} of {plane_bytes})')
    return buf.view('<u2').reshape(n_z, ny, nx)


def pick_channel(meta, channel='auto', path=None):
    """Channel dict by name ('CH1'), uuid, or 'auto' = the longest-wavelength emission filter (the red
    PMT behind the FV30-FGR cube); without filter names, the brightest channel of `path`."""
    chans = meta['channels']
    if channel != 'auto':
        hit = [c for c in chans if channel in (c['name'], c['id'])]
        if not hit:
            raise ValueError(f'channel {channel!r} not in {[c["name"] for c in chans]}')
        return hit[0]
    with_em = [c for c in chans if c['emission_nm']]
    if with_em:
        return max(with_em, key=lambda c: c['emission_nm'][0])
    if path is None or not chans:
        return None
    means = [read_oir_channel(path, c['id'], meta['n_z'], meta['ny'], meta['nx']).mean() for c in chans]
    return chans[int(np.argmax(means))]
