"""move_legacy_outputs (old flat layout -> analysis/) and fuse_tiles(int_dir=, absolute stitched_name)."""
import os, sys, time, tempfile
import numpy as np
import tifffile
sys.path.insert(0, r'X:\FLIM_2026')
import vista_tiles as vt

BASE = 'Slice1_X+0.50_Decay128_512x512_Z72_Ch1_T1_P1_S1'


def touch(p, data=b'x'):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, 'wb') as f:
        f.write(data)
    os.utime(p, (1_700_000_000, 1_700_000_000))   # fixed old mtime: a move must keep it


with tempfile.TemporaryDirectory() as td:
    raw = os.path.join(td, 'session')
    j = lambda *a: os.path.join(raw, *a)
    # old flat layout (as written by the notebook before the change)
    touch(j(f'{BASE}.tif'))                                         # raw data: stays
    touch(j(f'{BASE}.fbs.xml'))                                     # metadata: stays
    touch(j(f'{BASE}_Intensity.tif'))                               # not ours (no matching pattern): stays
    touch(j('Map_A01.oir'))                                         # Olympus: stays
    touch(j('tiles_uint8', f'{BASE}_t00_pos0.tif'))
    touch(j('tiles_uint8', f'{BASE}_tiles_intensity.tif'))
    touch(j('tiles_uint8', f'{BASE}_conversion_report.json'))
    touch(j(f'{BASE}_tile_positions.json'))
    touch(j(f'{BASE}_block_map.json'))
    touch(j(f'{BASE}_stitched_intensity.tif'))
    touch(j(f'{BASE}_coverage.tif'))
    touch(j(f'{BASE}_stitched_CH1.tif'))
    touch(j(f'{BASE}_stitched_CH1.json'))
    for t in range(3):
        touch(j(f'{BASE}_T{t:02d}_stitched_intensity.tif'))
        touch(j(f'{BASE}_T{t:02d}_coverage.tif'))
        touch(j(f'stitched_T{t:02d}', f'{BASE}_T{t:02d}_stitched_Z000.tif'))
    touch(j('stitched', f'{BASE}_stitched_Z000.tif'))
    touch(j('stitched', '20261118_1239_FLIM_Analysis', f'{BASE}_cell_acc.npz'))
    touch(j('stitched_other', 'foo.tif'))                           # not a stitched_T<tt> folder: stays

    out = j('analysis'); intd = os.path.join(out, 'intensity_coverage'); tpd = os.path.join(out, 'stitched_timepoints')
    # an item already in the new layout must not be overwritten (old copy stays, reported)
    touch(os.path.join(intd, f'{BASE}_T02_coverage.tif'), b'NEW')

    r = vt.move_legacy_outputs(raw, BASE, out, intd, tpd)
    left = sorted(os.listdir(raw))
    assert left == sorted(['analysis', f'{BASE}.tif', f'{BASE}.fbs.xml', f'{BASE}_Intensity.tif', 'Map_A01.oir',
                           'stitched_other', f'{BASE}_T02_coverage.tif']), left
    exp = [os.path.join(out, 'tiles_uint8', f'{BASE}_t00_pos0.tif'),
           os.path.join(out, 'tiles_uint8', f'{BASE}_conversion_report.json'),
           os.path.join(out, f'{BASE}_tile_positions.json'), os.path.join(out, f'{BASE}_block_map.json'),
           os.path.join(intd, f'{BASE}_stitched_intensity.tif'), os.path.join(intd, f'{BASE}_coverage.tif'),
           os.path.join(intd, f'{BASE}_stitched_CH1.tif'), os.path.join(intd, f'{BASE}_stitched_CH1.json'),
           os.path.join(intd, f'{BASE}_T01_stitched_intensity.tif'), os.path.join(intd, f'{BASE}_T00_coverage.tif'),
           os.path.join(out, 'stitched', f'{BASE}_stitched_Z000.tif'),
           os.path.join(out, 'stitched', '20261118_1239_FLIM_Analysis', f'{BASE}_cell_acc.npz'),
           os.path.join(tpd, 'T00', f'{BASE}_T00_stitched_Z000.tif'),
           os.path.join(tpd, 'T02', f'{BASE}_T02_stitched_Z000.tif')]
    for p in exp:
        assert os.path.isfile(p), p
        assert os.path.getmtime(p) == 1_700_000_000, ('mtime changed', p)
    assert open(os.path.join(intd, f'{BASE}_T02_coverage.tif'), 'rb').read() == b'NEW'   # not overwritten
    assert len(r['kept']) == 1 and not os.path.exists(j('tiles_uint8'))
    r2 = vt.move_legacy_outputs(raw, BASE, out, intd, tpd, verbose=False)               # idempotent
    assert not r2['moved'] and len(r2['kept']) == 1
    print(f'move_legacy_outputs OK  ({len(r["moved"])} moved, {len(r["kept"])} kept)')

    # fuse_tiles: intensity/coverage into int_dir, planes into an absolute stitched folder; default unchanged
    rng = np.random.default_rng(0)
    nz, nbin, ny, nx = 3, 4, 16, 20
    tile = rng.integers(0, 5, (nz, nbin, ny, nx)).astype(np.uint8)
    tp = os.path.join(td, 'tile.tif')
    tifffile.imwrite(tp, tile, imagej=True, metadata={'axes': 'ZCYX'})
    reg = dict(positions=[[0, 0, 0]], canvas_shape=[nz, ny, nx], tile_shape=[nz, ny, nx], norm_lo=[0.0], norm_hi=[10.0])
    ti = tile.sum(axis=1, dtype=np.float32)[np.newaxis]
    st = vt.fuse_tiles([tp], reg, ti, out, f'{BASE}_T00', stitched_name=os.path.join(tpd, 'T05'), int_dir=intd,
                       verbose=False)
    assert st['stitched_dir'] == os.path.join(tpd, 'T05') and len(os.listdir(st['stitched_dir'])) == nz
    assert os.path.dirname(st['int_path']) == intd and os.path.dirname(st['cov_path']) == intd
    assert st['photons_fused'] == st['photons_tiles']
    st2 = vt.fuse_tiles([tp], reg, ti, out, f'{BASE}_T00', stitched_name=os.path.join(tpd, 'T05'), int_dir=intd,
                        verbose=False)
    assert st2['photons_fused'] is None                                                 # reused
    d0 = os.path.join(td, 'legacy')
    st3 = vt.fuse_tiles([tp], reg, ti, d0, 'b', verbose=False)                          # Tile notebook call
    assert st3['int_path'] == os.path.join(d0, 'b_stitched_intensity.tif') and st3['stitched_dir'] == os.path.join(d0, 'stitched')
    print('fuse_tiles int_dir / absolute stitched_name / default layout OK')
print('ALL OK')
