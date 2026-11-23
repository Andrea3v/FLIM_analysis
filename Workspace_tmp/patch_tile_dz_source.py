"""AnalyzeTCSPC_XYTZTile.ipynb: wider tile z search (max_dz 12), tile registration source ('flim' / 'olympus'),
Step 0c re-registers when any registration setting changes, Step 2 caches must be newer than the stitched input."""
import json

NB = r'X:\FLIM_2026\AnalyzeTCSPC_XYTZTile.ipynb'
raw = open(NB, encoding='utf-8', newline='').read()
indent = len(raw.splitlines()[1]) - len(raw.splitlines()[1].lstrip(' '))
nb = json.loads(raw)
cells = {c.get('id'): c for c in nb['cells']}
assert 'tile_reg_source' not in ''.join(cells['code-tile0a']['source']), 'already patched'


def sub(cid, old, new):
    c = cells[cid]
    s = ''.join(c['source'])
    assert s.count(old) == 1, (cid, old[:60], s.count(old))
    c['source'] = s.replace(old, new).splitlines(keepends=True)
    if c['cell_type'] == 'code':
        c['outputs'], c['execution_count'] = [], None


sub('md-tile0', '''  `tile_layout='auto'` scores raster/snake × start corner (and the transposed grid) and `tile_layout_fallback` is
  used when nothing matches or it ties with the best.''',
    '''  `tile_layout='auto'` scores raster/snake × start corner (and the transposed grid) and `tile_layout_fallback` is
  used when nothing matches or it ties with the best. `max_dz` bounds the z offset between neighbours: with a tilted
  objective, neighbouring tiles are several planes apart (2026-11-06: 3–10 planes); a pair at the bound gives a
  warning. `tile_reg_source='olympus'` registers on the complete PMT channel instead of the time-projected FLIM
  intensity (same scan, so the positions apply to both). Positions are recomputed when a registration setting or
  the block assignment of 0b changes.''')

sub('code-tile0a', '''max_dz               = 2            # max z offset between neighbouring tiles (slices)
''', '''max_dz               = 12           # max z offset between neighbouring tiles (planes); a tilted objective
                                    #  puts neighbours several planes apart (2026-11-06: 3-10 planes)
tile_reg_source      = 'flim'       # register on 'flim' (time-projected TCSPC) or 'olympus' (PMT channel, Step 0b)
''')

sub('code-tile0c', '''_bos     = None if block_of_slot is None else [int(b) for b in block_of_slot]
_old_reg = vt.load_tile_positions(pos_json) if os.path.exists(pos_json) else None
if _old_reg is not None and not rerun_register and _old_reg.get('block_of_slot') == _bos:
    tile_reg = _old_reg
    print(f'Reloaded tile positions <- {os.path.basename(pos_json)}')
else:
    if _old_reg is not None and not rerun_register:
        print('Saved tile positions were made with another block assignment (Step 0b) -> registering again')
    tile_reg = vt.register_tiles(tile_int, tile_grid, overlap=tile_overlap, layout=_reg_layout,
                                 fallback=tile_layout_fallback, max_dz=max_dz)
    vt.save_tile_positions(pos_json, tile_reg, raw_path=raw_path, block_of_slot=_bos)''',
    '''if tile_reg_source not in ('flim', 'olympus'):
    raise ValueError(f"tile_reg_source must be 'flim' or 'olympus', not {tile_reg_source!r}")
if tile_reg_source == 'olympus' and oir_tiles is None:
    raise RuntimeError("tile_reg_source='olympus' but this acquisition has no Olympus tiles (Step 0b)")
_bos     = None if block_of_slot is None else [int(b) for b in block_of_slot]
_reg_key = dict(grid=list(tile_grid), overlap=tile_overlap, layout=_reg_layout, fallback=tile_layout_fallback,
                max_dz=max_dz, source=tile_reg_source, block_of_slot=_bos)     # any change -> register again
_old_reg = vt.load_tile_positions(pos_json) if os.path.exists(pos_json) else None
if _old_reg is not None and not rerun_register and _old_reg.get('reg_key') == _reg_key:
    tile_reg = _old_reg
    print(f'Reloaded tile positions <- {os.path.basename(pos_json)}')
else:
    if _old_reg is not None and not rerun_register:
        _diff = [k for k in _reg_key if (_old_reg.get('reg_key') or {}).get(k) != _reg_key[k]]
        print(f'Saved tile positions were made with other settings ({", ".join(_diff)}) -> registering again')
    tile_reg = vt.register_tiles(tile_int if tile_reg_source == 'flim' else oir_tiles.astype(np.float32),
                                 tile_grid, overlap=tile_overlap, layout=_reg_layout,
                                 fallback=tile_layout_fallback, max_dz=max_dz)
    if tile_reg_source == 'olympus':   # Step 0d normalises the segmentation intensity per FLIM tile
        tile_reg['norm_lo'] = [float(np.percentile(t, 1)) for t in tile_int]
        tile_reg['norm_hi'] = [float(np.percentile(t, 99.5)) for t in tile_int]
    vt.save_tile_positions(pos_json, tile_reg, raw_path=raw_path, reg_key=_reg_key)''')

sub('a5874f7b', '''if os.path.exists(_filt_path) and os.path.exists(_mask_path):
    vol_norm_3d = tifffile.imread(_filt_path).astype(np.float32)''',
    '''_newer = lambda p, *srcs: os.path.exists(p) and all(os.path.getmtime(p) > os.path.getmtime(s) for s in srcs)
if _newer(_filt_path, stitch['int_path']) and _newer(_mask_path, stitch['int_path']):
    vol_norm_3d = tifffile.imread(_filt_path).astype(np.float32)''')
sub('a5874f7b', '''    if os.path.exists(_cp_path) and _old == _cp_settings and not _rerun_cp:''',
    '''    _seg_src = stitch_oir if seg_source == 'olympus' else stitch['int_path']
    if _newer(_cp_path, _seg_src) and _old == _cp_settings and not _rerun_cp:''')

with open(NB, 'w', encoding='utf-8', newline='') as f:
    f.write(json.dumps(nb, indent=indent, ensure_ascii=False) + ('\n' if raw.endswith('\n') else ''))
print('patched md-tile0, code-tile0a, code-tile0c, a5874f7b')
