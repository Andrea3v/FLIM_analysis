"""AnalyzeTCSPC_XYTZTile.ipynb: re-assign Vista z-blocks to their true tile / z when planes were dropped.
Step 0b matches every block to the Olympus tiles (vista_tiles.align_blocks), remaps tile_int and warns; Step 0c
re-registers when the block assignment changed; Step 0d stitches through the map; Step 0e reuses the Olympus tiles."""
import json

NB = r'X:\FLIM_2026\AnalyzeTCSPC_XYTZTile.ipynb'
raw = open(NB, encoding='utf-8', newline='').read()
indent = len(raw.splitlines()[1]) - len(raw.splitlines()[1].lstrip(' '))
nb = json.loads(raw)
cells = {c.get('id'): c for c in nb['cells']}
assert 'align_blocks' not in ''.join(cells['code-tile0b']['source']), 'already patched'


def src(cid):
    return ''.join(cells[cid]['source'])


def put(cid, text):
    c = cells[cid]
    c['source'] = text.splitlines(keepends=True)
    if c['cell_type'] == 'code':
        c['outputs'], c['execution_count'] = [], None


def sub(cid, old, new):
    s = src(cid)
    assert s.count(old) == 1, (cid, old[:60], s.count(old))
    put(cid, s.replace(old, new))


sub('md-tile0', '''- **0b** float32 → uint8: one ImageJ hyperstack per tile (`ZCYX`, decay bins as channels) in `tiles_uint8/`.
  Vista writes classic TIFFs past 4 GB: pages after ~4 GB have no IFD, and their IFDs are written over plane data
  in the first 4 GB. Those pixels are located exactly and zeroed (reported as *corrupted bins*).''',
    '''- **0b** float32 → uint8: one ImageJ hyperstack per tile (`ZCYX`, decay bins as channels) in `tiles_uint8/`, in the
  file's block order. Vista writes classic TIFFs past 4 GB: pages after ~4 GB have no IFD, and their IFDs are written
  over plane data in the first 4 GB. Those pixels are located exactly and zeroed (reported as *corrupted bins*).
  **Dropped planes:** Vista can miss planes the microscope acquired; every later block then sits in a slot too early
  and the last blocks of the file stay empty. With Olympus tiles (same scan, complete), every block is matched to the
  Olympus frame it shows and moved to its true tile / z (`<base>_block_map.json`); planes never recorded stay empty
  (no photons, coverage 0) and a block straddling a gap is discarded. A warning lists the missing planes. Without
  Olympus tiles, empty blocks only trigger a warning (the stitch is then unreliable).''')

put('code-tile0b', '''# ── Step 0b: float32 Vista TIFF → uint8 tiles ─────────────────────────────────
conv = vt.convert_to_uint8(raw_path, tiles_dir, n_pos=n_tiles, xy_pixel_um=xy_pixel_um,
                           z_step_um=z_step_um, overwrite=rerun_convert)
tile_int = conv['tile_int']          # (tile, z, y, x) photons summed over bins, file block order
_rep = conv['report']
print(f'Corrupted bins zeroed: {100 * _rep["corrupted_fraction"]:.3f}%  |  '
      f'residual junk: {_rep["residual_junk_pixels"]}  |  clipped >255: {_rep["clipped_pixels"]}  |  '
      f'max count/bin/px: {_rep["max_count"]:g}')

# ── dropped planes: true tile / z of every block, from the Olympus tiles of the same scan ──
_ny, _nx  = tile_int.shape[2:]
_empty    = np.flatnonzero(tile_int.reshape(n_tiles * n_z_tile, -1).max(axis=1) == 0)
blk, oir_tiles = None, None
block_of_slot  = None   # None = file layout; else the file block shown at slot tile*n_z + z (-1 = no data)
if (matl_info and oir_ch and all(matl_info['files'])
        and (oir_meta['n_z'], oir_meta['ny'], oir_meta['nx']) == tuple(tile_int.shape[1:])):
    oir_tiles = np.stack([oo.read_oir_channel(f, oir_ch['id'], oir_meta['n_z'], oir_meta['ny'], oir_meta['nx'])
                          for f in matl_info['files']])
    blk = vt.align_blocks(tile_int.reshape(-1, _ny, _nx), oir_tiles.reshape(-1, _ny, _nx))
    with open(os.path.join(raw_dir, f'{vista_info["base"]}_block_map.json'), 'w') as _f:
        json.dump({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in blk.items() if k != 'ncc_matrix'}, _f)
    _kept = blk['slot_of_block'] >= 0
    print(f'Block check vs Olympus {oir_ch["name"]}: {int(_kept.sum())} of {_kept.size} blocks matched  '
          f'(NCC median {np.nanmedian(blk["ncc"][_kept]):.2f}, min {np.nanmin(blk["ncc"][_kept]):.2f})')
    if blk['shifted'] or blk['missing'].size:
        block_of_slot = blk['block_of_slot']
        _flat = tile_int.reshape(-1, _ny, _nx)
        _true = np.zeros_like(_flat)
        _true[block_of_slot >= 0] = _flat[block_of_slot[block_of_slot >= 0]]
        tile_int = _true.reshape(tile_int.shape)
        _miss = '; '.join(f'tile {t}: z {[int(s % n_z_tile) for s in blk["missing"] if s // n_z_tile == t]}'
                          for t in range(n_tiles) if any(s // n_z_tile == t for s in blk['missing']))
        warnings.warn(f'Vista did not record {blk["missing"].size} of {block_of_slot.size} planes '
                      f'({blk["empty"].size} empty blocks at the end of the file'
                      + (f', {blk["discarded"].size} block(s) straddling a gap discarded: {blk["discarded"].tolist()}'
                         if blk['discarded'].size else '')
                      + '). The recorded blocks were moved to their true tile / z (matched to the Olympus channel); '
                      f'the missing planes stay empty (no photons, coverage 0) -> {_miss}', RuntimeWarning)
    else:
        print('  every block is at its file position: no dropped planes')
elif _empty.size:
    _t0, _z0 = divmod(int(_empty[0]), n_z_tile)
    warnings.warn(f'{_empty.size} of {n_tiles * n_z_tile} empty z-blocks (first: tile {_t0} z {_z0}): Vista recorded '
                  f'fewer planes than the microscope acquired. Blocks after the first dropped plane sit at the wrong '
                  f'tile / z and there are no Olympus tiles to re-assign them: registration and stitching are '
                  f'unreliable.', RuntimeWarning)

_ph = tile_int.sum(axis=(2, 3))
fig, axs = plt.subplots(2, n_tiles, figsize=(3.2 * n_tiles, 5.2),
                        gridspec_kw=dict(height_ratios=[3, 1.2]))
for p in range(n_tiles):
    _mp = tile_int[p].max(axis=0)
    axs[0, p].imshow(_mp, cmap='gray', vmax=np.percentile(_mp, 99.5))
    axs[0, p].set_title(f'tile {p}  (max over z)', fontsize=9)
    axs[0, p].axis('off')
    axs[1, p].plot(_ph[p] / 1e3, '.-', lw=0.8)
    if block_of_slot is not None:
        _mz = np.flatnonzero(block_of_slot.reshape(n_tiles, n_z_tile)[p] < 0)
        axs[1, p].plot(_mz, np.zeros(_mz.size), 'rx', ms=5, label='not recorded')
    axs[1, p].set_xlabel('z')
axs[1, 0].set_ylabel('photons / 1e3')
if block_of_slot is not None:
    axs[1, 0].legend(fontsize=7)
plt.suptitle(f'{vista_info["base"]}  —  tiles in acquisition order'
             + ('  (blocks moved to their true tile / z)' if block_of_slot is not None else ''), fontsize=9)
plt.tight_layout()
plt.show()

if blk is not None:   # QC: which Olympus frame each Vista block shows
    _S = blk['ncc_matrix']
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(_S, aspect='auto', cmap='magma', vmin=0, vmax=0.9, interpolation='nearest')
    ax.plot([0, _S.shape[0] - 1], [0, _S.shape[0] - 1], 'w--', lw=0.6, label='file layout')
    _b = np.flatnonzero(blk['slot_of_block'] >= 0)
    ax.plot(blk['slot_of_block'][_b], _b, 'c.', ms=2, label='assigned tile / z')
    if blk['discarded'].size:
        ax.plot(_S[blk['discarded']].argmax(axis=1), blk['discarded'], 'rx', ms=7, label='discarded (straddles a gap)')
    for t in range(1, n_tiles):
        ax.axvline(t * n_z_tile - 0.5, color='w', lw=0.3)
    ax.set_xlabel(f'Olympus {oir_ch["name"]} frame  (tile x {n_z_tile} + z)')
    ax.set_ylabel('Vista z-block (file order)')
    ax.set_title('NCC: Vista block vs Olympus frame', fontsize=10)
    ax.legend(loc='lower right', fontsize=8)
    plt.tight_layout()
    plt.show()
''')

sub('code-tile0c', '''if os.path.exists(pos_json) and not rerun_register:
    tile_reg = vt.load_tile_positions(pos_json)
    print(f'Reloaded tile positions <- {os.path.basename(pos_json)}')
else:
    tile_reg = vt.register_tiles(tile_int, tile_grid, overlap=tile_overlap, layout=_reg_layout,
                                 fallback=tile_layout_fallback, max_dz=max_dz)
    vt.save_tile_positions(pos_json, tile_reg, raw_path=raw_path)''',
    '''_bos     = None if block_of_slot is None else [int(b) for b in block_of_slot]
_old_reg = vt.load_tile_positions(pos_json) if os.path.exists(pos_json) else None
if _old_reg is not None and not rerun_register and _old_reg.get('block_of_slot') == _bos:
    tile_reg = _old_reg
    print(f'Reloaded tile positions <- {os.path.basename(pos_json)}')
else:
    if _old_reg is not None and not rerun_register:
        print('Saved tile positions were made with another block assignment (Step 0b) -> registering again')
    tile_reg = vt.register_tiles(tile_int, tile_grid, overlap=tile_overlap, layout=_reg_layout,
                                 fallback=tile_layout_fallback, max_dz=max_dz)
    vt.save_tile_positions(pos_json, tile_reg, raw_path=raw_path, block_of_slot=_bos)''')

sub('code-tile0d', '''stitch = vt.fuse_tiles(conv['tile_paths'], tile_reg, tile_int, raw_dir, vista_info['base'],
                       overwrite=rerun_fuse)''',
    '''stitch = vt.fuse_tiles(conv['tile_paths'], tile_reg, tile_int, raw_dir, vista_info['base'],
                       overwrite=rerun_fuse, block_of_slot=block_of_slot)''')

sub('code-tile0e', '''        _oir_tiles = np.stack([oo.read_oir_channel(f, oir_ch['id'], oir_meta['n_z'], oir_meta['ny'], oir_meta['nx'])
                               for f in matl_info['files']])''',
    '''        _oir_tiles = oir_tiles if oir_tiles is not None else np.stack(
            [oo.read_oir_channel(f, oir_ch['id'], oir_meta['n_z'], oir_meta['ny'], oir_meta['nx'])
             for f in matl_info['files']])''')

with open(NB, 'w', encoding='utf-8', newline='') as f:
    f.write(json.dumps(nb, indent=indent, ensure_ascii=False) + ('\n' if raw.endswith('\n') else ''))
print('patched code-tile0b/0c/0d/0e + md-tile0')
