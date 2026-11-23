"""AnalyzeTCSPC_XYTZTile.ipynb: tile_reg_source='auto' (Olympus PMT channel when its tiles exist, else the FLIM
intensity); Step 2 figure: room for the suptitle."""
import json

NB = r'X:\FLIM_2026\AnalyzeTCSPC_XYTZTile.ipynb'
raw = open(NB, encoding='utf-8', newline='').read()
indent = len(raw.splitlines()[1]) - len(raw.splitlines()[1].lstrip(' '))
nb = json.loads(raw)
cells = {c.get('id'): c for c in nb['cells']}
assert "_reg_src" not in ''.join(cells['code-tile0c']['source']), 'already patched'


def sub(cid, old, new):
    c = cells[cid]
    s = ''.join(c['source'])
    assert s.count(old) == 1, (cid, old[:60], s.count(old))
    c['source'] = s.replace(old, new).splitlines(keepends=True)
    if c['cell_type'] == 'code':
        c['outputs'], c['execution_count'] = [], None


sub('md-tile0', "`tile_reg_source='olympus'` registers on the complete PMT channel instead of the time-projected FLIM\n"
                "  intensity (same scan, so the positions apply to both).",
    "`tile_reg_source='auto'` registers on the Olympus PMT channel when its tiles exist (complete, no diffuse\n"
    "  background; same scan, so the positions apply to both), otherwise on the time-projected FLIM intensity.")
sub('code-tile0a', "tile_reg_source      = 'flim'       # register on 'flim' (time-projected TCSPC) or 'olympus' (PMT channel, Step 0b)",
    "tile_reg_source      = 'auto'       # 'auto' = Olympus PMT channel if its tiles exist, else 'flim'; or 'flim' / 'olympus'")
sub('code-tile0c', '''if tile_reg_source not in ('flim', 'olympus'):
    raise ValueError(f"tile_reg_source must be 'flim' or 'olympus', not {tile_reg_source!r}")
if tile_reg_source == 'olympus' and oir_tiles is None:
    raise RuntimeError("tile_reg_source='olympus' but this acquisition has no Olympus tiles (Step 0b)")''',
    '''if tile_reg_source not in ('auto', 'flim', 'olympus'):
    raise ValueError(f"tile_reg_source must be 'auto', 'flim' or 'olympus', not {tile_reg_source!r}")
_reg_src = ('olympus' if oir_tiles is not None else 'flim') if tile_reg_source == 'auto' else tile_reg_source
if _reg_src == 'olympus' and oir_tiles is None:
    raise RuntimeError("tile_reg_source='olympus' but this acquisition has no Olympus tiles (Step 0b)")
print(f'Registering on: {"Olympus " + oir_ch["name"] if _reg_src == "olympus" else "FLIM intensity"}')''')
sub('code-tile0c', "max_dz=max_dz, source=tile_reg_source, block_of_slot=_bos)",
    "max_dz=max_dz, source=_reg_src, block_of_slot=_bos)")
sub('code-tile0c', "tile_reg = vt.register_tiles(tile_int if tile_reg_source == 'flim' else oir_tiles.astype(np.float32),",
    "tile_reg = vt.register_tiles(tile_int if _reg_src == 'flim' else oir_tiles.astype(np.float32),")
sub('code-tile0c', "    if tile_reg_source == 'olympus':   # Step 0d normalises the segmentation intensity per FLIM tile",
    "    if _reg_src == 'olympus':   # Step 0d normalises the segmentation intensity per FLIM tile")
sub('a5874f7b', '''plt.suptitle(f'{FName}  —  Step 2: pre-processing & segmentation  (z={mid_idx}/{n_planes-1})', fontsize=9)
plt.tight_layout()''', '''plt.suptitle(f'{FName}  —  Step 2: pre-processing & segmentation  (z={mid_idx}/{n_planes-1})', fontsize=9)
plt.tight_layout(rect=(0, 0, 1, 0.98))''')

with open(NB, 'w', encoding='utf-8', newline='') as f:
    f.write(json.dumps(nb, indent=indent, ensure_ascii=False) + ('\n' if raw.endswith('\n') else ''))
print('patched md-tile0, code-tile0a, code-tile0c, a5874f7b')
