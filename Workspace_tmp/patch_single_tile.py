"""AnalyzeTCSPC_XYTZTile.ipynb Step 0: a single position (tile_grid = (1, 1)) runs through Steps 0a-0d."""
import json

NB = r'X:\FLIM_2026\AnalyzeTCSPC_XYTZTile.ipynb'
raw = open(NB, encoding='utf-8', newline='').read()
indent = len(raw.splitlines()[1]) - len(raw.splitlines()[1].lstrip(' '))
nb = json.loads(raw)
cells = {c.get('id'): c for c in nb['cells']}
assert 'squeeze=False' not in ''.join(cells['code-tile0b']['source']), 'already patched'


def sub(cid, old, new):
    c = cells[cid]
    s = ''.join(c['source'])
    assert s.count(old) == 1, (cid, old[:60], s.count(old))
    c['source'] = s.replace(old, new).splitlines(keepends=True)
    if c['cell_type'] == 'code':
        c['outputs'], c['execution_count'] = [], None


# 0a: say how to declare a single position when the grid does not divide the blocks
sub('code-tile0a', '''tile_grid    = (3, 2)    # (rows, cols)                <- matl.omp2info''',
    '''tile_grid    = (3, 2)    # (rows, cols)                <- matl.omp2info; (1, 1) = single position''')
sub('code-tile0a', '''    raise ValueError(f'_Z{vista_info["n_blocks"]} is not a multiple of {n_tiles} tiles')''',
    '''    raise ValueError(f'_Z{vista_info["n_blocks"]} is not a multiple of {n_tiles} tiles ({tile_grid[0]}x{tile_grid[1]} '
                     f'grid{"" if matl_info else ", default"}): set tile_grid = (1, 1) for a single position')''')
# 0b: plt.subplots(2, 1) returns a 1-D axes array
sub('code-tile0b', '''fig, axs = plt.subplots(2, n_tiles, figsize=(3.2 * n_tiles, 5.2),
                        gridspec_kw=dict(height_ratios=[3, 1.2]))''',
    '''fig, axs = plt.subplots(2, n_tiles, figsize=(max(3.2 * n_tiles, 4.5), 5.2), squeeze=False,
                        gridspec_kw=dict(height_ratios=[3, 1.2]))''')
# 0c: no pairs to tabulate
sub('code-tile0c', '''print(_df_pairs.to_string(index=False))''',
    '''print(_df_pairs.to_string(index=False) if len(_df_pairs) else 'Single position: no tile pairs, nothing to register')''')

with open(NB, 'w', encoding='utf-8', newline='') as f:
    f.write(json.dumps(nb, indent=indent, ensure_ascii=False) + ('\n' if raw.endswith('\n') else ''))
print('patched code-tile0a, code-tile0b, code-tile0c')
