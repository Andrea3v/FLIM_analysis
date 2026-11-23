"""Step 0b of AnalyzeTCSPC_XYTZTile.ipynb: warn when z-blocks hold no photons (Vista recorded fewer frames than the
microscope acquired; every block after the first dropped frame then sits at the wrong tile / z)."""
import json

NB = r'X:\FLIM_2026\AnalyzeTCSPC_XYTZTile.ipynb'
raw = open(NB, encoding='utf-8', newline='').read()
indent = len(raw.splitlines()[1]) - len(raw.splitlines()[1].lstrip(' '))
nb = json.loads(raw)
c = next(c for c in nb['cells'] if c.get('id') == 'code-tile0b')
s = ''.join(c['source'])
assert 'empty z-blocks' not in s, 'already patched'
old = "_ph = np.array(_rep['photons_per_block'])\n"
new = """_ph = np.array(_rep['photons_per_block'])
_empty = np.flatnonzero(_ph.ravel() == 0)
if _empty.size:
    _t0, _z0 = divmod(int(_empty[0]), n_z_tile)
    warnings.warn(f'{_empty.size} of {_ph.size} empty z-blocks (first: tile {_t0} z {_z0}): Vista recorded fewer '
                  f'frames than the microscope acquired. Blocks after the first dropped frame sit at the wrong '
                  f'tile / z, so registration and stitching are unreliable.', RuntimeWarning)
"""
assert s.count(old) == 1
c['source'] = s.replace(old, new).splitlines(keepends=True)
c['outputs'], c['execution_count'] = [], None
with open(NB, 'w', encoding='utf-8', newline='') as f:
    f.write(json.dumps(nb, indent=indent, ensure_ascii=False) + ('\n' if raw.endswith('\n') else ''))
print('patched code-tile0b')
