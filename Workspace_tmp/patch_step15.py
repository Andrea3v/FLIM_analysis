import json, sys
sys.stdout.reconfigure(encoding='utf-8')

path = r'X:\FLIM_2026\AnalyzeTCSPC.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

c = nb['cells'][33]
assert c['id'] == '6e4891ec', f"Wrong cell: {c['id']}"

# source can be list of strings or one string — normalise to single string
src = c['source'] if isinstance(c['source'], str) else ''.join(c['source'])

# 1. Add percentile computation before the theta line
old1 = '    _theta = np.linspace(0, np.pi, 400)\n'
new1 = (
    '    _aw_lo, _aw_hi = np.nanpercentile(_aw, [5, 95])  # clamp colormap to 5-95 pct\n'
    '    _vlims = [(_aw_lo, _aw_hi), (None, None)]        # (vmin, vmax) per panel\n'
    '\n'
    '    _theta = np.linspace(0, np.pi, 400)\n'
)
assert old1 in src, 'anchor 1 not found'
src = src.replace(old1, new1)

# 2. Extend the for-loop zip to include vlims
old2 = (
    "    for _ax, _c, _lbl, _cm in zip(\n"
    "            axes_cp,\n"
    "            [_aw, _cz],\n"
    "            ['aw_LT (ns)', f'centroid z  (0 – {n_planes - 1})'],\n"
    "            ['plasma', 'viridis']):\n"
)
new2 = (
    "    for _ax, _c, _lbl, _cm, (_vmin, _vmax) in zip(\n"
    "            axes_cp,\n"
    "            [_aw, _cz],\n"
    "            ['aw_LT (ns)', f'centroid z  (0 – {n_planes - 1})'],\n"
    "            ['plasma', 'viridis'],\n"
    "            _vlims):\n"
)
assert old2 in src, 'anchor 2 not found'
src = src.replace(old2, new2)

# 3. Add vmin/vmax to the scatter call
old3 = (
    "        _sc = _ax.scatter(_g_plot, _s_plot, c=_c, cmap=_cm, s=60,\n"
    "                          edgecolors='black', linewidths=0.5, zorder=5)\n"
)
new3 = (
    "        _sc = _ax.scatter(_g_plot, _s_plot, c=_c, cmap=_cm, s=60,\n"
    "                          edgecolors='black', linewidths=0.5, zorder=5,\n"
    "                          vmin=_vmin, vmax=_vmax)\n"
)
assert old3 in src, 'anchor 3 not found'
src = src.replace(old3, new3)

c['source'] = src
with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print('done')
