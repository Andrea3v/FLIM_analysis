import json, sys
sys.stdout.reconfigure(encoding='utf-8')

path = r'X:\FLIM_2026\AnalyzeTCSPC.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

c = nb['cells'][30]
assert c['id'] == 'b3fb5433'
src = c['source'] if isinstance(c['source'], str) else ''.join(c['source'])

# bump search window still references _peak_bulk — should use target now
old = (
    "_srch_lo = min(nbin - 1, _peak_bulk + max(5, int(round(bump_srch_lo_ns / dt))))\n"
    "_srch_hi = min(nbin,     _peak_bulk + int(round(bump_srch_hi_ns / dt)))\n"
)
new = (
    "_srch_lo = min(nbin - 1, target + max(5, int(round(bump_srch_lo_ns / dt))))\n"
    "_srch_hi = min(nbin,     target + int(round(bump_srch_hi_ns / dt)))\n"
)
assert old in src, 'bump-search anchor not found'
src = src.replace(old, new)

c['source'] = src
with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print('done')
