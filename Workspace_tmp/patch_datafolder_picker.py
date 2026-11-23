import json, sys
sys.stdout.reconfigure(encoding='utf-8')

path = r'X:\FLIM_2026\AnalyzeTCSPC.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

c = nb['cells'][4]
assert c['id'] == 'code-03', f"Wrong cell: {c['id']}"
src = c['source'] if isinstance(c['source'], str) else ''.join(c['source'])

old = (
    "data_folder = r'X:\\FLIM_2026\\test_data\\XYZ_288_X0.587_Y0.938_1'\n"
    "# irf_path    = _pick_file('Select IRF TIFF')\n"
    "# data_folder = _pick_folder('Select folder containing data TIFFs (z-stack)')\n"
)
new = (
    "# irf_path    = _pick_file('Select IRF TIFF')\n"
    "data_folder = _pick_folder('Select folder containing data TIFFs (z-stack)')\n"
)
assert old in src, 'data_folder anchor not found'
src = src.replace(old, new)

c['source'] = src
with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print('done')
