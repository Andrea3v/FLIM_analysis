import json, sys
sys.stdout.reconfigure(encoding='utf-8')

path = r'X:\FLIM_2026\AnalyzeTCSPC.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

c = nb['cells'][4]
assert c['id'] == 'code-03', f"Wrong cell: {c['id']}"
src = c['source'] if isinstance(c['source'], str) else ''.join(c['source'])

old = (
    "def _pick_folder(title):\n"
    "    root = tk.Tk(); root.withdraw(); root.attributes('-topmost', True)\n"
    "    p = filedialog.askdirectory(title=title)\n"
    "    root.destroy()\n"
    "    return p or None\n"
)
new = (
    "def _pick_folder(title, initialdir=None):\n"
    "    root = tk.Tk(); root.withdraw(); root.attributes('-topmost', True)\n"
    "    p = filedialog.askdirectory(title=title, initialdir=initialdir)\n"
    "    root.destroy()\n"
    "    return p or None\n"
)
assert old in src, 'pick_folder anchor not found'
src = src.replace(old, new)

old2 = "data_folder = _pick_folder('Select folder containing data TIFFs (z-stack)')\n"
new2 = "data_folder = _pick_folder('Select folder containing data TIFFs (z-stack)', initialdir=r'X:\\FLIM_2026\\test_data')\n"
assert old2 in src, 'call anchor not found'
src = src.replace(old2, new2)

c['source'] = src
with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print('done')
