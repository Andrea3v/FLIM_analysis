import json, sys
sys.stdout.reconfigure(encoding='utf-8')

path = r'X:\FLIM_2026\AnalyzeTCSPC.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# ── 1. Add datetime import to code-01 (idx=1) ────────────────────────────────
c1 = nb['cells'][1]
assert c1['id'] == 'code-01', f"Wrong cell: {c1['id']}"
src1 = c1['source'] if isinstance(c1['source'], str) else ''.join(c1['source'])
assert 'from datetime import datetime' not in src1, 'datetime already imported'
src1 = src1.replace(
    'import numpy as np',
    'from datetime import datetime\nimport numpy as np'
)
c1['source'] = src1

# ── 2. Patch code-03 (idx=4) ─────────────────────────────────────────────────
c4 = nb['cells'][4]
assert c4['id'] == 'code-03', f"Wrong cell: {c4['id']}"
src4 = c4['source'] if isinstance(c4['source'], str) else ''.join(c4['source'])

# 2a. Remove hardcoded newFolder from params block
src4 = src4.replace(
    "newFolder = 'X:/FLIM_2026/results'\n",
    ""
)

# 2b. Remove standalone makedirs (was right after the params block)
src4 = src4.replace(
    "os.makedirs(newFolder, exist_ok=True)\n",
    ""
)

# 2c. After FName is derived, create timestamped newFolder as subfolder of data_folder
old_fname = "FName = os.path.basename(data_folder)\n"
new_fname = (
    "FName = os.path.basename(data_folder)\n"
    "_ts       = datetime.now().strftime('%Y%m%d_%H%M')\n"
    "newFolder = os.path.join(data_folder, f'{_ts}_FLIM_Analysis')\n"
    "os.makedirs(newFolder, exist_ok=True)\n"
)
assert old_fname in src4, 'FName anchor not found'
src4 = src4.replace(old_fname, new_fname)

c4['source'] = src4

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print('done: datetime import + newFolder patched')
