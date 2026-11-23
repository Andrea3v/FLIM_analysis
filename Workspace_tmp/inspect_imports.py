import json, sys
sys.stdout.reconfigure(encoding='utf-8')

path = r'X:\FLIM_2026\AnalyzeTCSPC.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for idx in range(4):
    c = nb['cells'][idx]
    src = c['source'] if isinstance(c['source'], str) else ''.join(c['source'])
    print(f"\n=== Cell idx={idx}  id={c['id']}  type={c['cell_type']} ===")
    print(src[:800])
    print("=== END ===")
