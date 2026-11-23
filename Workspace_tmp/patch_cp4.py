import json, sys
sys.stdout.reconfigure(encoding='utf-8')

path = r'X:\FLIM_2026\AnalyzeTCSPC.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

c = nb['cells'][6]
assert c['id'] == 'a5874f7b', f"Wrong cell: {c['id']}"
src = c['source'] if isinstance(c['source'], str) else ''.join(c['source'])

# 1. model type default value + comment
src = src.replace(
    "cp_model_type      = 'cyto3' # 'cyto3' | 'cyto2' | 'nuclei'",
    "cp_model_type      = 'cpsam' # Cellpose 4 unified model; 'cyto3'/'nuclei' are aliases"
)

# 2. CellposeModel init — drop model_type arg (ignored in v4, causes warning)
src = src.replace(
    "_cp_model = _cp_models.CellposeModel(gpu=cp_gpu, model_type=cp_model_type)",
    "_cp_model = _cp_models.CellposeModel(gpu=cp_gpu)"
)

# 3. print statement still useful to show params, just drop model_type from it
src = src.replace(
    "print(f'Running Cellpose 3D  model={cp_model_type}  diam={cp_diameter}px  '",
    "print(f'Running Cellpose 3D  model=cpsam  diam={cp_diameter}px  '"
)

c['source'] = src
with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print('done')
