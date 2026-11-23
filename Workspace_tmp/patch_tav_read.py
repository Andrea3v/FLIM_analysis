import json, sys
sys.stdout.reconfigure(encoding='utf-8')

path = r'X:\FLIM_2026\AnalyzeTCSPC.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# ── Fix 1: Step 16 (cell 35) — read all pages, not just the first series ─────
c16 = nb['cells'][35]
assert c16['id'] == '61a152a1', f"Wrong cell: {c16['id']}"
src16 = c16['source'] if isinstance(c16['source'], str) else ''.join(c16['source'])

old16 = (
    "    _tav_stack = tifffile.imread(stack_tav_filt).astype(float)  # (Z, Y, X)\n"
    "    if _tav_stack.ndim == 2:\n"
    "        _tav_stack = _tav_stack[np.newaxis]\n"
    "    _nz = _tav_stack.shape[0]\n"
    "    print(f'Loaded tav_filt_stack: {_nz} planes, shape {_tav_stack.shape}')\n"
)
new16 = (
    "    # tifffile writes each plane as a separate series when contiguous=True is\n"
    "    # omitted — imread then returns only the first page.  Read all pages explicitly.\n"
    "    with tifffile.TiffFile(stack_tav_filt) as _tf:\n"
    "        _tav_stack = np.stack([p.asarray() for p in _tf.pages]).astype(float)\n"
    "    if _tav_stack.ndim == 2:\n"
    "        _tav_stack = _tav_stack[np.newaxis]\n"
    "    _nz = _tav_stack.shape[0]\n"
    "    print(f'Loaded tav_filt_stack: {_nz} planes, shape {_tav_stack.shape}')\n"
)
assert old16 in src16, 'Step 16 anchor not found'
src16 = src16.replace(old16, new16)
c16['source'] = src16

# ── Fix 2: Step 13 (cell 28) — add contiguous=True to scalar/float writes ────
c13 = nb['cells'][28]
assert c13['id'] == '98f90894', f"Wrong cell: {c13['id']}"
src13 = c13['source'] if isinstance(c13['source'], str) else ''.join(c13['source'])

replacements = [
    (
        "        _tw_sum.write(sumZ.astype(np.float32))\n",
        "        _tw_sum.write(sumZ.astype(np.float32), contiguous=True)\n"
    ),
    (
        "        _tw_norm.write(sumZ_norm.astype(np.float32))\n",
        "        _tw_norm.write(sumZ_norm.astype(np.float32), contiguous=True)\n"
    ),
    (
        "        _tw_mask.write(maskZ.astype(np.float32))\n",
        "        _tw_mask.write(maskZ.astype(np.float32), contiguous=True)\n"
    ),
    (
        "        _tw_traw.write(tav_z.astype(np.float32))\n",
        "        _tw_traw.write(tav_z.astype(np.float32), contiguous=True)\n"
    ),
    (
        "        _tw_tflt.write(tav_filt_z.astype(np.float32))\n",
        "        _tw_tflt.write(tav_filt_z.astype(np.float32), contiguous=True)\n"
    ),
]
for old, new in replacements:
    assert old in src13, f'Step 13 anchor not found: {repr(old[:60])}'
    src13 = src13.replace(old, new)

c13['source'] = src13

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print('done')
