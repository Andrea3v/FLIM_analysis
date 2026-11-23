import json, sys
sys.stdout.reconfigure(encoding='utf-8')

path = r'X:\FLIM_2026\AnalyzeTCSPC.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

c = nb['cells'][28]
assert c['id'] == '98f90894', f"Wrong cell: {c['id']}"
src = c['source'] if isinstance(c['source'], str) else ''.join(c['source'])

# ── 1. Create z_decay_plots and FLIM_RGB_planes dirs before the loop ─────────
old_dirs = (
    "# ── open all TiffWriters once — keeps planes in the same series ───────────────\n"
)
new_dirs = (
    "# ── create output subfolders ─────────────────────────────────────────────────\n"
    "_zdecay_dir  = os.path.join(newFolder, 'z_decay_plots')\n"
    "_flimrgb_dir = os.path.join(newFolder, 'FLIM_RGB_planes')\n"
    "os.makedirs(_zdecay_dir,  exist_ok=True)\n"
    "os.makedirs(_flimrgb_dir, exist_ok=True)\n"
    "\n"
    "# ── open all TiffWriters once — keeps planes in the same series ───────────────\n"
)
assert old_dirs in src, 'TiffWriters anchor not found'
src = src.replace(old_dirs, new_dirs)

# ── 2. Route decay plots to z_decay_plots subfolder ──────────────────────────
old_save = (
    "        _fig_d.savefig(os.path.join(newFolder, f'{FName}_z{z:03d}_decay.png'),\n"
    "                       dpi=100, bbox_inches='tight')\n"
)
new_save = (
    "        _fig_d.savefig(os.path.join(_zdecay_dir, f'{FName}_z{z:03d}_decay.png'),\n"
    "                       dpi=100, bbox_inches='tight')\n"
)
assert old_save in src, 'decay savefig anchor not found'
src = src.replace(old_save, new_save)

# ── 3. Route FastFLIM PNG base to FLIM_RGB_planes, remove redundant makedirs ─
old_png = (
    "        # ── 8. False-colour FLIM image ────────────────────────────────────────\n"
    "        _png_base = os.path.join(newFolder, f'{FName}_z{z:03d}')\n"
    "        LTImg_z, _ = FastFLIM_mod_AT0(_png_base, sumZ_norm, tav_filt_z, thresh_norm, limits,\n"
    "                                       show=(z == mid_idx))\n"
    "        # save in dedicated subfolder for FLIM RGB stacks, to keep single-plane PNGs in one place\n"
    "        _png_path = os.path.join(newFolder, 'FLIM_RGB_planes', f'{FName}_z{z:03d}_FLIM.png')\n"
    "        os.makedirs(os.path.dirname(_png_path), exist_ok=True)\n"
    "        _tw_rgb.write((np.clip(LTImg_z, 0, 1) * 255).astype(np.uint8), photometric='rgb')\n"
)
new_png = (
    "        # ── 8. False-colour FLIM image ────────────────────────────────────────\n"
    "        _png_base = os.path.join(_flimrgb_dir, f'{FName}_z{z:03d}')\n"
    "        LTImg_z, _ = FastFLIM_mod_AT0(_png_base, sumZ_norm, tav_filt_z, thresh_norm, limits,\n"
    "                                       show=(z == mid_idx))\n"
    "        _tw_rgb.write((np.clip(LTImg_z, 0, 1) * 255).astype(np.uint8), photometric='rgb')\n"
)
assert old_png in src, 'FastFLIM PNG anchor not found'
src = src.replace(old_png, new_png)

c['source'] = src
with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print('done: step-13 subfolders patched')
