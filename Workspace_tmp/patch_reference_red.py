"""MultiDay_Reference.ipynb: interim reference = 2026-11-06 Olympus red stitched volume (11-02 excluded: Vista dropped
planes); tuning crop inside the new canvas.  AnalyzeTCSPC_XYTZTile.ipynb Step 2b: register the session's segmentation
volume (Olympus red when present) to the reference."""
import json


def load(path):
    raw = open(path, encoding='utf-8', newline='').read()
    return raw, json.loads(raw), len(raw.splitlines()[1]) - len(raw.splitlines()[1].lstrip(' '))


def save(path, raw, nb, indent):
    with open(path, 'w', encoding='utf-8', newline='') as f:
        f.write(json.dumps(nb, indent=indent, ensure_ascii=False) + ('\n' if raw.endswith('\n') else ''))


def sub(nb, cid, old, new):
    c = next(c for c in nb['cells'] if c.get('id') == cid)
    s = ''.join(c['source'])
    assert s.count(old) == 1, (cid, old[:60], s.count(old))
    c['source'] = s.replace(old, new).splitlines(keepends=True)
    if c['cell_type'] == 'code':
        c['outputs'], c['execution_count'] = [], None


REF = r'X:\FLIM_2026\MultiDay_Reference.ipynb'
raw, nb, ind = load(REF)
sub(nb, 'mdr-00', "Fluoview-stitched ZYX TIFF), or, until it exists, a session's `<base>_stitched_intensity.tif`.",
    "Fluoview-stitched ZYX TIFF), or, until it exists, a session's stitched volume: preferably the Olympus red channel\n"
    "   (`<base>_stitched_CH1.tif`, complete and without the diffuse green background), else `<base>_stitched_intensity.tif`.\n"
    "   Do not use a session where Vista dropped planes without Olympus tiles to repair it (e.g. 11-02).")
sub(nb, 'coder-02', '''# dedicated reference stack (ZYX TIFF) + its voxel size, or a session's stitched intensity (interim reference)
ref_path     = r'D:\\Peredox_FLIM\\XYZT1_336_X+0.50_Y+0.80_Decay128_512x512_Z186_Ch1_T1_P1_S1\\XYZT1_336_X+0.50_Y+0.80_Decay128_512x512_Z186_Ch1_T1_P1_S1_stitched_intensity.tif'
ref_spacing  = (3.0, 0.99, 0.99)   # (dz, dy, dx) µm''',
    '''# dedicated reference stack (ZYX TIFF) + its voxel size, or a session's stitched volume (interim reference).
# Interim: the 2026-11-06 session's Olympus red channel. Not 11-02 (..._X+0.50_Y+0.80_..._Z186): Vista dropped
# 8 planes there and, without Olympus tiles, they cannot be put back, so its stitch is unreliable.
ref_path     = r'D:\\Peredox_FLIM\\XYZTTile1_336_X0.4Y-2.8_Decay128_512x512_Z210_Ch1_T1_P1_S1\\XYZTTile1_336_X0.4Y-2.8_Decay128_512x512_Z210_Ch1_T1_P1_S1_stitched_CH1.tif'
ref_spacing  = (4.0, 0.9944, 0.9944)   # (dz, dy, dx) µm''')
sub(nb, 'coder-03', "crop_yx      = (250, 700)            # top-left (y, x) of the crop, px",
    "crop_yx      = (600, 300)            # top-left (y, x) of the crop, px")
save(REF, raw, nb, ind)

TILE = r'X:\FLIM_2026\AnalyzeTCSPC_XYTZTile.ipynb'
raw, nb, ind = load(TILE)
sub(nb, 'md-2b', "1. **Register** this session's stitched intensity (fixed) to the reference (moving), in µm:",
    "1. **Register** this session's segmentation volume (Olympus red when present, else the FLIM intensity; fixed) to the\n"
    "   reference (moving), in µm:")
sub(nb, 'code-2b', "    _ses_p     = md.reg_prep(vol_sum3d, _ses_sp)",
    "    _ses_p     = md.reg_prep(vol_seg3d, _ses_sp)      # Olympus red when present (Step 2), else the FLIM intensity")
save(TILE, raw, nb, ind)
print('patched MultiDay_Reference.ipynb (mdr-00, coder-02, coder-03) and AnalyzeTCSPC_XYTZTile.ipynb (md-2b, code-2b)')
