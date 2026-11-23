"""Remove the raw-Vista quick-look cells from AnalyzeTCSPC.ipynb and derive
AnalyzeTCSPC_XYTZTile.ipynb (Step 0: tiled XYTZ conversion / registration / stitching)."""
import copy
import json

SRC = r'X:\FLIM_2026\AnalyzeTCSPC.ipynb'
DST = r'X:\FLIM_2026\AnalyzeTCSPC_XYTZTile.ipynb'
RAW_IDS = {'md-raw0', 'code-raw0', 'md-raw1', 'code-raw1'}


def lines(s):
    out = s.split('\n')
    return [x + '\n' for x in out[:-1]] + ([out[-1]] if out[-1] else [])


def md(cid, text):
    return {'cell_type': 'markdown', 'id': cid, 'metadata': {}, 'source': lines(text)}


def code(cid, text):
    return {'cell_type': 'code', 'execution_count': None, 'id': cid, 'metadata': {},
            'outputs': [], 'source': lines(text)}


def cell(nb, cid):
    return next(c for c in nb['cells'] if c.get('id') == cid)


def replace(c, old, new):
    s = ''.join(c['source'])
    assert s.count(old) == 1, f'{c.get("id")}: expected exactly one match for {old[:60]!r}'
    c['source'] = lines(s.replace(old, new))


def dump(nb, path, indent, trailing_nl):
    with open(path, 'w', encoding='utf8', newline='') as f:
        f.write(json.dumps(nb, indent=indent, ensure_ascii=False) + ('\n' if trailing_nl else ''))


raw = open(SRC, encoding='utf8', newline='').read()
indent = 1 if raw.startswith('{\n "') else 2
nb = json.loads(raw)

# ── 1. original notebook: drop the quick-look cells ───────────────────────────
nb['cells'] = [c for c in nb['cells'] if c.get('id') not in RAW_IDS]
dump(nb, SRC, indent, raw.endswith('\n'))

# ── 2. tiled notebook ─────────────────────────────────────────────────────────
nt = copy.deepcopy(nb)
for c in nt['cells']:
    if c['cell_type'] == 'code':
        c['outputs'] = []
        c['execution_count'] = None

cell(nt, 'md-01')['source'] = lines(
    '# AnalyzeTCSPC — tiled XYTZ\n'
    'TCSPC-FLIM analysis for single-file ISS Vista exports of an Olympus MATL tile grid '
    '(`..._Decay<nbin>_<nx>x<ny>_Z<tiles·z>_...tif`).\n'
    '**Step 0** converts the raw float32 file to uint8 tiles, registers the tiles and writes a stitched '
    'TCSPC volume (one TYX TIFF per z); Steps 1–16 are the `AnalyzeTCSPC.ipynb` pipeline run on that volume.\n'
    '\n'
    'Run cells **in order**. Edit the **Step 0a** cell for each experiment.')

STEP0_MD = '''## Step 0: Tiled XYTZ acquisition → uint8 tiles → registration → stitched TCSPC
- **0a** parameters. Bins, tile size and number of z-blocks come from the file name (`_Decay128`, `_512x512`, `_Z96`).
- **0b** float32 → uint8: one ImageJ hyperstack per tile (`ZCYX`, decay bins as channels) in `tiles_uint8/`.
  Vista writes classic TIFFs past 4 GB: pages after ~4 GB have no IFD, and their IFDs are written over plane data
  in the first 4 GB. Those pixels are located exactly and zeroed (reported as *corrupted bins*).
- **0c** registration: masked normalised cross-correlation on neighbour overlaps → integer (dz, dy, dx) per tile,
  global least squares. `tile_layout='auto'` scores raster/snake × start corner (and the transposed grid);
  `tile_layout_fallback` is used when nothing matches or it ties with the best.
- **0d** stitching: TCSPC overlaps are **summed** (raw counts, more photons per decay) → `stitched/` (= `data_folder`
  for Steps 1–16). Segmentation intensity is per-tile normalised and feather-blended → `<base>_stitched_intensity.tif`
  (used by Step 2 instead of summing `data_paths`, so overlaps are not double-counted).

Outputs are reused when present; set `rerun_convert / rerun_register / rerun_fuse` to redo a step
(re-registering re-stitches automatically). Tile offsets: `<base>_tile_positions.json`.'''

STEP0A = r'''# ── Step 0a: tiled acquisition parameters ─────────────────────────────────────
import vista_tiles as vt
importlib.reload(vt)

def _pick_raw():
    root = tk.Tk(); root.withdraw(); root.attributes('-topmost', True)
    p = filedialog.askopenfilename(title='Select ISS Vista XYTZ TIFF',
                                   filetypes=[('TIFF files', '*.tif *.tiff'), ('All files', '*.*')])
    root.destroy()
    return p or None

# ── raw file — swap comments to switch between hardcoded and interactive ──────
raw_path = r'D:\Peredox_FLIM\20261029_Peredox_336_awake\XYZT1_336__Decay128_512x512_Z96_Ch1_T1_P1_S1\XYZT1_336__Decay128_512x512_Z96_Ch1_T1_P1_S1.tif'
# raw_path = _pick_raw()
if not raw_path:
    raise RuntimeError('No raw file selected.')

# ── tile grid ─────────────────────────────────────────────────────────────────
tile_grid            = (3, 2)       # (rows, cols); 'auto' also tests the transposed grid
tile_overlap         = 0.20         # nominal fractional overlap (Olympus 2P default); true step is measured
tile_layout          = 'auto'       # 'auto' or one of vt.LAYOUTS, e.g. 'snake_tl', 'raster_tl'
tile_layout_fallback = 'snake_tl'   # used when auto finds no match or it ties with the best
tile_shape_px        = None         # (ny, nx) per tile; None = from file name '_<nx>x<ny>_'
max_dz               = 2            # max z offset between neighbouring tiles (slices)

# ── voxel size & laser period (Olympus metadata loader: later) ───────────────
xy_pixel_um = 0.99   # XY pixel size in µm
z_step_um   = 3.0    # Z step size in µm
Tcycle      = 12.5   # laser period in ns (80 MHz); a .fbs.xml next to the data still overrides it

rerun_convert  = False
rerun_register = False
rerun_fuse     = False

# ── derived ───────────────────────────────────────────────────────────────────
vista_info = vt.parse_vista_name(raw_path)
if tile_shape_px is not None and tuple(tile_shape_px) != (vista_info['ny'], vista_info['nx']):
    raise ValueError(f'tile_shape_px {tile_shape_px} != file name {vista_info["ny"]}x{vista_info["nx"]}')
n_tiles  = tile_grid[0] * tile_grid[1]
if vista_info['n_blocks'] % n_tiles:
    raise ValueError(f'_Z{vista_info["n_blocks"]} is not a multiple of {n_tiles} tiles')
n_z_tile = vista_info['n_blocks'] // n_tiles
raw_dir   = os.path.dirname(raw_path)
tiles_dir = os.path.join(raw_dir, 'tiles_uint8')
pos_json  = os.path.join(raw_dir, f'{vista_info["base"]}_tile_positions.json')

print(f'Raw file: {raw_path}')
print(f'  {vista_info["nbin"]} bins  |  tile {vista_info["ny"]}x{vista_info["nx"]} px  |  '
      f'{vista_info["n_blocks"]} z-blocks = {n_tiles} tiles x {n_z_tile} z  |  '
      f'{os.path.getsize(raw_path) / 1e9:.1f} GB')'''

STEP0B = r'''# ── Step 0b: float32 Vista TIFF → uint8 tiles ─────────────────────────────────
conv = vt.convert_to_uint8(raw_path, tiles_dir, n_pos=n_tiles, xy_pixel_um=xy_pixel_um,
                           z_step_um=z_step_um, overwrite=rerun_convert)
tile_int = conv['tile_int']          # (tile, z, y, x) photons summed over bins
_rep = conv['report']
print(f'Corrupted bins zeroed: {100 * _rep["corrupted_fraction"]:.3f}%  |  '
      f'residual junk: {_rep["residual_junk_pixels"]}  |  clipped >255: {_rep["clipped_pixels"]}  |  '
      f'max count/bin/px: {_rep["max_count"]:g}')

_ph = np.array(_rep['photons_per_block'])
fig, axs = plt.subplots(2, n_tiles, figsize=(3.2 * n_tiles, 5.2),
                        gridspec_kw=dict(height_ratios=[3, 1.2]))
for p in range(n_tiles):
    _mp = tile_int[p].max(axis=0)
    axs[0, p].imshow(_mp, cmap='gray', vmax=np.percentile(_mp, 99.5))
    axs[0, p].set_title(f'tile {p}  (max over z)', fontsize=9)
    axs[0, p].axis('off')
    axs[1, p].plot(_ph[p] / 1e3, '.-', lw=0.8)
    axs[1, p].set_xlabel('z')
axs[1, 0].set_ylabel('photons / 1e3')
plt.suptitle(f'{vista_info["base"]}  —  tiles in acquisition order', fontsize=9)
plt.tight_layout()
plt.show()'''

STEP0C = r'''# ── Step 0c: tile registration ────────────────────────────────────────────────
if os.path.exists(pos_json) and not rerun_register:
    tile_reg = vt.load_tile_positions(pos_json)
    print(f'Reloaded tile positions <- {os.path.basename(pos_json)}')
else:
    tile_reg = vt.register_tiles(tile_int, tile_grid, overlap=tile_overlap, layout=tile_layout,
                                 fallback=tile_layout_fallback, max_dz=max_dz)
    vt.save_tile_positions(pos_json, tile_reg, raw_path=raw_path)
    print(f'Saved -> {pos_json}')
    rerun_fuse = True   # new positions -> re-stitch

_pos = np.array(tile_reg['positions'])
_nz_t, _ny_t, _nx_t = tile_reg['tile_shape']
_Zc, _Yc, _Xc = tile_reg['canvas_shape']
print(f'Layout {tile_reg["layout"]}  grid {tile_reg["grid"][0]} rows x {tile_reg["grid"][1]} cols  |  '
      f'canvas (z, y, x) = {tuple(tile_reg["canvas_shape"])}')
_df_pairs = pd.DataFrame([dict(pair=f'{p["a"]}->{p["b"]}', direction='down' if p['axis'] == 1 else 'right',
                               offset_zyx=tuple(p['offset']), NCC=round(p['score'], 3), used=p['used'],
                               residual_zyx=tuple(p['residual'])) for p in tile_reg['pairs']])
print(_df_pairs.to_string(index=False))

# QC: max-projection preview on the canvas with tile outlines
_prev = np.zeros((_Yc, _Xc), np.float32)
for p in range(n_tiles):
    _n = (tile_int[p].max(axis=0) - tile_reg['norm_lo'][p]) / max(tile_reg['norm_hi'][p] - tile_reg['norm_lo'][p], 1e-9)
    _y0, _x0 = _pos[p, 1], _pos[p, 2]
    _sub = _prev[_y0:_y0 + _ny_t, _x0:_x0 + _nx_t]
    np.maximum(_sub, _n, out=_sub)
fig, ax = plt.subplots(figsize=(8, 8 * _Yc / _Xc))
ax.imshow(_prev, cmap='gray', vmin=0, vmax=np.percentile(_prev, 99.5))
for p in range(n_tiles):
    ax.add_patch(plt.Rectangle((_pos[p, 2], _pos[p, 1]), _nx_t, _ny_t, fill=False, ec=f'C{p}', lw=1))
    ax.text(_pos[p, 2] + 10, _pos[p, 1] + 35, f'{p}  z0={_pos[p, 0]}', color=f'C{p}', fontsize=11)
ax.set_title(f'Registered tiles (max over z)  —  {tile_reg["layout"]}', fontsize=10)
plt.tight_layout()
plt.show()'''

STEP0D = r'''# ── Step 0d: stitched TCSPC (overlaps summed) + segmentation intensity ─────────
stitch = vt.fuse_tiles(conv['tile_paths'], tile_reg, tile_int, raw_dir, vista_info['base'],
                       overwrite=rerun_fuse)
stitched_dir = stitch['stitched_dir']

_fi  = tifffile.imread(stitch['int_path'])
_cov = tifffile.imread(stitch['cov_path'])
_zm  = _fi.shape[0] // 2
fig, axs = plt.subplots(1, 3, figsize=(16, 7))
for _a, _img, _ttl in [(axs[0], _fi[_zm], f'segmentation intensity  z={_zm}'),
                       (axs[1], _fi.max(axis=0), 'segmentation intensity  (max over z)')]:
    _a.imshow(_img, cmap='gray', vmin=0, vmax=np.percentile(_img, 99.5))
    _a.set_title(_ttl, fontsize=10); _a.axis('off')
_im = axs[2].imshow(_cov.max(axis=0), cmap='viridis', interpolation='nearest')
axs[2].set_title('tiles per voxel  (max over z)', fontsize=10); axs[2].axis('off')
plt.colorbar(_im, ax=axs[2], fraction=0.046)
plt.tight_layout()
plt.show()
print(f'data_folder for Steps 1-16: {stitched_dir}  ({len(stitch["plane_paths"])} planes)')'''

i = next(k for k, c in enumerate(nt['cells']) if c.get('id') == 'md-02') + 1
nt['cells'][i:i] = [md('md-tile0', STEP0_MD), code('code-tile0a', STEP0A), code('code-tile0b', STEP0B),
                    code('code-tile0c', STEP0C), code('code-tile0d', STEP0D)]

# ── Step 1: parameters / data folder come from Step 0 ─────────────────────────
c03 = cell(nt, 'code-03')
replace(c03, 'Tcycle    = 12.5      # fallback if no .fbs.xml found — overridden automatically\n',
        '# Tcycle: set in Step 0a (overridden if a .fbs.xml is found next to the data)\n')
replace(c03, '# ── voxel dimensions (µm) — update per acquisition ─────────────────────────────────────\n'
             'xy_pixel_um = 0.99   # XY pixel size in µm\n'
             'z_step_um   = 3.0    # Z step size in µm\n',
        '# ── voxel dimensions (µm): xy_pixel_um / z_step_um set in Step 0a ─────────────\n')
replace(c03, "data_folder = _pick_folder('Select folder containing data TIFFs (z-stack)', "
             "initialdir=r'X:\\FLIM_2026\\test_data')",
        "data_folder = stitched_dir   # stitched TCSPC planes from Step 0d\n"
        "# data_folder = _pick_folder('Select folder containing data TIFFs (z-stack)', "
        "initialdir=r'X:\\FLIM_2026\\test_data')")
replace(c03, '# ── FName derived from folder name ────────────────────────────────────────────\n'
             'FName = os.path.basename(data_folder)',
        '# ── FName from the raw Vista file name ────────────────────────────────────────\n'
        "FName = vista_info['base']")
replace(cell(nt, 'md-03'),
        'Edit the analysis parameters, then run the cell — file dialogs will open (IRF first, then data folder).',
        'Edit the analysis parameters, then run the cell. `data_folder` is the `stitched/` folder from Step 0d; '
        'voxel size and `Tcycle` come from Step 0a.')

# ── Step 2: segmentation intensity from Step 0d ───────────────────────────────
replace(cell(nt, 'a5874f7b'),
        "    # ── build 3D sumTCSPC volume ──────────────────────────────────────────\n"
        "    print(f'Loading {n_planes} planes...', end=' ', flush=True)\n"
        "    vol_sum3d = np.stack([\n"
        "        _to_yxt(tifffile.imread(p).astype(float), dim_data).sum(axis=2)\n"
        "        for p in data_paths\n"
        "    ], axis=0).astype(np.float32)\n"
        "    print(f'done.  Shape: {vol_sum3d.shape}')\n",
        "    # ── 3D intensity volume: per-tile normalised, feather-blended (Step 0d) ─\n"
        "    # (summing data_paths would double-count photons in the tile overlaps)\n"
        "    vol_sum3d = tifffile.imread(stitch['int_path']).astype(np.float32)\n"
        "    print(f'Loaded stitched intensity {vol_sum3d.shape}  <- {os.path.basename(stitch[\"int_path\"])}')\n")
replace(cell(nt, 'c001adff'),
        '- **No results** → runs 3D NLM denoising',
        '- **No results** → loads the stitched segmentation intensity from Step 0d, runs 3D NLM denoising')

dump(nt, DST, indent, True)
print('original cells:', len(nb['cells']), ' tiled notebook cells:', len(nt['cells']))
