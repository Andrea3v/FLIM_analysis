"""Group the Step-0 outputs of AnalyzeTCSPC_XYTZTime.ipynb below <raw folder>/analysis/:
tiles_uint8/, intensity_coverage/ (stitched intensity + coverage, time sum / per timepoint / Olympus),
stitched/ (time sum = data_folder, FLIM_Analysis inside), stitched_timepoints/T<tt>/ (registered planes)."""
import json
import sys

NB = r'X:\FLIM_2026\AnalyzeTCSPC_XYTZTime.ipynb'

EDITS = {
    'code-tile0a': [(
"""raw_dir   = os.path.dirname(raw_path)
tiles_dir = os.path.join(raw_dir, 'tiles_uint8')
pos_json  = os.path.join(raw_dir, f'{vista_info["base"]}_tile_positions.json')
print(f'Raw file: {raw_path}  ({os.path.getsize(raw_path) / 1e9:.1f} GB)')
""",
"""raw_dir   = os.path.dirname(raw_path)
print(f'Raw file: {raw_path}  ({os.path.getsize(raw_path) / 1e9:.1f} GB)')

# ── outputs: everything Steps 0-17 write goes below <raw folder>/analysis/ ─────
out_root  = os.path.join(raw_dir, 'analysis')
tiles_dir = os.path.join(out_root, 'tiles_uint8')            # uint8 tile stacks per timepoint (0b)
int_dir   = os.path.join(out_root, 'intensity_coverage')     # stitched intensity + coverage volumes (0d, 0e)
tp_dir    = os.path.join(out_root, 'stitched_timepoints')    # registered TCSPC planes per timepoint: T00/, T01/, ... (0d)
pos_json  = os.path.join(out_root, f'{vista_info["base"]}_tile_positions.json')
vt.move_legacy_outputs(raw_dir, vista_info['base'], out_root, int_dir, tp_dir)   # old flat layout -> analysis/
os.makedirs(int_dir, exist_ok=True)
print(f'Outputs   -> {out_root}')
""")],
    'code-tile0b': [(
"""    with open(os.path.join(raw_dir, f'{vista_info["base"]}_block_map.json'), 'w') as _f:""",
"""    with open(os.path.join(out_root, f'{vista_info["base"]}_block_map.json'), 'w') as _f:""")],
    'code-tile0d': [
        ("""    stitch = vt.fuse_tiles(conv['tile_paths'], tile_reg, tile_int, raw_dir, vista_info['base'],
                           overwrite=rerun_fuse, block_of_slot=block_of_slot)
else:   # one canvas for all timepoints: their sum -> stitched/ (Steps 1-16), timepoint t -> stitched_T<tt>/ (Step 17)""",
         """    stitch = vt.fuse_tiles(conv['tile_paths'], tile_reg, tile_int, out_root, vista_info['base'],
                           overwrite=rerun_fuse, block_of_slot=block_of_slot, int_dir=int_dir)
else:   # one canvas for all timepoints: their sum -> stitched/ (Steps 1-16), timepoint t -> stitched_timepoints/T<tt>/ (Step 17)"""),
        ("""                           _ti_flat, raw_dir, vista_info['base'], overwrite=rerun_fuse)""",
         """                           _ti_flat, out_root, vista_info['base'], overwrite=rerun_fuse, int_dir=int_dir)"""),
        ("""                                      tile_int_t[t], raw_dir, f'{vista_info["base"]}_T{t:02d}',
                                      overwrite=rerun_fuse, stitched_name=f'stitched_T{t:02d}'))""",
         """                                      tile_int_t[t], out_root, f'{vista_info["base"]}_T{t:02d}',
                                      overwrite=rerun_fuse, stitched_name=os.path.join(tp_dir, f'T{t:02d}'),
                                      int_dir=int_dir))"""),
        ("""if stitch_t:
    print(f'Per timepoint (Step 17): {", ".join(os.path.basename(s["stitched_dir"]) for s in stitch_t)}')""",
         """print(f'Intensity / coverage: {int_dir}')
if stitch_t:
    print(f'Per timepoint (Step 17): {tp_dir}  ({", ".join(os.path.basename(s["stitched_dir"]) for s in stitch_t)})')"""),
    ],
    'code-tile0e': [(
"""    stitch_oir = os.path.join(raw_dir, f'{vista_info["base"]}_stitched_{oir_ch["name"]}.tif')""",
"""    stitch_oir = os.path.join(int_dir, f'{vista_info["base"]}_stitched_{oir_ch["name"]}.tif')""")],
    'md-tile0': [
        ("""decay bins as channels) in `tiles_uint8/`, in the""",
         """decay bins as channels) in `analysis/tiles_uint8/`, in the"""),
        ("""→ `stitched/` (= `data_folder`
  for Steps 1–16). Segmentation intensity is per-tile normalised and feather-blended → `<base>_stitched_intensity.tif`""",
         """→ `analysis/stitched/` (= `data_folder`
  for Steps 1–16). Segmentation intensity is per-tile normalised and feather-blended →
  `analysis/intensity_coverage/<base>_stitched_intensity.tif`"""),
        ("""stitched with the **same tile positions** as 0c → `<base>_stitched_<CH>.tif`""",
         """stitched with the **same tile positions** as 0c → `intensity_coverage/<base>_stitched_<CH>.tif`"""),
        ("""**each timepoint** → `stitched_T<tt>/` (Step 17""",
         """**each timepoint** → `stitched_timepoints/T<tt>/` (Step 17"""),
        ("""(re-registering re-stitches automatically). Tile offsets: `<base>_tile_positions.json`.""",
         """(re-registering re-stitches automatically). Tile offsets: `<base>_tile_positions.json`.

**Output layout:** everything goes below `analysis/` in the raw file's folder (the raw folder keeps only the raw data):
```
analysis/
  <base>_tile_positions.json, <base>_block_map.json
  tiles_uint8/            uint8 tile stacks (per timepoint) + conversion report        (0b)
  intensity_coverage/     <base>[_T<tt>]_stitched_intensity.tif, <base>[_T<tt>]_coverage.tif,
                          <base>_stitched_<CH>.tif (Olympus)                           (0d, 0e)
  stitched/               time-sum TCSPC planes = data_folder of Steps 1–16           (0d)
    <YYYYMMDD_HHMM>_FLIM_Analysis/   results of Steps 1–17
  stitched_timepoints/
    T00/, T01/, ...       registered (drift-corrected) TCSPC planes per timepoint      (0d, Step 17)
```
Outputs of the old flat layout (loose next to the raw file) are moved here by Step 0a: a rename on the same drive,
so the modification times and every cache stay valid; nothing is overwritten."""),
    ],
    'md-tile0d': [
        ("""- **Time sum** → `stitched/` (`data_folder` of Steps 1–16)""",
         """- **Time sum** → `analysis/stitched/` (`data_folder` of Steps 1–16)"""),
        ("""- **Each timepoint** → `stitched_T<tt>/` (Step 17): same canvas, that timepoint's photons only.
- **Segmentation intensity** `<base>_stitched_intensity.tif` (Step 2)""",
         """- **Each timepoint** → `analysis/stitched_timepoints/T<tt>/` (Step 17): same canvas, that timepoint's photons only
  (`<base>_T<tt>_stitched_Z###.tif`).
- **Segmentation intensity** `analysis/intensity_coverage/<base>_stitched_intensity.tif` (Step 2)"""),
        ("""- **Coverage** `<base>_coverage.tif`: timepoints × tiles per voxel.""",
         """- **Coverage** `intensity_coverage/<base>_coverage.tif`: timepoints × tiles per voxel (per timepoint:
  `<base>_T<tt>_coverage.tif`, next to `<base>_T<tt>_stitched_intensity.tif`)."""),
        ("""Files go next to the raw file and are reused when present;""",
         """Files go to `analysis/` in the raw file's folder and are reused when present;"""),
    ],
    'md-tile0e': [(
"""Output: `<base>_stitched_<CH>.tif` (z, y, x;""",
"""Output: `analysis/intensity_coverage/<base>_stitched_<CH>.tif` (z, y, x;""")],
    'md-03': [(
"""`data_folder` is the `stitched/` folder from Step 0d;""",
"""`data_folder` is the `analysis/stitched/` folder from Step 0d (the dated `*_FLIM_Analysis` results folder is created inside it);""")],
    'md-17': [(
"""For each timepoint stack (`stitched_T<tt>/`), per cell:""",
"""For each timepoint stack (`analysis/stitched_timepoints/T<tt>/`), per cell:""")],
}


def main():
    nb = json.load(open(NB, encoding='utf-8'))
    cells = {c.get('id'): c for c in nb['cells']}
    for cid, edits in EDITS.items():
        c = cells[cid]
        src = ''.join(c['source'])
        for old, new in edits:
            n = src.count(old)
            if n != 1:
                sys.exit(f'{cid}: expected 1 match, found {n}:\n{old}')
            src = src.replace(old, new)
        c['source'] = src.splitlines(keepends=True)
    with open(NB, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(nb, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print('patched', ', '.join(EDITS))


if __name__ == '__main__':
    main()
