"""AnalyzeTCSPC_XYTZTime.ipynb: markdown description above code-tile0d (time sum + per-timepoint stacks)."""
import json

NB = r'X:\FLIM_2026\AnalyzeTCSPC_XYTZTime.ipynb'
raw = open(NB, encoding='utf-8', newline='').read()
indent = len(raw.splitlines()[1]) - len(raw.splitlines()[1].lstrip(' '))
nb = json.loads(raw)
ids = [c.get('id') for c in nb['cells']]
assert 'md-tile0d' not in ids, 'already patched'

MD = '''### Step 0d: time sum and per-timepoint TCSPC stacks on one canvas
Every timepoint (and tile) goes onto one canvas at its drift-corrected whole-voxel position from Step 0c, so a canvas
voxel is the same spot in the tissue at every timepoint. The decays are only moved, never interpolated, and the photon
count is checked (fused = tiles). The canvas is larger than one stack by the drift range.
- **Time sum** → `stitched/` (`data_folder` of Steps 1–16): per voxel, the decays of all timepoints (and overlapping
  tiles) summed as raw counts; one `TYX` TIFF per z (`<base>_stitched_Z###.tif`, uint8, uint16 if a voxel exceeds 255).
- **Each timepoint** → `stitched_T<tt>/` (Step 17): same canvas, that timepoint's photons only.
- **Segmentation intensity** `<base>_stitched_intensity.tif` (Step 2): every timepoint and tile scaled to its own
  1–99.5 % range (evens out bleaching), then feather-blended: an average, not a sum.
- **Coverage** `<base>_coverage.tif`: timepoints × tiles per voxel. At the canvas edges drift leaves fewer timepoints,
  so the time sum has fewer photons there; Step 17 uses the per-timepoint coverage for `fov_frac`.

Files go next to the raw file and are reused when present; new positions from 0c or `rerun_fuse = True` rebuild them.
Disk: the timepoint stacks take about as much as the converted tiles, the sum one stack more.
With `n_timepoints = 1` this is the plain tile stitch.'''

i = ids.index('code-tile0d')
nb['cells'].insert(i, dict(cell_type='markdown', id='md-tile0d', metadata={}, source=MD.splitlines(keepends=True)))
with open(NB, 'w', encoding='utf-8', newline='') as f:
    f.write(json.dumps(nb, indent=indent, ensure_ascii=False) + ('\n' if raw.endswith('\n') else ''))
print('added md-tile0d above code-tile0d')
