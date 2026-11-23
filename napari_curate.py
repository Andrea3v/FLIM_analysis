"""
napari_curate.py
Curate a 3D label volume in napari (run with the napari-env interpreter, see MultiDay_Reference.ipynb).

    python napari_curate.py <image.tif> <labels.tif> <out_labels.tif> <dz,dy,dx>

Edit the 'cells' layer (paint / erase / fill, new label = next free ID); Ctrl+Shift+S saves to
<out_labels.tif>, and closing the window saves as well.  IDs are kept as they are (no renumbering).
"""
import sys
import numpy as np
import tifffile
import napari

img_path, lab_path, out_path, spacing = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
scale = tuple(float(s) for s in spacing.split(','))

img = np.squeeze(tifffile.imread(img_path)).astype(np.float32)
lab = tifffile.imread(lab_path).astype(np.uint16)
if img.shape != lab.shape:
    sys.exit(f'image {img.shape} and labels {lab.shape} differ')
viewer = napari.Viewer(title=f'curate: {out_path}')
viewer.add_image(img, name='reference', scale=scale, colormap='gray',
                 contrast_limits=tuple(np.percentile(img, [1, 99.7])))
cells = viewer.add_labels(lab, name='cells', scale=scale, opacity=0.4)
cells.selected_label = int(lab.max()) + 1


def _save(_viewer=None):
    tifffile.imwrite(out_path, cells.data.astype(np.uint16), compression='zlib')
    print(f'saved {int(np.count_nonzero(np.unique(cells.data)))} labels -> {out_path}', flush=True)


viewer.bind_key('Control-Shift-S', _save)
napari.run()
_save()
