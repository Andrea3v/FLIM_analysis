"""AnalyzeTCSPC_XYTZTile.ipynb Step 2: cp_model_path (a model fine-tuned in Cellpose_FineTune.ipynb)."""
import json

NB = r'X:\FLIM_2026\AnalyzeTCSPC_XYTZTile.ipynb'
raw = open(NB, encoding='utf-8', newline='').read()
indent = len(raw.splitlines()[1]) - len(raw.splitlines()[1].lstrip(' '))
nb = json.loads(raw)
cells = {c.get('id'): c for c in nb['cells']}
assert 'cp_model_path' not in ''.join(cells['a5874f7b']['source']), 'already patched'


def sub(cid, old, new):
    c = cells[cid]
    s = ''.join(c['source'])
    assert s.count(old) == 1, (cid, old[:60], s.count(old))
    c['source'] = s.replace(old, new).splitlines(keepends=True)
    if c['cell_type'] == 'code':
        c['outputs'], c['execution_count'] = [], None


sub('c001adff', "- **`use_subject_reference = True`** skips Cellpose",
    "- **Fine-tuned model** `cp_model_path` (from `Cellpose_FineTune.ipynb`; None = generic `cpsam`): `cp_diameter_um` is then\n"
    "  taken from the model (its training scale).\n"
    "- **`use_subject_reference = True`** skips Cellpose")
sub('a5874f7b', '''cp_gpu              = True
_rerun_cp           = False        # True: re-run Cellpose even if masks with these settings exist''',
    '''cp_model_path       = None         # fine-tuned model (Cellpose_FineTune.ipynb); None = generic cpsam
cp_gpu              = True
_rerun_cp           = False        # True: re-run Cellpose even if masks with these settings exist''')
sub('a5874f7b', '''_voxel_sp    = (z_step_um, xy_pixel_um, xy_pixel_um)
if seg_source == 'auto':''',
    '''_voxel_sp    = (z_step_um, xy_pixel_um, xy_pixel_um)
if cp_model_path:   # the model was trained at one scale: use its diameter
    _ft_meta = json.load(open(cp_model_path + '.json'))
    if _ft_meta['diameter_um'] != cp_diameter_um:
        print(f'cp_diameter_um {cp_diameter_um} -> {_ft_meta["diameter_um"]} µm (training scale of '
              f'{os.path.basename(cp_model_path)})')
        cp_diameter_um = _ft_meta['diameter_um']
if seg_source == 'auto':''')
sub('a5874f7b', '''                    stitch_threshold=cp_stitch_threshold, min_volume_um3=cp_min_volume_um3)

_filt_path''', '''                    stitch_threshold=cp_stitch_threshold, min_volume_um3=cp_min_volume_um3,
                    model=[cp_model_path, os.path.getmtime(cp_model_path)] if cp_model_path else None)

_filt_path''')
sub('a5874f7b', '''        print(f'Running Cellpose ({cp_mode}) on {_seg_name}  d={cp_diameter_um} µm  cellprob={cp_cellprob_thresh}  '
              f'input={cp_input}  gpu={cp_gpu}...', flush=True)''',
    '''        print(f'Running Cellpose ({cp_mode}, {os.path.basename(cp_model_path) if cp_model_path else "cpsam"}) on '
              f'{_seg_name}  d={cp_diameter_um} µm  cellprob={cp_cellprob_thresh}  input={cp_input}  gpu={cp_gpu}...',
              flush=True)''')
sub('a5874f7b', '''                                       stitch_threshold=cp_stitch_threshold, min_volume_um3=cp_min_volume_um3,
                                       gpu=cp_gpu)''',
    '''                                       stitch_threshold=cp_stitch_threshold, min_volume_um3=cp_min_volume_um3,
                                       gpu=cp_gpu, model_path=cp_model_path)''')

with open(NB, 'w', encoding='utf-8', newline='') as f:
    f.write(json.dumps(nb, indent=indent, ensure_ascii=False) + ('\n' if raw.endswith('\n') else ''))
print('patched c001adff, a5874f7b')
