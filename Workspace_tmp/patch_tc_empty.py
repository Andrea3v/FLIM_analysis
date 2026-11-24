"""Step 17 (code-17): no crash when no cell passes the summary selection (2026-11-23).

With a dim recording (Slice3 sucrose: 19 kept cells, median 47 photons per cell and timepoint) no cell is usable at
every timepoint (Step 2: >= cell_min_photons and >= cell_min_fov at each), the summary selection is empty and
np.nanpercentile on the empty (0 x n_timepoints) table returned the wrong shape -> matplotlib ValueError.
Now: print why, keep the per-cell CSV, skip summary / pooled fit / plot.

Idempotent. Usage: python Workspace_tmp/patch_tc_empty.py [notebook]
"""
import json, os, sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB = os.path.join(HERE, 'AnalyzeTCSPC_XYTZTime.ipynb')

ANCHOR = """    _sel_txt = f'fov_frac >= {tc_min_fov}' + (', cells usable at every timepoint' if cells_time_ok is not None else '')
"""
GUARD = """    if _in_fov.empty:
        _med = df_tc.loc[df_tc['fov_frac'] >= tc_min_fov, 'n_photons'].median()
        print(f'\\nNo cell for the summary ({_sel_txt}): '
              + (f'{len(cells_time_ok)} cells usable at every timepoint (Step 2: >= {cell_min_photons} photons and '
                 f'>= {cell_min_fov} of the cell recorded, at each timepoint)' if cells_time_ok is not None
                 else 'no row in the field of view')
              + f'; median {_med:.0f} photons per cell and timepoint in the field of view.\\n'
              f'The per-cell table is saved; summary, pooled fit and plot are skipped. For a dim recording lower '
              f'cell_min_photons (Step 2) and min_photons (Step 14).')
    else:
"""


def main(nb_path=NB):
    nb = json.load(open(nb_path, encoding='utf-8'))
    c = next(c for c in nb['cells'] if c.get('id') == 'code-17')
    s = ''.join(c['source'])
    if 'No cell for the summary' in s:
        print(f'{os.path.basename(nb_path)}: already patched')
        return
    if s.count(ANCHOR) != 1 or 'fit_per_cell' not in s:
        sys.exit('code-17: anchor not found once (or patch_fit_per_cell.py not applied yet), nothing written')
    head, tail = s.split(ANCHOR)
    tail = ''.join(('    ' + l) if l.strip() else l for l in tail.splitlines(keepends=True))
    c['source'] = (head + ANCHOR + GUARD + tail).splitlines(keepends=True)
    with open(nb_path, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(nb, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(f'{os.path.basename(nb_path)}: code-17 patched')


if __name__ == '__main__':
    main(*sys.argv[1:])
