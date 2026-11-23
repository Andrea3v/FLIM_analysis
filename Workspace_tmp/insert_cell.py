import json, sys
sys.stdout.reconfigure(encoding='utf-8')

path = r'X:\FLIM_2026\AnalyzeTCSPC.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

new_source = (
    '# ── Cell to inspect: change this number ──────────────────────────────────────\n'
    'CELL_NUM = 1          # 1-based cell ID (1 ... n_cells)\n'
    '\n'
    '# ── Reload flim_helpers so any edits to the module take effect ───────────\n'
    'import flim_helpers as _fh_mod; importlib.reload(_fh_mod)\n'
    'from flim_helpers import Fit_decay\n'
    '\n'
    '# ── Load saved decays + models from disk ───────────────────────────\n'
    'if not os.path.exists(_fits_path):\n'
    '    raise FileNotFoundError(\n'
    '        f"Fits file not found -- run the per-cell fitting cell first.\\n{_fits_path}")\n'
    '\n'
    '_fits       = np.load(_fits_path, allow_pickle=True)\n'
    '_t          = _fits["tau_axis"]         # (nbin,) time axis in ns\n'
    '_irf        = _fits["irf_aligned"]      # (nbin,) aligned IRF\n'
    '_all_decays = _fits["decays"]           # (n_cells, nbin) raw TCSPC\n'
    '_all_models = _fits["models"]           # (n_cells, nbin) fitted model\n'
    '_t_stop_val = _fits["t_stop_ns"]\n'
    '_t_stop     = float(_t_stop_val) if np.isfinite(float(_t_stop_val)) else None\n'
    '\n'
    '# ── Reload per-cell CSV for metadata ──────────────────────────────\n'
    'df_cells = pd.read_csv(csv_out)\n'
    '\n'
    '# ── Validate requested cell ────────────────────────────────────────\n'
    '_nc = _all_decays.shape[0]\n'
    'if not (1 <= CELL_NUM <= _nc):\n'
    '    raise ValueError(f"CELL_NUM={CELL_NUM} out of range -- dataset has {_nc} cells (1-{_nc})")\n'
    '\n'
    '_raw = _all_decays[CELL_NUM - 1].astype(float)\n'
    '_mod = _all_models[CELL_NUM - 1].astype(float)\n'
    '_row = df_cells[df_cells["cell_id"] == CELL_NUM].iloc[0]\n'
    '\n'
    '# ── Normalise to peak of raw decay ─────────────────────────────────\n'
    '_pk    = _raw.max() if _raw.max() > 0 else 1.0\n'
    '_raw_n = _raw / _pk\n'
    '_mod_n = _mod / _pk\n'
    '_irf_n = _irf / _irf.max()\n'
    '\n'
    '# ── Plot ────────────────────────────────────────────────────────────────────\n'
    'fig, ax = plt.subplots(figsize=(7, 4))\n'
    'ax.scatter(_t, _raw_n, s=8,  color="steelblue", alpha=0.75, zorder=2, label="raw data")\n'
    'ax.plot   (_t, _mod_n, "-",  color="tomato",    linewidth=1.8, zorder=3, label="model fit")\n'
    'ax.plot   (_t, _irf_n, "--", color="0.45",      linewidth=0.8,           label="IRF (aligned)")\n'
    'if _t_stop is not None:\n'
    '    ax.axvline(_t_stop, color="red", linestyle=":", linewidth=1.0,\n'
    '               label=f"t_stop = {_t_stop:.2f} ns")\n'
    '\n'
    'ax.set_yscale("log")\n'
    'ax.set_xlim(_t[0], _t[-1])\n'
    'ax.set_ylim(5e-3, 1.5)\n'
    'ax.set_xlabel("time / ns")\n'
    'ax.set_ylabel("intensity  (norm. to peak)")\n'
    'ax.set_title(\n'
    '    f"{FName}  --  cell {CELL_NUM}  |  "\n'
    '    f"aw_LT = {_row[\'aw_LT\']:.3f} ns   "\n'
    '    f"chi2 = {_row[\'chi2\']:.2f}   "\n'
    '    f"n_photons = {int(_row[\'n_photons\']):,}"\n'
    ')\n'
    'ax.legend(fontsize=9, framealpha=0.6)\n'
    'plt.tight_layout()\n'
    'plt.show()\n'
)

new_cell = {
    'cell_type': 'code',
    'execution_count': None,
    'id': 'decay-inspect',
    'metadata': {},
    'outputs': [],
    'source': new_source
}

insert_idx = next(i for i, c in enumerate(nb['cells']) if c.get('id') == 'b3fb5433') + 1
nb['cells'].insert(insert_idx, new_cell)

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f'Inserted cell at index {insert_idx}')
