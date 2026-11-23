import json, sys
sys.stdout.reconfigure(encoding='utf-8')

path = r'X:\FLIM_2026\AnalyzeTCSPC.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

c = nb['cells'][30]
assert c['id'] == 'b3fb5433', f"Wrong cell: {c['id']}"
src = c['source'] if isinstance(c['source'], str) else ''.join(c['source'])

# ── Replace the IRF alignment block ──────────────────────────────────────────
old = (
    "# ── IRF alignment to bulk data peak ──────────────────────────────────────────\n"
    "_peak_bulk = int(np.argmax(bulk_decay))\n"
    "_peak_irf  = int(np.argmax(IRF2))\n"
    "_shift_irf = _peak_bulk - _peak_irf\n"
    "IRF2_fit   = np.roll(IRF2, _shift_irf)\n"
    "print(f'Bulk peak bin {_peak_bulk}  |  IRF shifted {_shift_irf:+d} bins')\n"
)
new = (
    "# ── Shift bulk decay + all cell decays to 8% target (consistent with Step 5) ──\n"
    "_peak_bulk       = int(np.argmax(bulk_decay))\n"
    "_peak_irf        = int(np.argmax(IRF2))\n"
    "_shift_to_target = target - _peak_bulk          # bins to move peak to target\n"
    "bulk_decay = np.roll(bulk_decay, _shift_to_target)\n"
    "_cell_mat  = np.roll(_cell_mat,  _shift_to_target, axis=1)\n"
    "IRF2_fit   = np.roll(IRF2, target - _peak_irf)  # IRF also goes to target\n"
    "print(f'Bulk peak bin {_peak_bulk}  →  shifted {_shift_to_target:+d} bins  →  target bin {target}  |  IRF shifted to target')\n"
)
assert old in src, 'IRF alignment anchor not found'
src = src.replace(old, new)

# ── Update QC plot: axvline now points at target, not _peak_bulk ─────────────
old2 = "ax_bulk.axvline(tau[_peak_bulk], color='orange', lw=0.8, ls=':', label=f'peak  bin {_peak_bulk}')\n"
new2 = "ax_bulk.axvline(tau[target], color='orange', lw=0.8, ls=':', label=f'peak  bin {target}  (8% target)')\n"
assert old2 in src, 'axvline anchor not found'
src = src.replace(old2, new2)

c['source'] = src
with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print('done')
