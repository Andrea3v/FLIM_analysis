import json, sys
sys.stdout.reconfigure(encoding='utf-8')

path = r'X:\FLIM_2026\AnalyzeTCSPC.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

c = nb['cells'][33]
assert c['id'] == '6e4891ec', f"Wrong cell: {c['id']}"
src = c['source'] if isinstance(c['source'], str) else ''.join(c['source'])

old = (
    "    plt.savefig(_fig_cp_phas, facecolor='white', bbox_inches='tight')\n"
    "    plt.show()\n"
    "    print(f'Saved: {_fig_cp_phas}')"
)
new = (
    "    plt.savefig(_fig_cp_phas, facecolor='white', bbox_inches='tight')\n"
    "    plt.savefig(_fig_cp_phas.replace('.png', '.svg'), facecolor='white', bbox_inches='tight')\n"
    "    plt.show()\n"
    "    print(f'Saved: {_fig_cp_phas}  +  .svg')"
)
assert old in src, 'anchor not found'
src = src.replace(old, new)

c['source'] = src
with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print('done')
