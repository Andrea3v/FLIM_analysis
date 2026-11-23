import json, sys
sys.stdout.reconfigure(encoding='utf-8')

path = r'X:\FLIM_2026\AnalyzeTCSPC.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Check datetime in imports
c1 = nb['cells'][1]
src1 = c1['source'] if isinstance(c1['source'], str) else ''.join(c1['source'])
print("datetime import:", "OK" if "from datetime import datetime" in src1 else "MISSING")

# Check code-03
c4 = nb['cells'][4]
src4 = c4['source'] if isinstance(c4['source'], str) else ''.join(c4['source'])
print("hardcoded newFolder removed:", "OK" if "X:/FLIM_2026/results" not in src4 else "STILL THERE")
print("timestamped newFolder:", "OK" if "_ts_FLIM_Analysis" in src4 or "_FLIM_Analysis" in src4 else "MISSING")
# print relevant snippet
for line in src4.split('\n'):
    if 'FName' in line or 'newFolder' in line or 'makedirs' in line or '_ts' in line:
        print(' >', line)

print()
# Check step-13
c28 = nb['cells'][28]
src28 = c28['source'] if isinstance(c28['source'], str) else ''.join(c28['source'])
print("z_decay_plots dir:", "OK" if "_zdecay_dir" in src28 else "MISSING")
print("FLIM_RGB_planes dir:", "OK" if "_flimrgb_dir" in src28 else "MISSING")
print("decay save uses _zdecay_dir:", "OK" if "_zdecay_dir" in src28 and "_fig_d.savefig" in src28 else "MISSING")
print("png_base uses _flimrgb_dir:", "OK" if "_flimrgb_dir" in src28 and "_png_base" in src28 else "MISSING")
# print relevant snippets
for line in src28.split('\n'):
    if any(k in line for k in ['_zdecay_dir', '_flimrgb_dir', '_png_base', 'savefig', 'makedirs']):
        print(' >', line)
