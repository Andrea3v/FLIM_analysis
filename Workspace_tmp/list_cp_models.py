from cellpose import models
import json, os, inspect, re

print("MODEL_DIR:", models.MODEL_DIR)

if os.path.exists(models.MODEL_LIST_PATH):
    with open(models.MODEL_LIST_PATH) as f:
        print("MODEL_LIST_PATH:", json.load(f))
else:
    print("MODEL_LIST_PATH not on disk yet:", models.MODEL_LIST_PATH)

src = inspect.getsource(models)
found = re.findall(r'["\']([a-z][a-z0-9_]*(?:cp3|cp2|_cp\d)?)["\']', src)
# keep only plausible model names
candidates = sorted(set(m for m in found
                        if any(m.startswith(p) for p in
                               ('cyto','nuclei','tissue','live','yeast',
                                'bact','deep','cpsam','CPx'))))
print("Candidate model names in source:", candidates)
