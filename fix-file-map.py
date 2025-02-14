# %%
import pickle

fm = pickle.load(open('file_map.pkl', 'rb'))
# %%
print(fm)
# %%
from pathlib import Path

file_table = []
k: Path
v: Path
for k, vs in fm.items():
    for v in vs:
        file_table.append([k.as_posix(), v.as_posix()])

print(file_table)
# %%

with open('file_table.tsv', 'w') as f:
    for row in file_table:
        f.write('\t'.join(row) + '\n')
# %%
