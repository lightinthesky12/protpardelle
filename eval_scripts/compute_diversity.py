import sys
import os
import json
from collections import defaultdict
import glob

is_allatom = sys.argv[2].lower() == 'true'

result_type = "allatom" if is_allatom else "backbone"


target_dir = sys.argv[1]
tm_dir = f"{target_dir}/tms/"
scores = defaultdict(lambda: {})
tm_files = sorted(glob.glob(f"{tm_dir}/*_samp*_self.json"))

for file in tm_files:
    f = open(file)
    name = file.split("/")[-1].replace("tm_scores_", "").replace("_self.json", "")
    x = f.read()
    curr_scores = json.loads(x)
    for score in curr_scores:
        scores[name][score['pdb']] = 1 - score['tm_score']


pdbs = list(scores.keys())
pdbs.sort()

result = [[scores[pdbs[i]][pdbs[j]] if i != j else 0 for i in range(0, len(pdbs))] for j in range(0, len(pdbs))]
for i in range(0, len(pdbs)):
    for j in range(0, len(pdbs)):
        if result[i][j] != result[j][i]:
            min_dist = min(result[i][j], result[j][i])
            result[i][j] = min_dist
            result[j][i] = min_dist

import numpy as np

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform



mat = np.array(result)
dists = squareform(mat)
linkage_matrix = linkage(dists, "single")
clusters = fcluster(linkage_matrix, criterion="distance", t=0.4)
cluster_map = defaultdict(lambda: [])
for idx, cluster in enumerate(clusters):
    cluster_map[int(cluster)].append(pdbs[idx])

print(target_dir, "DIV", len(cluster_map)/len(pdbs), len(cluster_map), len(pdbs))
with open(f'{target_dir}/diversity.json', 'w') as f:
    f.write(json.dumps(cluster_map))


sc_scores = sorted(glob.glob(f"{target_dir}/sc_metrics/{result_type}*_results.json"))
count = 0
total_count = 0
for sc_score in sc_scores:
    f = open(sc_score)
    x = f.read()
    y = json.loads(x)
    for pdb in y:
        total_count += 1
        z = y[pdb]
        if z['j_sc_rmsd_best'] < 2:
            count += 1

print(target_dir, "SC", count/total_count, count, total_count)
