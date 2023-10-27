import matplotlib.pyplot as plt
import json
import numpy as np
import pickle
import os
import sys
import glob
import seaborn
import matplotlib

def parse_len_from_pdb(pdb_name):
    return int(pdb_name.split('_')[0][3:])

window = 21
# Load data
target_dir = sys.argv[1]
is_allatom = sys.argv[2].lower() == 'true'
suffix = ""
if len(sys.argv) > 3:
    suffix = "_single"
model_type = "allatom" if is_allatom else "backbone"
print(model_type)
all_metrics = json.loads(open(f'analysis/decent_{model_type}_proteins_pdbs/{target_dir}/all_metrics_new.json', 'r').read())
output_dir = f"analysis/decent_{model_type}_proteins_pdbs/{target_dir}/plots"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

matplotlib.rc('font', size=12)

# Quality (RMSD) vs length
data = [(parse_len_from_pdb(k), v[f'bb_sc_rmsd_best{suffix}'], v[f'bb_sc_plddt_best{suffix}']) for k, v in all_metrics.items()]
lens = [l for l, r, p in data]
data = np.array(data).T
sorted_lens = np.array(sorted(set(lens)))
proportions = []
for l in sorted_lens:
    rmsds = data[1][data[0] == l]
    proportions.append((rmsds < 2).mean())
proportions = ([proportions[0]] * (window // 2)) + proportions + ([proportions[-1]] * (window//2))
proportions = np.convolve(proportions, np.ones(window), 'valid') / window
fig, ax1 = plt.subplots(figsize=(5, 3))
ax1.set_xlabel('protein length')
sc = ax1.scatter(data[0], data[1], c=data[2], s=2, vmin=0.2, vmax=1.0)
fig.colorbar(sc, label='pLDDT', pad=0.2)
ax1.axhline(y=2, color='r', linestyle='--', lw=1)
ax1.set_ylabel('scRMSD')
ax1.set_ylim(0, 25)
ax2 = ax1.twinx()
ax2.set_ylabel(r'% samples with scRMSD < 2')
ax2.yaxis.label.set_color('tab:orange')
ax2.plot(sorted_lens, proportions, color='tab:orange')
ax2.set_ylim(-0.01, 1.02)
fig.tight_layout()
plt.savefig(f'{output_dir}/sc_plot{suffix}.pdf')


# # Quality (TM) vs length
data = [(parse_len_from_pdb(k), 1 - v[f'bb_sc_tmscore_best{suffix}'], v[f'bb_sc_plddt_best{suffix}']) for k, v in all_metrics.items()]
lens = [l for l, r, p in data]
data = np.array(data).T
sorted_lens = np.array(sorted(set(lens)))
proportions = []
for l in sorted_lens:
    tms = data[1][data[0] == l]
    proportions.append((tms > 0.5).mean())
proportions = ([proportions[0]] * (window // 2)) + proportions + ([proportions[-1]] * (window//2))
proportions = np.convolve(proportions, np.ones(window), 'valid') / window
fig, ax1 = plt.subplots(figsize=(5, 3))
ax1.set_xlabel('protein length')
sc = ax1.scatter(data[0], data[1], c=data[2], s=2, vmin=0.2, vmax=1.0)
fig.colorbar(sc, label='pLDDT', pad=0.2)
ax1.axhline(y=0.5, color='r', linestyle='--', lw=1)
ax1.set_ylabel('TM')
ax1.set_ylim(0, 1)
ax2 = ax1.twinx()
ax2.set_ylabel(r'% samples with TM > 0.5')
ax2.yaxis.label.set_color('tab:orange')
ax2.plot(sorted_lens, proportions, color='tab:orange')
ax2.set_ylim(-0.01, 1.02)
fig.tight_layout()
plt.savefig(f'{output_dir}/sc_plot_tm{suffix}.pdf')


# alpha beta
#ss_file = open(f"../protpardelle_paper/samples/{target_dir}/sequence_structure.json")
#ss_info = json.loads(ss_file.read())
#ss_data = [(int(os.path.basename(k).split('_')[0][3:]), v['sample_pct_alpha'], v['sample_pct_beta']) for k, v in ss_info.items()]
#plt.clf()
#fig, ax1 = plt.subplots(figsize=(4, 3))
#ss_data = np.array(ss_data).T

#plt.scatter(ss_data[1], ss_data[2], c=ss_data[0], s=4, alpha=0.5, cmap='plasma')
#plt.colorbar(label='len')
#plt.ylabel(r'% sheet')
#plt.xlabel(r'% helix')
#plt.ylim(0, 1)
#plt.xlim(0, 1)
#fig.tight_layout()
#plt.savefig(f'{output_dir}/structure.pdf')


# # Scatter nnTM vs scrmsd
data = [(int(k.split('_')[0][3:]), v['bb_sc_rmsd_best'], v['tm']['tm_score']) for k, v in all_metrics.items() if 'tm' in v]
data = np.array(data).T
plt.clf()
plt.scatter(data[2], data[1], c=data[0], s=1, cmap="plasma")
plt.xlabel('nnTM')
plt.ylabel('scRMSD')
plt.xlim(0.3, 1)
plt.ylim(0, 5)
plt.colorbar(label='len')
plt.tight_layout()
plt.savefig(f'{output_dir}/nn_metrics.pdf')


# Bond lengths histogram
bond_lengths = json.loads(open(f'../protpardelle_paper/samples/{target_dir}/bond_metrics/bond_length_metrics_samp.json', 'r').read())['bond_lengths']
bond_lengths_data = json.loads(open(f'eval_scripts/bond_metrics_data/bond_length_metrics.json', 'r').read())['bond_lengths']

bond_lengths_data = [x for x in bond_lengths_data if x != 0]
plt.clf()
plt.xlabel('bond length')
plt.hist(bond_lengths, bins=np.arange(0.9, 2, 0.05), alpha=0.5, weights=np.ones_like(bond_lengths) / len(bond_lengths), label='samples')
plt.hist(bond_lengths_data, bins=np.arange(0.9, 2, 0.05), alpha=0.5, weights=np.ones_like(bond_lengths_data) / len(bond_lengths_data), label='data')
plt.ylabel('frequency')
plt.legend()
plt.tight_layout()
plt.savefig(f'{output_dir}/bond_lengths_overlaid.pdf')


# Bond angles histogram
bond_angles = json.loads(open(f'../protpardelle_paper/samples/{target_dir}/bond_metrics/bond_angle_metrics_samp.json', 'r').read())['bond_angles']
bond_angles_data = json.loads(open(f'eval_scripts/bond_metrics_data/bond_angle_metrics.json', 'r').read())['bond_angles']

plt.clf()
plt.xlabel('bond angle')
plt.hist(bond_angles, bins=np.arange(90, 150, 2), alpha=0.5, weights=np.ones_like(bond_angles) / len(bond_angles), label='samples')
plt.hist(bond_angles_data, bins=np.arange(90, 150, 2), alpha=0.5, weights=np.ones_like(bond_angles_data) / len(bond_angles_data), label='data')
plt.ylabel('frequency')
plt.legend()
plt.tight_layout()
plt.savefig(f'{output_dir}/bond_angles_overlaid.pdf')

# Chi angles histogram
chi_angles = json.loads(open(f'../protpardelle_paper/samples/{target_dir}/bond_metrics/chi_angle_metrics_samp.json', 'r').read())['chi_angles']
chi_angles_data = json.loads(open(f'eval_scripts/bond_metrics_data/chi_angle_metrics.json', 'r').read())['chi_angles']

plt.clf()
plt.xlabel('chi angle')
plt.hist(chi_angles, bins=np.arange(0, 180, 2), alpha=0.5, weights=np.ones_like(chi_angles) / len(chi_angles), label='samples')
plt.hist(chi_angles_data, bins=np.arange(0, 180, 2), alpha=0.5, weights=np.ones_like(chi_angles_data) / len(chi_angles_data), label='data')
plt.ylabel('frequency')
plt.legend()
plt.tight_layout()
plt.savefig(f'{output_dir}/chi_angles_overlaid.pdf')
