import glob
import json
import os
import shutil
import sys
import random

# consolidate_metrics.py goldensun_1000_050_400
prefix = sys.argv[1]

is_allatom = sys.argv[2].lower() == 'true'

result_type = "allatom" if is_allatom else "backbone"

dest_dir = f"analysis/decent_{result_type}_proteins_pdbs/{prefix}"
if not os.path.exists(dest_dir):
    os.mkdir(dest_dir)

pdb_good_dest_dir = f"{dest_dir}/pdbs_good"
if not os.path.exists(pdb_good_dest_dir):
    os.mkdir(pdb_good_dest_dir)
pdb_random_dest_dir = f"{dest_dir}/pdbs_random"
if not os.path.exists(pdb_random_dest_dir):
    os.mkdir(pdb_random_dest_dir)


paths_to_pdbs = ["../protpardelle_paper/samples/" + x for x in os.listdir("../protpardelle_paper/samples") if x.startswith(prefix)]
all_results = {}

def get_pdb_name(path):
    return os.path.basename(path)

for path_to_pdbs in paths_to_pdbs:
    results = {}

    sc_scores = sorted(glob.glob(f"{path_to_pdbs}/sc_metrics/{result_type}*_results.json"))
    for sc_score in sc_scores:
        f = open(sc_score)
        x = f.read()
        y = json.loads(x)
        for pdb in y:
            z = y[pdb]
            z = {f"{k.replace('j_', 'bb_')}": v for k, v in z.items()}
            y[pdb] = z
        results.update(y)

    results = {f"{get_pdb_name(k)}": v for k, v in results.items()}

    sc_scores = sorted(glob.glob(f"{path_to_pdbs}/sc_metrics_1/{result_type}*_results.json"))
    for sc_score in sc_scores:
        f = open(sc_score)
        x = f.read()
        y = json.loads(x)
        for pdb in y:
            z = y[pdb]
            z = {f"{k.replace('j_', 'bb_') + '_single'}": v for k, v in z.items()}
            results[get_pdb_name(pdb)].update(z)

    tms = sorted(glob.glob(f"{path_to_pdbs}/foldseek/parsed_*.json"))
    for tm in tms:
        f = open(tm)
        x = f.read().strip()
        y = json.loads(x) if x else []
        pdb_name = os.path.basename(tm).replace("parsed_", "").replace("pred", "samp").replace(".json", ".pdb")
        if pdb_name in results:
            results[pdb_name]['tm'] = y[0] if len(y) >= 1 else {'tm_score': 0}


    tms_self = sorted(glob.glob(f"{path_to_pdbs}/foldseek_self/parsed_*.json"))
    for tm in tms_self:
        f = open(tm)
        x = f.read().strip()
        y = json.loads(x) if x else []
        pdb_name = os.path.basename(tm).replace("parsed_", "").replace("pred", "samp").replace(".json", ".pdb")
        if pdb_name in results:
            results[pdb_name]['tm_self'] = y

    if is_allatom:
        bond_lengths = json.loads(open(f'{path_to_pdbs}/bond_metrics/bond_length_metrics_samp.json', 'r').read())['per_pdb_bond_length']
        for bond_length in bond_lengths:
            pdb_name = os.path.basename(bond_length)

            print(pdb_name)
            print(results)
            results[pdb_name]['bond_length'] = bond_lengths[bond_length]

    # copy good proteins to folder for visual inspection
    MAX_NUM = 20
    filtered_results = [(pdb, result) for pdb, result in results.items() if result['bb_sc_rmsd_best'] < 2]
    sorted_results = [x for x in filtered_results if ('tm' in x[1]) and (0 < x[1]['tm']['tm_score'])]
    sorted_results.sort(key=lambda x: x[1]['tm']['tm_score'])
    for pdb, result in sorted_results[0:MAX_NUM]:
        src_pdb = f"{path_to_pdbs}/samples/{pdb}"
        scrmsd =  "%.3f" % result['bb_sc_rmsd_best']
        plddt =  "%.3f" % result['bb_sc_plddt_best']
        nntm = "%.3f" % result['tm']['tm_score']
        bond_length = result.get('bond_length')
        bond_length = "%.3f" % bond_length if bond_length else None
        info_name = f"{pdb.replace('.pdb', '')}_scrmsd_{scrmsd}_plddt_{plddt}_nntm_{nntm}_bondlength_{bond_length}.pdb"
        dest_pdb = f"{pdb_good_dest_dir}/{info_name}"
        shutil.copy(src_pdb, dest_pdb)

        pdb_base = pdb.split(".")[0]
        print(pdb, result['bb_sc_rmsd_best'], result['tm']['tm_score'])
        if 'pdb' in result['tm']:
            dataset_pdb = result['tm']['pdb']
            print(dataset_pdb)
            src_nn_pdb = f"/scratch/users/alexechu/datasets/ingraham_cath_dataset/pdb_store/{dataset_pdb}"
            dest_nn_pdb = f"{pdb_good_dest_dir}/{pdb_base}_nn.pdb"
            shutil.copy(src_nn_pdb, dest_nn_pdb)

        if is_allatom:
            src_pred_pdb = f"{path_to_pdbs}/preds_allatom/{pdb_base.replace('samp', 'samppred')}.pdb"
        else:
            src_pred_pdb = f"{path_to_pdbs}/preds_backbone/{pdb_base.replace('samp', 'pred')}.pdb"
        dest_pred_pdb = f"{pdb_good_dest_dir}/{pdb_base}_pred.pdb"
        shutil.copy(src_pred_pdb, dest_pred_pdb)

    # select random proteins for visual inspection
    random.seed(2)
    random_results = random.choices(list(results.items()), k=40)
    max_count = 0
    for pdb, result in random_results:
        src_pdb = f"{path_to_pdbs}/samples/{pdb}"
        scrmsd =  "%.3f" % result['bb_sc_rmsd_best']
        plddt =  "%.3f" % result['bb_sc_plddt_best']
        nntm = result.get('tm', {}).get('tm_score')
        nntm = "%.3f" % nntm if nntm else None
        bond_length = result.get('bond_length')
        bond_length = "%.3f" % bond_length if bond_length else None
        info_name = f"{pdb.replace('.pdb', '')}_scrmsd_{scrmsd}_plddt_{plddt}_nntm_{nntm}_bondlength_{bond_length}.pdb"
        dest_pdb = f"{pdb_random_dest_dir}/{info_name}"
        shutil.copy(src_pdb, dest_pdb)

        pdb_base = pdb.split(".")[0]
        if 'tm' in result and 'pdb' in result['tm']:
            dataset_pdb = result['tm']['pdb']
            print(dataset_pdb)
            src_nn_pdb = f"/scratch/users/alexechu/datasets/ingraham_cath_dataset/pdb_store/{dataset_pdb}"
            dest_nn_pdb = f"{pdb_random_dest_dir}/{pdb_base}_nn.pdb"
            shutil.copy(src_nn_pdb, dest_nn_pdb)

        if is_allatom:
            src_pred_pdb = f"{path_to_pdbs}/preds_allatom/{pdb_base.replace('samp', 'samppred')}.pdb"
        else:
            src_pred_pdb = f"{path_to_pdbs}/preds_backbone/{pdb_base.replace('samp', 'pred')}.pdb"
        dest_pred_pdb = f"{pdb_random_dest_dir}/{pdb_base}_pred.pdb"
        shutil.copy(src_pred_pdb, dest_pred_pdb)
        max_count += 1
        if max_count > 25:
            break
    all_results.update(results)

all_metrics_file = open(f"{dest_dir}/all_metrics_new.json", "w")
json.dump(all_results, all_metrics_file)
