import os
import shlex
import subprocess
import sys
import torch
import argparse
import glob
import json
from uuid import uuid4
import statistics
import shutil
from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP
import math
import signal
from core import utils
import evaluation
from collections import defaultdict
from joblib import Parallel, delayed, cpu_count


PATH_TO_TMALIGN = "/home/groups/possu/TMalign"

class Manager(object):
    def __init__(self):

        self.parser = argparse.ArgumentParser()

        self.parser.add_argument("--samples", type=str, required=True, help="Dataset of samples to analyze")
        self.parser.add_argument("--missing", type=str, default='temp_missing_tm.txt', help="Dataset of samples to analyze")
        self.parser.add_argument("--task", type=str, required=True, help="Which task to run")
        self.parser.add_argument("--dataset", type=str, default='/scratch/users/alexechu/datasets/ingraham_cath_dataset',help="CATH dataset of coordinate files")
        self.parser.add_argument("--offset", type=int, default=0, help="Which pdb to work on")
        self.parser.add_argument("--limit", type=int, default=10, help="Which pdb to work on")

    def add_argument(self, *args, **kwargs):
        self.parser.add_argument(*args, **kwargs)

    def parse_args(self):
        self.args = self.parser.parse_args()

        return self.args


def quick_tmalign(
    pdb1_path, pdb2_path, tmscore_type="avg"
):
    success = False
    tries = 3
    while (not success) and (tries > 0):
        tmout = None
        try:
            cmd = f"{PATH_TO_TMALIGN} {pdb1_path} {pdb2_path}"
            outputs = subprocess.run(shlex.split(cmd), capture_output=True, text=True)

            # Get RMSD and TM scores
            tmout = outputs.stdout.split("\n")
            rmsd = float(tmout[16].split()[4][:-1])
            tmscore1 = float(tmout[17].split()[1])
            tmscore2 = float(tmout[18].split()[1])
            success = True
            tries = tries - 1
        except:
            print(tmout)

    if tmscore_type == "avg":
        tmscore = (tmscore1 + tmscore2) / 2
    elif tmscore_type == "1" or tmscore_type == "query":
        tmscore = tmscore1
    elif tmscore_type == "2":
        tmscore = tmscore2
    elif tmscore_type == "both":
        tmscore = (tmscore1, tmscore2)
    elif tmscore_type == "max":
        tmscore = max(tmscore1, tmscore2)
    result = {"pdb": os.path.basename(pdb2_path).strip('.pdb'), "rmsd": rmsd, "tm_score": tmscore, "tm1": tmscore1, "tm2": tmscore2, "maxtm": max(tmscore1, tmscore2)}
    print(result)
    return result

def generate_tm_pdb(pdb1, target_pdbs, output_path, metric, include_self=False):
    """Generate TM score and output highest k based on average TM score"""
    print(cpu_count())
    print(pdb1)
    tms = Parallel(n_jobs=cpu_count(), backend='multiprocessing')(delayed(quick_tmalign)(pdb1, pdb2) for pdb2 in target_pdbs if (include_self or (pdb1 != pdb2)))
    tms.sort(key=lambda tup: tup['tm_score'], reverse=True)

    tm_name = os.path.basename(pdb1).strip('.pdb')
    json.dump(tms, open(f"{output_path}/tm_scores_{tm_name}_{metric}.json", 'w'))

def handler(signum, frame):
    raise Exception("Timeout")

def parse_struct(pdb):
    """Parse the PDB structure into a BioPython-compatible object"""
    return PDBParser(QUIET=True).get_structure('structure', pdb)[0]


def get_dssp_string(pdb, structure):
    """Returns output from DSSP"""
    dssp = DSSP(structure, pdb, dssp="mkdssp")
    dssp_string = "".join([dssp[k][2] for k in dssp.keys()])
    return dssp_string


def pool_dssp_symbols(dssp_string, newchar=None, chars=["-", "T", "S", "C", " "]):
    """Replaces all instances of chars with newchar. DSSP chars are helix=GHI, strand=EB, loop=- TSC"""
    if newchar is None:
        newchar = chars[0]
    string_out = dssp_string
    for c in chars:
        string_out = string_out.replace(c, newchar)
    return string_out


def get_3state_dssp(pdb, structure):
    """Returns Helix/Coild/Loop String"""
    dssp_string = get_dssp_string(pdb, structure)

    if dssp_string is not None:
        dssp_string = pool_dssp_symbols(dssp_string, newchar="L")
        dssp_string = pool_dssp_symbols(dssp_string, chars=["H", "G", "I"])
        dssp_string = pool_dssp_symbols(dssp_string, chars=["E", "B"])
    sample_pct_beta = sum([c == "E" for c in dssp_string])/len(dssp_string)
    sample_pct_alpha = sum([c == "H" for c in dssp_string])/len(dssp_string)

    return {'sample_pct_beta':sample_pct_beta, 'sample_pct_alpha':sample_pct_alpha}

def get_info(pdb):
    print(pdb)
    structure = parse_struct(pdb)
    num_residues = len(list(structure.get_residues()))
    dssp = get_3state_dssp(pdb, structure)
    dssp['num_residues'] = num_residues
    print(dssp)
    return (pdb, dssp)

def get_bond_lengths(pdbs, filter_diverged_samples=True, n_samp_per_len=2):
    sample_feats_list = [utils.load_feats_from_pdb(pdb) for pdb in pdbs]
    reject_rate = defaultdict(lambda: 0)
    if filter_diverged_samples:  # filter out samples which are just long helices
        filtered_feats_list = []
        for feat in sample_feats_list:
            if feat['atom_positions'][:, :3].var().sqrt() > 20:
                reject_rate[feat['aatype'].shape[0]] += 1
            else:
                filtered_feats_list.append(feat)
        reject_rate = {k: v / n_samp_per_len for k, v in reject_rate.items()}
    else:
        filtered_feats_list = sample_feats_list
    bond_mse = evaluation.compute_bond_length_metric([feat["atom_positions"] for feat in filtered_feats_list], 
                                                     [feat["aatype"] for feat in filtered_feats_list])
    bond_lengths_dict = utils.batched_fullatom_bond_lengths_from_coords([feat["atom_positions"] for feat in filtered_feats_list], 
                                                     [feat["aatype"] for feat in filtered_feats_list])
    bond_lengths_dict_flat = [v for k, v in bond_lengths_dict.items()]
    bond_lengths_list = [v for nested in bond_lengths_dict_flat for k, v in nested.items()]
    bond_lengths = [v for nested in bond_lengths_list for v in nested]

    per_pdb_bond_length = {}
    for pdb, feat in zip(pdbs, sample_feats_list):
        per_pdb_bond_length[pdb] = evaluation.compute_bond_length_metric([feat["atom_positions"]], 
                                    [feat["aatype"]])

    return bond_mse, reject_rate, bond_lengths, per_pdb_bond_length


def get_bond_angles(pdbs, filter_diverged_samples=True, n_samp_per_len=2):
    sample_feats_list = [utils.load_feats_from_pdb(pdb) for pdb in pdbs]
    reject_rate = defaultdict(lambda: 0)
    if filter_diverged_samples:  # filter out samples which are just long helices
        filtered_feats_list = []
        for feat in sample_feats_list:
            if feat['atom_positions'][:, :3].var().sqrt() > 20:
                reject_rate[feat['aatype'].shape[0]] += 1
            else:
                filtered_feats_list.append(feat)
        reject_rate = {k: v / n_samp_per_len for k, v in reject_rate.items()}
    else:
        filtered_feats_list = sample_feats_list
    bond_angles_dict = utils.batched_fullatom_bond_angles_from_coords([feat["atom_positions"] for feat in filtered_feats_list], 
                                                     [feat["aatype"] for feat in filtered_feats_list])
    # bond_mse = evaluation.compute_bond_angle_metric(bond_angles_dict)
    bond_mse = -1
    bond_angles_dict_flat = [v for k, v in bond_angles_dict.items()]
    bond_angles_list = [v for nested in bond_angles_dict_flat for k, v in nested.items()]
    bond_angles = [v for nested in bond_angles_list for v in nested]
    return bond_mse, reject_rate, bond_angles

def get_chi_angles(pdbs, filter_diverged_samples=True, n_samp_per_len=2):
    sample_feats_list = [utils.load_feats_from_pdb(pdb) for pdb in pdbs]
    reject_rate = defaultdict(lambda: 0)
    if filter_diverged_samples:  # filter out samples which are just long helices
        filtered_feats_list = []
        for feat in sample_feats_list:
            if feat['atom_positions'][:, :3].var().sqrt() > 20:
                reject_rate[feat['aatype'].shape[0]] += 1
            else:
                filtered_feats_list.append(feat)
        reject_rate = {k: v / n_samp_per_len for k, v in reject_rate.items()}
    else:
        filtered_feats_list = sample_feats_list
    
    chi_angles_masks = [utils.get_chi_angles(feat["atom_positions"], feat['aatype']) for feat in filtered_feats_list]
    chi_angles_tups = [x[m.bool()] for x, m in chi_angles_masks]
    chi_angles = [v.item() for nested in chi_angles_tups for v in nested]
    print(chi_angles)
    chi_mse = -1
    return chi_mse, reject_rate, chi_angles


def main():
    manager = Manager()
    manager.parse_args()
    args = manager.args

    print(args.samples)
    base_dir = args.samples
    sample_dir = f"{base_dir}/samples"
    tm_dir = f"{base_dir}/tms"
    foldseek_dir = f"{base_dir}/foldseek"
    foldseek_self_dir = f"{base_dir}/foldseek_self"

    if "foldseek" in args.task:
        if not os.path.exists(foldseek_dir):
            os.makedirs(foldseek_dir, exist_ok=True)
    if "foldseek_self" in args.task:
        if not os.path.exists(foldseek_self_dir):
            os.makedirs(foldseek_self_dir, exist_ok=True)
    if "tm" in args.task:
        if not os.path.exists(tm_dir):
            os.makedirs(tm_dir, exist_ok=True)

    if args.task == 'tm_score_self':
        all_pdbs = sorted(glob.glob(sample_dir+ '/*.pdb'))
        search_pdbs = all_pdbs[args.offset:args.offset + args.limit]
        for pdb in search_pdbs:
            generate_tm_pdb(pdb, all_pdbs, tm_dir, "self")
    elif args.task == 'tm_score_dataset':
        dataset_path = args.dataset
        train_file = open(f"{dataset_path}/train_pdb_keys.list")
        train_pdbs = train_file.read().strip().split("\n")
        dataset_pdbs = [f"{dataset_path}/pdb_store/{pdb_name}" for pdb_name in train_pdbs]

        all_pdbs = sorted(glob.glob(sample_dir + '/*.pdb'))
        search_pdbs = all_pdbs[args.offset:args.offset + args.limit]
        for pdb in search_pdbs:
            generate_tm_pdb(pdb, dataset_pdbs, tm_dir, "dataset")
    # elif args.task == 'copy_nn':
    #     dataset_path = args.dataset
    #     all_tms = sorted(glob.glob(f"{tm_dir}/tm_score_max_*dataset.json"))

    #     nn_dir = f"{base_dir}/nns"
    #     if not os.path.exists(nn_dir):
    #         os.makedirs(nn_dir)

    #     for tm in all_tms:
    #         print(tm)
    #         sample_pdb = tm.split("tm_score_max_")[1].replace("_dataset.json", "")
    #         with open(tm) as f:
    #             tm_infos = json.loads(f.read())
    #             for i in range(0, 3):
    #                 dataset_pdb = tm_infos[i]['pdb']
    #                 src_file = f"{dataset_path}/pdb_store/{dataset_pdb}"
    #                 score = round(100 * tm_infos[i]['tm_score'])
    #                 dest_file = f"{nn_dir}/{sample_pdb}_nn_{i}_{score}_{dataset_pdb}.pdb"
    #                 shutil.copyfile(src_file, dest_file)
    elif args.task == 'structure':
        ss = [get_info(pdb1) for pdb1 in glob.glob(f"{sample_dir}/*.pdb")]
        ss_dict = dict((x,y) for x, y in ss)
        print(ss_dict)
        json.dump(ss_dict, open(f"{base_dir}/sequence_structure.json", 'w'))
    elif args.task == 'bond_length' or args.task == 'bond_angle' or args.task == 'chi_angle':
        metrics_dir = f"{base_dir}/bond_metrics"
        if not os.path.exists(metrics_dir):
            os.makedirs(metrics_dir, exist_ok=True)

        samp_pdbs = sorted(glob.glob(sample_dir + '/*samp*.pdb'))[::16]
        init_pdbs = sorted(glob.glob(sample_dir + '_inits/*init*.pdb'))[::16]

        if args.task == 'bond_length':
            bond_metric, reject_rate, bond_lengths, per_pdb_bond_length = get_bond_lengths(samp_pdbs, filter_diverged_samples=False)
            with open(f"{metrics_dir}/bond_length_metrics_samp.json", "w") as f:
                f.write(json.dumps({"bond_length_metric": bond_metric, "reject_rate": reject_rate, "bond_lengths": bond_lengths, "per_pdb_bond_length": per_pdb_bond_length}))

            bond_metric, reject_rate, bond_lengths, per_pdb_bond_length = get_bond_lengths(init_pdbs, filter_diverged_samples=False)
            with open(f"{metrics_dir}/bond_length_metrics_init.json", "w") as f:
                f.write(json.dumps({"bond_length_metric": bond_metric, "reject_rate": reject_rate, "bond_lengths": bond_lengths, "per_pdb_bond_length": per_pdb_bond_length}))

        elif args.task == 'bond_angle':
            bond_metric, reject_rate, bond_angles = get_bond_angles(samp_pdbs, filter_diverged_samples=False)
            with open(f"{metrics_dir}/bond_angle_metrics_samp.json", "w") as f:
                f.write(json.dumps({"bond_angle_metric": bond_metric, "reject_rate": reject_rate, "bond_angles": bond_angles}))

            bond_metric, reject_rate, bond_angles = get_bond_angles(init_pdbs, filter_diverged_samples=False)
            with open(f"{metrics_dir}/bond_length_metrics_init.json", "w") as f:
                f.write(json.dumps({"bond_angle_metric": bond_metric, "reject_rate": reject_rate, "bond_angles": bond_angles}))
        else:
            chi_metric, reject_rate, chi_angles = get_chi_angles(samp_pdbs, filter_diverged_samples=False)
            with open(f"{metrics_dir}/chi_angle_metrics_samp.json", "w") as f:
                f.write(json.dumps({"chi_angle_metric": chi_metric, "reject_rate": reject_rate, "chi_angles": chi_angles}))

            chi_metric, reject_rate, chi_angles = get_chi_angles(init_pdbs, filter_diverged_samples=False)
            with open(f"{metrics_dir}/chi_length_metrics_init.json", "w") as f:
                f.write(json.dumps({"chi_angle_metric": chi_metric, "reject_rate": reject_rate, "chi_angles": chi_angles}))

    elif args.task == 'bond_length_data' or args.task == 'bond_angle_data' or args.task == 'chi_angle_data':
        metrics_dir = f"eval_scripts/bond_metrics_data"
        if not os.path.exists(metrics_dir):
            os.makedirs(metrics_dir, exist_ok=True)

        dataset_path = args.dataset
        train_file = f"{dataset_path}/train_pdb_keys.list"
        with open(train_file, 'r') as f:
            train_list = f.read().strip().split("\n")
        train_pdbs = [x for x in train_list if len(x) == 7][0:100]

        if args.task == "bond_length_data":
            with open(f"{metrics_dir}/bond_length_metrics.json", "w") as f:
                samp_pdbs = [f"{dataset_path}/pdb_store/{pdb}" for pdb in train_pdbs]
                bond_metric, reject_rate, bond_lengths = get_bond_lengths(samp_pdbs, filter_diverged_samples=False)
                f.write(json.dumps({"bond_length_metric": bond_metric, "reject_rate": reject_rate, "bond_lengths": bond_lengths}))
        elif args.task == "bond_angle_data":
            with open(f"{metrics_dir}/bond_angle_metrics.json", "w") as f:
                samp_pdbs = [f"{dataset_path}/pdb_store/{pdb}" for pdb in train_pdbs]
                bond_metric, reject_rate, bond_angles = get_bond_angles(samp_pdbs, filter_diverged_samples=False)
                f.write(json.dumps({"bond_angle_metric": bond_metric, "reject_rate": reject_rate, "bond_angles": bond_angles}))
        else:
            with open(f"{metrics_dir}/chi_angle_metrics.json", "w") as f:
                samp_pdbs = [f"{dataset_path}/pdb_store/{pdb}" for pdb in train_pdbs]
                chi_metric, reject_rate, chi_angles = get_chi_angles(samp_pdbs, filter_diverged_samples=False)
                f.write(json.dumps({"chi_angle_metric": chi_metric, "reject_rate": reject_rate, "chi_angles": chi_angles}))

if __name__ == "__main__":
    main()
