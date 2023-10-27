import glob
import json
from collections import defaultdict
import os
import statistics

def mean(x):
    if len(x) == 0:
        return 0
    return sum(x) / len(x)

prefix = 'allatom'
all_dirs = sorted(glob.glob(f"../protpardelle_paper/samples/{prefix}*"))
result = defaultdict(lambda: {})
for dir in all_dirs:
    info = dir.replace(f'../protpardelle_paper/samples/{prefix}', '')

    # sc metrics
    try:
        vals = defaultdict(lambda: [])
        vals_init = defaultdict(lambda: [])
        for file in sorted(glob.glob(f"{dir}/sc_metrics/allatom_*results.json")):
            with open(file) as f:
                print(dir)
                all_results = json.loads(f.read())
                rejects = []
                print(len(all_results), len(rejects))
                for pdb in all_results:
                    stats = all_results[pdb]
                    if "_samp" in os.path.basename(pdb) and pdb.replace("/scratch/users/alexechu/", "../") not in rejects:
                        for key in stats:
                            if "_rmsd_best" in key:
                                vals[key].append(stats[key])
                    if "_init" in os.path.basename(pdb) and pdb.replace("/scratch/users/alexechu/", "../") not in rejects:
                        for key in stats:
                            if "_rmsd_best" in key:
                                vals_init[key].append(stats[key])

        avgs = {}
        for key in vals:
            avgs[key] = {"samp": {"mean": mean(vals[key]), "stddev": statistics.stdev(vals[key]), "samples": len(vals[key])}}
        for key in vals_init:
            avgs[key]["init"] = {"mean": mean(vals_init[key]), "stddev": statistics.stdev(vals_init[key]), "samples": len(vals_init[key])}
    except Exception as e:
        print(e)
        avgs = {}

    # bond metrics
    bond_info = {}
    try:
        with open(f'{dir}/bond_metrics/bond_length_metrics_init.json') as f:
            print(dir)
            bond_length_info = json.loads(f.read())
            bond_lengths = bond_length_info.get('bond_length_metric', bond_length_info.get("bond_lengths"))
    except:
        bond_lengths = -1
    bond_info['init'] = bond_lengths

    try:
        with open(f'{dir}/bond_metrics/bond_length_metrics_samp.json') as f:
            print(dir)
            bond_length_info = json.loads(f.read())
            bond_lengths = bond_length_info.get('bond_length_metric', bond_length_info.get("bond_lengths"))
    except:
        bond_lengths = -1
    bond_info['samp'] = bond_lengths

    params = {}
    model = ""
    sampling = ""
    runtime = ""
    with open(f'{dir}/readme.txt') as f:
        started = False
        for line in f:
            print(line)
            if line.startswith("Model run time: "):
                runtime = line.strip().replace("Model run time: ", "")
            if line.startswith("Model checkpoint: "):
                model = line.strip().replace("Model checkpoint: ", "")
            if "samples per length" in line:
                sampling = line.strip()
            if line.startswith("Sampling params:"):
                started = True
                continue
            if line.startswith("Total job"):
                started = False
                continue
            if started:
                split = line.split()
                print(split)
                if len(split) >= 2:
                    params[split[0]] = split[1]
        text = f.read()

    result[info] = {"sc_metrics": avgs, "bond_length": bond_info, "params": params, "model": model, "sampling": sampling, "runtime": runtime}

print(json.dumps(result, indent=2))


with open('analysis/decent_allatom_proteins_pdbs/table.csv', 'w') as f: 
    f.write('model,sampling,runtime,n_steps,s_churn,step_scale,sc_rmsd_mean,sc_rmsd_stddev,sc_count,bond_length_metric\n')
    for key in result:
        stats = result[key]
        f.write(f"{key}")
        f.write(f",{stats['model']}")
        f.write(f",{stats['sampling']}")
        f.write(f",{stats['runtime']}")
        f.write(f",{stats['params']['n_steps']}")
        f.write(f",{stats['params']['s_churn']}")
        f.write(f",{stats['params']['step_scale']}")
        f.write(f",{stats['sc_metrics'].get('j_sc_rmsd_best', {}).get('samp', {}).get('mean')}")
        f.write(f",{stats['sc_metrics'].get('j_sc_rmsd_best', {}).get('samp', {}).get('stddev')}")
        f.write(f",{stats['sc_metrics'].get('j_sc_rmsd_best', {}).get('samp', {}).get('samples')}")
        f.write(f",{stats.get('bond_length', {}).get('samp')}\n")
