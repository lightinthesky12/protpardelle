import glob
import pickle

from transformers import AutoTokenizer, EsmForProteinFolding
import torch

import evaluation
from core import protein_mpnn
from core import utils
from core import data
import sys
import json
from collections import defaultdict
import statistics
import os
import argparse
from colabdesign.af import mk_af_model


device = "cuda" if torch.cuda.is_available() else "cpu"

# fb alphafold
# esmfold = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1").to(device)
# esmfold.esm = esmfold.esm.half()
# tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")


# seq prediction from structure
mpnn = protein_mpnn.get_mpnn_model(
    device=device,
)

all_metrics = {}


METRIC_HELP_STRING = """Which metric to run. Can be:
- backbone: given a struct sample, generates seq with mpnn, then predict structure with esm. compare structure of sample and esm prediction
- allatom: given a allatom (seq + struct) sample, predict structure with esm. compare structure of sample and esm prediction
"""

class Manager(object):
    def __init__(self):

        self.parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

        self.parser.add_argument("--samples", type=str, required=True, help="Directory of dataset with samples to analyze. Samples should be in $DIR/samples")
        self.parser.add_argument("--modeldir", type=str, help="Model base directory, ex '/scratch/users/alexechu/training_logs/recover/deft-flower-63'")
        self.parser.add_argument("--modelepoch", type=int, help="Model epoch, ex 200")
        self.parser.add_argument("--singlesample", type=str, default=None, help="Use to run on a single sample. Overrides --samples")
        self.parser.add_argument("--metric", type=str, required=True, help=METRIC_HELP_STRING)
        self.parser.add_argument("--dataset", type=str, default='/scratch/users/alexechu/datasets/ingraham_cath_dataset',help="CATH dataset of coordinate files")
        self.parser.add_argument("--offset", type=int, default=0, help="Use with limit to work on a subset of samples. Offset to start on in sorted pdbs list.")
        self.parser.add_argument("--limit", type=int, default=10, help="Use with offset to work on a subset of samples. Max number of pdbs to process.")
        self.parser.add_argument("--param", type=str, default=None, help="Which sampling param to vary")
        self.parser.add_argument("--paramval", type=float, default=None, help="Which param val to use")
        self.parser.add_argument("--mpnn", type=int, default=None, help="How many tries for mpnn for backbone")

    def add_argument(self, *args, **kwargs):
        self.parser.add_argument(*args, **kwargs)

    def parse_args(self):
        self.args = self.parser.parse_args()

        return self.args

def mean(x):
    if len(x) == 0:
        return 0
    return sum(x) / len(x)


def eval_backbone_generation(structure, output_file, num_seqs, mpnn_model=mpnn, struct_pred_model=mk_af_model()):
    # this takes generated protein structure, generates seq with mpnn, then generates structure with esm
    metrics, best_idx = evaluation.compute_self_consistency(
        comparison_structures=[structure],
        mpnn_model=mpnn_model,
        struct_pred_model=struct_pred_model,
        # tokenizer=tokenizer,
        num_seqs=num_seqs,
        metric="both",
        output_file=output_file
    )
    assert best_idx == 0  # make sure we only ran on 1 structure
    metrics = {f"bb_{k}": v for k, v in metrics.items()}

    return metrics


def eval_allatom_backbone(structure, sequence, output_file, struct_pred_model=mk_af_model()):
    sc_metrics, best_idx, aux = evaluation.compute_self_consistency(
        [structure],
        [sequence],
        struct_pred_model=struct_pred_model,
        # tokenizer=tokenizer,
        return_aux=True,
        metric="both",
        output_file=output_file
    )
    sc_metrics = {f"j_{k}": v for k, v in sc_metrics.items()}

    return sc_metrics


def convert_metric_tensors(metrics):
    for metric_name in metrics:
        if isinstance(metrics[metric_name], torch.Tensor):
            metrics[metric_name] = metrics[metric_name].numpy().tolist()
        if isinstance(metrics[metric_name], list):
            for i in range(0, len(metrics[metric_name])):
                if isinstance(metrics[metric_name][i], torch.Tensor):
                    metrics[metric_name][i] = metrics[metric_name][i].numpy().tolist()
    return metrics


def main():
    manager = Manager()
    manager.parse_args()
    args = manager.args
    print(args)

    metric = args.metric
    limit = args.limit
    offset = args.offset
    all_metrics = {}

    sample_dir = f"{args.samples}/samples"

    preds_dir = f"{args.samples}/preds_{metric}"
    if not os.path.exists(preds_dir):
        os.makedirs(preds_dir, exist_ok=True)

    metrics_dir = f"{args.samples}/sc_metrics"
    if not os.path.exists(metrics_dir):
        os.makedirs(metrics_dir, exist_ok=True)

    if metric in ["backbone", "allatom"]:
        count = 0
        if args.singlesample:
            all_pdbs = [args.singlesample]
        else:
            all_pdbs = sorted(glob.glob(f"{sample_dir}/*samp*.pdb"))[offset:]

        num_mpnn_seqs = 1 if metric == "allatom" else 8
        if args.mpnn:
            if metric != "backbone":
                raise Exception("--mpnn only supported for backbone")

            num_mpnn_seqs = args.mpnn
            preds_dir = preds_dir + f"_{args.mpnn}"

            if not os.path.exists(preds_dir):
                os.makedirs(preds_dir, exist_ok=True)

        for pdb in all_pdbs:
            # load generated protein structure
            sample_feats = utils.load_feats_from_pdb(pdb)
            sequence = utils.aatype_to_seq(sample_feats["aatype"].long())
            structure = sample_feats["bb_coords"]
            pdb_file_name = os.path.basename(pdb)

            if metric == "backbone":
                pred_file_name = pdb_file_name.replace("samp", "pred")
                output_file = f"{preds_dir}/{pred_file_name}"
                metrics = eval_backbone_generation(structure, output_file, num_mpnn_seqs)
            else:
                pred_file_name = pdb_file_name.replace("samp", "samppred").replace("init", "initpred")
                output_file = f"{preds_dir}/{pred_file_name}"
                metrics = eval_allatom_backbone(structure, sequence, output_file)
            print(metrics)

            all_metrics[pdb] = metrics
            count = count + 1

            print(f"Finished {count} of {limit}, offset {offset}")
            if limit and count >= limit:
                break
    else:
        raise Exception("Not implemented")

    print(all_metrics)

    # don't write file if only a single sample
    if args.singlesample:
        return

    combined = defaultdict(list)
    for pdb in all_metrics:
        for metric_name in all_metrics[pdb]:
            if isinstance(all_metrics[pdb][metric_name], torch.Tensor):
                all_metrics[pdb][metric_name] = all_metrics[pdb][metric_name].cpu().numpy().tolist()
                if metric_name == "seq_rec":
                    all_metrics[pdb][metric_name] = all_metrics[pdb][metric_name][0]
                combined[metric_name].append(all_metrics[pdb][metric_name])
            if isinstance(all_metrics[pdb][metric_name], list):
                for i in range(0, len(all_metrics[pdb][metric_name])):
                    if isinstance(all_metrics[pdb][metric_name][i], torch.Tensor):
                        all_metrics[pdb][metric_name][i] = all_metrics[pdb][metric_name][i].cpu().numpy().tolist()

    if args.mpnn:
        metrics_dir = metrics_dir + f"_{args.mpnn}"
        if not os.path.exists(metrics_dir):
            os.makedirs(metrics_dir, exist_ok=True)

    output_file_name = f"{metrics_dir}/{metric}_{offset}_{limit}_results.json"
    output_file = open(output_file_name, "w")
    json.dump(all_metrics, output_file)

    print(combined)
    stats = {}
    for metric_name in combined:
        stats[metric_name] = {"mean": mean(combined[metric_name]), "stdev": statistics.stdev(combined[metric_name])}

    stats_file_name = f"{metrics_dir}/{metric}_{offset}_{limit}_stats.json"
    stats_file = open(stats_file_name, "w")
    json.dump(stats, stats_file)

if __name__ == "__main__":
    main()

