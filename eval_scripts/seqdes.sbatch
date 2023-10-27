#!/bin/bash
#SBATCH --time=0-12:00:00
#SBATCH --mem=48G
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu,bioe
#SBATCH --job-name=lc-seqdes
#SBATCH --array=0-650:50%2
#SBATCH --output=/scratch/users/alexechu/protpardelle/slurms/%j_seqdes.out

source activate folding
mkdir -p ../protpardelle_paper/samples/deft_flower_epoch200
python eval_sc_metrics.py --samples ../protpardelle_paper/samples/deft_flower_epoch200 --metric seqdes --limit 50 --offset $SLURM_ARRAY_TASK_ID --modeldir /scratch/users/alexechu/training_logs/recover/deft-flower-63 --modelepoch 200