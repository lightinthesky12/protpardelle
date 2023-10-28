#!/bin/bash
#SBATCH --time=0-12:00:00
#SBATCH --mem=48G
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=bioe
##SBATCH --nodelist=sh03-16n14
#SBATCH --job-name=lc-sc
#SBATCH --array=0-200:100%16
#SBATCH --output=/scratch/users/alexechu/protpardelle/slurms/%j_sc.out

source activate delle
ml load cudnn/8.6

targetdir=$1
metric=$2

echo "python eval_sc_metrics.py  --samples ${targetdir} --metric ${metric} --offset $SLURM_ARRAY_TASK_ID --limit 100"
python eval_sc_metrics.py  --samples ${targetdir} --metric ${metric} --offset $SLURM_ARRAY_TASK_ID --limit 100
