#!/bin/bash
#SBATCH --time=1-00:00:00
#SBATCH --mem=64G
#SBATCH -c 8
#SBATCH --partition=possu,normal,owners
#SBATCH --job-name=lc-div
#SBATCH --array=0-200:100%1
#SBATCH --output=/scratch/users/alexechu/protpardelle/slurms/%j_div.out


module load gcc/12
source activate delle

targetdir=$1


echo "python eval_novelty.py --samples ${targetdir} --task tm_score_self --offset $SLURM_ARRAY_TASK_ID --limit 100"
python eval_novelty.py --samples ${targetdir} --task tm_score_self --offset $SLURM_ARRAY_TASK_ID --limit 100
