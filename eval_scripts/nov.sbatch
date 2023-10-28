#!/bin/bash
#SBATCH --time=1-00:00:00
#SBATCH --mem=64G
#SBATCH -c 8
#SBATCH --partition=possu,normal,owners
#SBATCH --job-name=lc-nov
#SBATCH --array=0-800:100%1
#SBATCH --output=/scratch/users/alexechu/protpardelle/slurms/%j_nov.out

module load gcc/12

source activate delle

targetdir=$1
allatom=$2

echo "python eval_novelty.py --samples ${targetdir} --task foldseek --offset $SLURM_ARRAY_TASK_ID --limit 100 --allatom ${allatom} --foldseekdb foldseek_dbs/trainDB --foldseektmpdir ./fs_tmpdirs"
python eval_novelty.py --samples ${targetdir} --task foldseek --offset $SLURM_ARRAY_TASK_ID --limit 100 --allatom ${allatom} --foldseekdb foldseek_dbs/trainDB --foldseektmpdir ./fs_tmpdirs
echo "python eval_novelty.py --samples ${targetdir} --task parse_foldseek --offset $SLURM_ARRAY_TASK_ID --limit 100 --allatom ${allatom}"
python eval_novelty.py --samples ${targetdir} --task parse_foldseek --offset $SLURM_ARRAY_TASK_ID --limit 100 --allatom ${allatom}
