#!/bin/bash
#SBATCH --time=1-00:00:00
#SBATCH --mem=64G
#SBATCH -c 8
#SBATCH --partition=possu,normal,owners
#SBATCH --job-name=lc-bond
#SBATCH --output=/scratch/users/alexechu/protpardelle/slurms/%j_bond.out

module load gcc/12

source activate folding

targetdir=$1

echo "python eval_novelty.py --samples /${targetdir} --task bond_length"
python eval_novelty.py --samples /${targetdir} --task bond_length
echo "python eval_novelty.py --samples /${targetdir} --task bond_angle"
python eval_novelty.py --samples /${targetdir} --task bond_angle
echo "python eval_novelty.py --samples /${targetdir} --task chi_angle"
python eval_novelty.py --samples /${targetdir} --task chi_angle
echo "python eval_novelty.py --samples /${targetdir} --task secondary_structure"
python eval_novelty.py --samples ${targetdir} --task secondary_structure
