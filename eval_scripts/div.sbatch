#!/bin/bash
#SBATCH --time=1-00:00:00
#SBATCH --mem=64G
#SBATCH -c 8
#SBATCH --partition=possu,normal,owners
#SBATCH --job-name=lc-div
#SBATCH --array=0-7

module load gcc/12

targetdir=$1
echo "ls ${targetdir}/samples/*samp[0-${SLURM_ARRAY_TASK_ID}].pdb > ${targetdir}/pdbs_${SLURM_ARRAY_TASK_ID}.list"
ls ${targetdir}/samples/*samp[0-${SLURM_ARRAY_TASK_ID}].pdb > ${targetdir}/pdbs_${SLURM_ARRAY_TASK_ID}.list
echo "/scratch/users/alexechu/maxcluster64bit -l ${targetdir}/pdbs_${SLURM_ARRAY_TASK_ID}.list -C 1  -T 0.4 -Tm 0.4 -in > ${targetdir}/clusters_${SLURM_ARRAY_TASK_ID}.txt"
/scratch/users/alexechu/maxcluster64bit -l ${targetdir}/pdbs_${SLURM_ARRAY_TASK_ID}.list -C 1  -T 0.4 -Tm 0.4 -in > ${targetdir}/clusters_${SLURM_ARRAY_TASK_ID}.txt

