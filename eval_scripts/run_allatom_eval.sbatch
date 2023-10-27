#!/bin/bash
#SBATCH --time=0-12:00:00
#SBATCH --mem=48G
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu,bioe
#SBATCH --job-name=aa_eval
#SBATCH --output=/scratch/users/alexechu/protpardelle/slurms/%j.out

source activate folding

minlen=050
maxlen=400
perlen=2
steplen=1
prefix=allatom_eval
targetdir=../eval_results
model_checkpoint=../checkpoints
mpnnmodel=../checkpoints/minimpnn_state_dict.pth

echo "python draw_samples.py --model_checkpoint ${model_checkpoint} --type allatom --perlen ${perlen} --minlen ${minlen} --maxlen ${maxlen} --steplen ${steplen} --targetdir ${targetdir}" --mpnnpath ${mpnnmodel}
python draw_samples.py  --model_checkpoint ${model_checkpoint} --type allatom --perlen ${perlen} --minlen ${minlen} --maxlen ${maxlen} --steplen ${steplen} --targetdir ${targetdir} --mpnnpath ${mpnnmodel}

echo "sbatch eval_scripts/sc.sbatch ${targetdir} allatom"
sbatch eval_scripts/sc.sbatch ${targetdir} allatom

##echo "sbatch eval_scripts/nov.sbatch ${targetdir} True"
##sbatch eval_scripts/nov.sbatch ${targetdir} True

##echo "sbatch eval_scripts/bond.sbatch ${targetdir}"
##sbatch eval_scripts/bond.sbatch ${targetdir}
