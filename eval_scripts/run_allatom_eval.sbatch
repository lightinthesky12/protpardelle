#!/bin/bash
#SBATCH --time=0-12:00:00
#SBATCH --mem=48G
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=bioe
#SBATCH --job-name=aa_eval
#SBATCH --output=/scratch/users/alexechu/protpardelle/slurms/%j.out

source activate delle

## expects MINLEN, MAXLEN, PERLEN, STEPLEN, SWEEP_NAME
for ARGUMENT in "$@"
do
   KEY=$(echo $ARGUMENT | cut -f1 -d=)

   KEY_LENGTH=${#KEY}
   VALUE="${ARGUMENT:$KEY_LENGTH+1}"

   export "$KEY"="$VALUE"
done

parampath=""
if [ -z $SWEEP_NAME ];
then SWEEP_NAME="default";
else parampath="--parampath sweeps/${SWEEP_NAME}.json";
fi

prefix=allatom_checkpoint
targetdir=/scratch/users/alexechu/protpardelle_paper/samples/${prefix}_${MINLEN}_${MAXLEN}_${SWEEP_NAME}
modelcheckpoint=/scratch/users/alexechu/protpardelle/checkpoints
mpnnmodel=/scratch/users/alexechu/protpardelle/checkpoints/minimpnn_state_dict.pth

echo "python draw_samples.py --model_checkpoint ${modelcheckpoint} --type allatom --perlen ${PERLEN} --minlen ${MINLEN} --maxlen ${MAXLEN} --steplen ${STEPLEN} --targetdir ${targetdir}" --mpnnpath ${mpnnmodel}
python draw_samples.py  --model_checkpoint ${modelcheckpoint} --type allatom --perlen ${PERLEN} --minlen ${MINLEN} --maxlen ${MAXLEN} --steplen ${STEPLEN} --targetdir ${targetdir} --mpnnpath ${mpnnmodel}

echo "sbatch eval_scripts/sc.sbatch ${targetdir} allatom"
sbatch eval_scripts/sc.sbatch ${targetdir} allatom

echo "sbatch eval_scripts/div.sbatch ${targetdir}"
sbatch eval_scripts/div.sbatch ${targetdir}

##echo "sbatch eval_scripts/nov.sbatch ${targetdir} True"
##sbatch eval_scripts/nov.sbatch ${targetdir} True

##echo "sbatch eval_scripts/bond.sbatch ${targetdir}"
##sbatch eval_scripts/bond.sbatch ${targetdir}
