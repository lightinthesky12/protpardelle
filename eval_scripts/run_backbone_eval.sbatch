#!/bin/bash
#SBATCH --time=0-12:00:00
#SBATCH --mem=48G
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu,bioe
#SBATCH --job-name=bb_eval
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
else parampath="--parampath ${SWEEP_NAME}.json";
fi

prefix=backbone_checkpoint
targetdir=/scratch/users/alexechu/protpardelle_paper/samples/${prefix}_${MINLEN}_${MAXLEN}_${SWEEP_NAME}
modelcheckpoint=/scratch/users/alexechu/protpardelle/checkpoints


echo "python draw_samples.py --model_checkpoint ${modelcheckpoint} --type backbone --perlen ${PERLEN} --minlen ${MINLEN} --maxlen ${MAXLEN} --steplen ${STEPLEN} --targetdir ${targetdir} ${parampath}"
python draw_samples.py --model_checkpoint ${modelcheckpoint} --type backbone --perlen ${PERLEN} --minlen ${MINLEN} --maxlen ${MAXLEN} --steplen ${STEPLEN} --targetdir ${targetdir} ${parampath}

echo "sbatch eval_scripts/sc.sbatch ${targetdir} backbone"
sbatch eval_scripts/sc.sbatch ${targetdir} backbone

echo "sbatch eval_scripts/div.sbatch ${targetdir}"
sbatch eval_scripts/div.sbatch ${targetdir}


##echo "sbatch eval_scripts/nov.sbatch ${targetdir} False"
##sbatch eval_scripts/nov.sbatch ${targetdir} False
