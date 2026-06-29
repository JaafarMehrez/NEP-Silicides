#!/bin/sh
#SBATCH --partition=gpu
#SBATCH --job-name=NEPTiSi-train
#SBATCH --nodes=1
#SBATCH --nodelist=g03
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1
#SBATCH --output=out.%j
#SBATCH --error=err.%j

module load cuda/12.1
module load conda/3-2023.09

export LD_LIBRARY_PATH=/share/app/cuda/cuda-12.1/targets/x86_64-linux/lib:$LD_LIBRARY_PATH

echo "Starting date : `date`"

srun --export=ALL ~/nep > outlog

echo "Program finished with exit code $? at: `date`"
