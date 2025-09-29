#!/bin/bash 
#SBATCH --nodes=1                        # requests 3 compute servers
#SBATCH --ntasks-per-node=4              # runs 2 tasks on each server
#SBATCH --cpus-per-task=8               # uses 1 compute core per task
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:a100:4
#SBATCH --account=pr_122_tandon_priority
#SBATCH --mem=300GB
#SBATCH --job-name=train_tgm
#SBATCH --output=train_tgm.out

# TGM training should not use mixed precision otherwise low accuracy
eval "$(conda shell.bash hook)"
conda activate ThermalGen

srun python3 main.py --config $CONFIG