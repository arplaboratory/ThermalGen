#!/bin/bash 
#SBATCH --nodes=1                        # requests 3 compute servers
#SBATCH --ntasks-per-node=1              # runs 2 tasks on each server
#SBATCH --cpus-per-task=16               # uses 1 compute core per task
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=250GB
#SBATCH --job-name=test_tgm
#SBATCH --output=test_tgm.out

# TGM training should not use mixed precision otherwise low accuracy
eval "$(conda shell.bash hook)"
conda activate ThermalGen

python3 test.py --config $CONFIG