#!/bin/bash

# Baselines
export CONFIG=configs/train/cyclegan/cyclegan_all.yml
sbatch --export=ALL,CONFIG=$CONFIG ./scripts/train_thermal_generation_baseline.sbatch
export CONFIG=configs/train/pix2pix/pix2pix_all.yml
sbatch --export=ALL,CONFIG=$CONFIG ./scripts/train_thermal_generation_baseline.sbatch
export CONFIG=configs/train/pix2pixHD/pix2pixHD_all.yml
sbatch --export=ALL,CONFIG=$CONFIG ./scripts/train_thermal_generation_baseline.sbatch
export CONFIG=configs/train/vqgan/vqgan_all.yml
sbatch --export=ALL,CONFIG=$CONFIG ./scripts/train_thermal_generation_baseline.sbatch

# KLVAE
export CONFIG=configs/train/ldm/klvae_all_256_1st.yml
sbatch --export=ALL,CONFIG=$CONFIG ./scripts/train_thermal_generation_long.sbatch
export CONFIG=configs/train/ldm/klvae_all_256_3rd.yml
sbatch --export=ALL,CONFIG=$CONFIG ./scripts/train_thermal_generation_long.sbatch

# SIT TRANSFORMER AND PATCH SIZE
export CONFIG=configs/train/sit/sit_s2_q.yml
sbatch --export=ALL,CONFIG=$CONFIG ./scripts/train_thermal_generation_long.sbatch
export CONFIG=configs/train/sit/sit_b2_q.yml
sbatch --export=ALL,CONFIG=$CONFIG ./scripts/train_thermal_generation_long.sbatch
export CONFIG=configs/train/sit/sit_l2_q.yml
sbatch --export=ALL,CONFIG=$CONFIG ./scripts/train_thermal_generation_long.sbatch
export CONFIG=configs/train/sit/sit_xl2_q.yml
sbatch --export=ALL,CONFIG=$CONFIG ./scripts/train_thermal_generation_long.sbatch
export CONFIG=configs/train/sit/sit_xl4_q.yml
sbatch --export=ALL,CONFIG=$CONFIG ./scripts/train_thermal_generation_long.sbatch
export CONFIG=configs/train/sit/sit_xl8_q.yml
sbatch --export=ALL,CONFIG=$CONFIG ./scripts/train_thermal_generation_long.sbatch

# RGB INJECTION
export CONFIG=configs/train/sit_cond/sit_l2_concat.yml
sbatch --export=ALL,CONFIG=$CONFIG ./scripts/train_thermal_generation_long.sbatch