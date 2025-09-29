#!/bin/bash

# CFG
export CONFIG=configs/test/sit_cond/sit_l2_concat_2.yml
sbatch --export=ALL,CONFIG=$CONFIG scripts/test_thermal_generation.sbatch
export CONFIG=configs/test/sit_cond/sit_l2_concat_un.yml
sbatch --export=ALL,CONFIG=$CONFIG scripts/test_thermal_generation.sbatch