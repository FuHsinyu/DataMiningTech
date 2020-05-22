#!/bin/bash

#SBATCH -t 02:25:00
#SBATCH -N 1
#SBATCH -p normal
#SBATCH --mem=60G

export PYTHONUNBUFFERED=yes

python main_lightgbm.py
# python run.py --train_csv ./data/training_set_VU_DM.csv --test_csv ./data/test_set_VU_DM.csv --output_dir ./log/