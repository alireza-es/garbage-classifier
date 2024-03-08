#!/bin/bash
####### Reserve computing resources #############
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=1G
#SBATCH --partition=gpu

####### Set environment variables ###############
module load python/anaconda3-2018.12
pip install -r requirements.txt

# Run your Python script
python mobile-net-v2.py