#!/bin/bash
####### Reserve computing resources #############
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --partition=gpu

####### Set environment variables ###############
module load python/anaconda3-2018.12
pip install  --user -r requirements.txt

wandb login 5c59fcaac06279f7c55d328b08db1a3a0a65058e
export WANDB_API_KEY=5c59fcaac06279f7c55d328b08db1a3a0a65058e


# Run your Python script
python mobile-net-v2.py