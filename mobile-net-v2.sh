#!/bin/bash
####### Reserve computing resources #############
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --partition=gpu

####### Set environment variables ###############

# Adjust the module load command to be compatible with Python 3.10.
# Note: You might need to change the module name or version based on what's available on your system.
module load python/3.10

# Set the environment name
ENV_NAME=py310

# Initialize Conda
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh

# Create the environment with Python 3.10 if it doesn't exist
conda create --name $ENV_NAME python=3.10 -y

# Activate the environment
conda activate $ENV_NAME

# Update pip, setuptools, and wheel to avoid installation issues
pip install --no-cache-dir --upgrade pip setuptools wheel

# Install required packages
pip install --no-cache-dir numpy matplotlib torch torchvision

# Reinstall wandb to ensure a clean, compatible installation
pip uninstall wandb -y
pip install --no-cache-dir wandb

# Install additional requirements from a file (if exists)
if [ -f requirements.txt ]; then
    pip install --user -r requirements.txt
fi

# Configure wandb
wandb login --relogin 5c59fcaac06279f7c55d328b08db1a3a0a65058e
export WANDB_API_KEY=5c59fcaac06279f7c55d328b08db1a3a0a65058e

# Run your Python script
python mobile-net-v2.py
