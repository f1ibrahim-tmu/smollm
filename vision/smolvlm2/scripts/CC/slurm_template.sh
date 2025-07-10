#!/bin/bash
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128000M
#SBATCH --time=300:0:0
#SBATCH --mail-user=f1ibrahim@torontomu.ca
#SBATCH --mail-type=ALL

## Using 4 V100 GPUs (more powerful than P100)
#SBATCH --gres=gpu:v100l:4

cd /home/f7ibrahi/projects/def-wangcs/f7ibrahi/projects/smolvlm/

## Load required modules
module restore smolvlm_modules_test

## Activate your conda environment
source /home/f7ibrahi/projects/def-wangcs/f7ibrahi/miniconda3/etc/profile.d/conda.sh
conda activate conda_smolvlm
#nvidia-smi
#eval "$(conda shell.bash hook)"
#conda activate conda_visionmamba

# Set environment variables for SLURM
export OMP_NUM_THREADS=4

## Set memory-related environment variables
export NCCL_BLOCKING_WAIT=1  # Wait indefinitely for NCCL operations
export NCCL_ASYNC_ERROR_HANDLING=1  # Handle errors asynchronously
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:2048,garbage_collection_threshold:0.6"

## Run script
bash ./vision/scripts/CC/
