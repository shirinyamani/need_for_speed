#!/bin/bash
echo "Hi there! Trial for allocation 1 GPU A100 for 10h started..."

#! /bin/bash
# ====================================
#SBATCH --job-name=1GPU_test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=0-04:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-a100
# ====================================

# Load required modules
module load python/3.10.4
module load cuda/12.1.1
