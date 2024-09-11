#!/bin/bash
echo "Hi there! Trial for allocation 1 GPU A100 for 10h started..."

# Load required modules
module load python/3.10.4
module load cuda/12.1.1

# Reserve compute resources
salloc --mem=1G -t 01:00:00 -p gpu-v100 --gres=gpu:1

# Wait for allocation
echo "Waiting for resource allocation..."
squeue -u $USER

# Once allocated, you can run your Python script or start an interactive session
# For example:
# python your_script.py
# or
# srun --pty bash
