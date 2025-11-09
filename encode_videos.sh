#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --account=ls_polle
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem-per-cpu=6G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:30G
#SBATCH --job-name=encode_videos
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

conda activate video_encoding

echo "=== GPU STATUS ==="
nvidia-smi
echo "========================="

echo "Current directory: $(pwd)"
echo "Python executable: $(which python)"
echo "Python path: $PYTHONPATH"
echo "SLURM_NTASKS: $SLURM_NTASKS"
echo "SLURM_GPUS: $SLURM_GPUS"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

python -m test.encode_videos

echo "========================="