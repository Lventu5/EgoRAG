#!/bin/bash
#SBATCH --time=5:00:00
#SBATCH --account=ls_polle
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=15
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:30G
#SBATCH --job-name=encode_videos
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

module load stack/2024-06 python_cuda/3.11.6 cuda/12.1.1 eth_proxy cudnn
module load ffmpeg

conda activate video_encoding
module load cudnn

echo "=== GPU STATUS ==="
nvidia-smi
echo "========================="

echo "Current directory: $(pwd)"
echo "Python executable: $(which python)"
echo "Python path: $PYTHONPATH"
echo "SLURM_NTASKS: $SLURM_NTASKS"
echo "SLURM_GPUS: $SLURM_GPUS"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

export PYTHONNOUSERSITE=1
export HUGGINGFACE_HUB_TOKEN=$(cat ~/.huggingface/token)
export HF_TOKEN=$HUGGINGFACE_HUB_TOKEN

# Cache setup is now handled by encode_videos.py
/cluster/project/cvg/students/lventuroli/miniconda3/envs/RAGu/bin/python -m core_tests.encode_videos
echo "========================="