#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --account=ls_polle
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=2
#SBATCH --gres=gpumem:32G
#SBATCH --job-name=benchmark_gt_video
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=ALL          
#SBATCH --mail-user=mronconi@ethz.ch

module load stack/2024-06 python_cuda/3.11.6 cuda/12.1.1 eth_proxy cudnn
module load ffmpeg

echo "=== GPU STATUS ==="
nvidia-smi
echo "========================="

export PYTHONNOUSERSITE=1
export HUGGINGFACE_HUB_TOKEN=$(cat ~/.huggingface/token)
export HF_TOKEN=$HUGGINGFACE_HUB_TOKEN
export HF_HOME=${TRANSFORMERS_CACHE:-~/.cache/huggingface}

cd /cluster/project/cvg/students/tnanni/EgoRAG && /cluster/project/cvg/students/lventuroli/miniconda3/envs/RAGu/bin/python -m test.benchmark_egolife_qa --mode gt_video

echo "========================="
echo "Done."
