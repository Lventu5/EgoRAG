#!/bin/bash
#SBATCH --job-name=encode_videos
#SBATCH --time=10:00:00
#SBATCH --account=dslab
#SBATCH --partition=jobs
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --mem=24G
#SBATCH --gpus=5060ti:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source /etc/profile

module load python/3.10
module load cuda/12.4


source /work/courses/dslab/team21/miniconda3/etc/profile.d/conda.sh 

conda activate video_encoding

cd /work/courses/dslab/team21/tommy/EgoRAG

echo "=== GPU STATUS BEFORE ==="
nvidia-smi
echo "========================="

echo "Current directory: $(pwd)"
echo "Python executable: $(which python)"
echo "Python path: $PYTHONPATH"
echo "SLURM_NTASKS: $SLURM_NTASKS"
echo "SLURM_GPUS: $SLURM_GPUS"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

python -m test.encode_videos

echo "=== GPU STATUS AFTER ==="
nvidia-smi