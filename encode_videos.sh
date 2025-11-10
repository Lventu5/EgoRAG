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

# Setup fast local cache (CRITICAL for performance!)
echo ""
echo "=== SETTING UP FAST LOCAL CACHE ==="
SLOW_CACHE="/cluster/scratch/tnanni/hub"
FAST_CACHE="/tmp/$USER/hf_cache"
mkdir -p "$FAST_CACHE"

# Copy LLaVA to local SSD (only once per job, ~2-3 min)
if [ ! -d "$FAST_CACHE/models--llava-hf--LLaVA-NeXT-Video-7B-hf" ]; then
    echo "Copying LLaVA (14GB) to local cache..."
    time rsync -ah --info=progress2 \
        "$SLOW_CACHE/models--llava-hf--LLaVA-NeXT-Video-7B-hf" \
        "$FAST_CACHE/"
    echo "✓ LLaVA cached locally"
else
    echo "✓ LLaVA already in local cache"
fi

# Copy other models (much smaller)
for model in "models--openai--clip-vit-base-patch32" \
             "models--openai--whisper-base" \
             "models--laion--clap-htsat-unfused" \
             "models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K" \
             "models--sentence-transformers--all-MiniLM-L6-v2"; do
    if [ -d "$SLOW_CACHE/$model" ] && [ ! -d "$FAST_CACHE/$model" ]; then
        rsync -ah "$SLOW_CACHE/$model" "$FAST_CACHE/"
    fi
done

# Use fast cache
export HF_HOME="$FAST_CACHE"
export TRANSFORMERS_CACHE="$FAST_CACHE"
echo "Using fast cache: $FAST_CACHE"
echo "===================================="
echo ""

python -m test.encode_videos

echo "========================="