#!/bin/bash
#SBATCH --account=dslab
#
# ==================== DYNAMIC GPU JOB LAUNCHER ====================
# Usage:
#   sbatch video_encoding_job.sh <partition> [time] [memory]
#
# Examples:
#   sbatch video_encoding_job.sh interactive
#   sbatch video_encoding_job.sh jobs 06:00:00 32G
# =================================================================

# ----------- READ ARGUMENTS -----------
PARTITION=${1:-interactive}    # default partition
TIME=${2:-02:00:00}            # default wall time
MEM=${3:-24G}                  # default memory
ACCOUNT=dslab                  # ✅ your valid ETH course tag
JOB_NAME="ego_gpu_${PARTITION}"

# ----------- SLURM DIRECTIVES -----------
#SBATCH --job-name=ego_gpu_encoding
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tuo.email@student.ethz.ch

# ----------- ENVIRONMENT SETUP -----------
module load cuda/12.1

# Usa la tua installazione locale di conda (non serve il modulo)
# NOTA: adattala se il path cambia, ma questa è quella standard del tuo env.
source /cluster/apps/miniconda3/bin/activate /work/courses/dslab/team21/conda_envs/video_encoding

# Piccolo check di debug per assicurarsi che l'env sia attivo
echo "Python path:" $(which python)
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# ----------- DEBUG INFO -----------
echo "------------------------------------------"
echo "Job Name:        ${JOB_NAME}"
echo "Partition:       ${PARTITION}"
echo "Time limit:      ${TIME}"
echo "Memory:          ${MEM}"
echo "Account:         ${ACCOUNT}"
echo "Node:            $(hostname)"
echo "Date:            $(date)"
echo "------------------------------------------"

# ----------- VALIDATE PARTITION -----------
VALID_PARTS=$(sinfo -h -o "%P" | tr -d '*')
if ! echo "$VALID_PARTS" | grep -qw "$PARTITION"; then
  echo "❌ ERROR: Partition '${PARTITION}' not found. Available partitions:"
  echo "$VALID_PARTS"
  exit 1
fi

# ----------- EXECUTION -----------
cd /work/courses/dslab/team21/EgoRAG/encoding_try

srun --account=${ACCOUNT} \
     --partition=${PARTITION} \
     --time=${TIME} \
     --mem=${MEM} \
     --cpus-per-task=8 \
     bash -c '
        echo "Running on $(hostname)"
        echo "GPU status:"
        nvidia-smi
        echo "------------------------------------------"
        python -u video_encoding.py --device cuda
     '
