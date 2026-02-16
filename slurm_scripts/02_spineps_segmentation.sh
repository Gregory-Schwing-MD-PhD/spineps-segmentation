#!/bin/bash
#SBATCH -q gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=v100
#SBATCH --time=12:00:00
#SBATCH --job-name=spineps
#SBATCH -o logs/spineps_%j.out
#SBATCH -e logs/spineps_%j.err

set -euo pipefail
MODE=${MODE:-production}

echo "================================================================"
echo "STEP 2: SPINEPS SEGMENTATION ($MODE)"
echo "Job: $SLURM_JOB_ID | GPU: $CUDA_VISIBLE_DEVICES"
echo "================================================================"

# Environment
export SINGULARITY_TMPDIR="/tmp/${USER}_job_${SLURM_JOB_ID}"
export NXF_SINGULARITY_CACHEDIR="${HOME}/singularity_cache"
export NXF_SINGULARITY_HOME_MOUNT=true
mkdir -p "$SINGULARITY_TMPDIR" "$NXF_SINGULARITY_CACHEDIR" logs
trap 'rm -rf "$SINGULARITY_TMPDIR"' EXIT

# Paths
PROJECT_DIR="$(pwd)"
CONTAINER="docker://go2432/spineps-segmentation:latest"
IMG_PATH="${NXF_SINGULARITY_CACHEDIR}/spineps-segmentation.sif"

if [[ ! -f "$IMG_PATH" ]]; then
    singularity pull "$IMG_PATH" "$CONTAINER"
fi

# Run
singularity exec --nv \
    --bind "$PROJECT_DIR":/work \
    --pwd /work \
    "$IMG_PATH" \
    python /work/scripts/run_spineps_segmentation.py \
        --input_dir /work/data/raw/train_images \
        --series_csv /work/data/raw/train_series_descriptions.csv \
        --output_dir /work/results/spineps_segmentation \
        --valid_ids /work/models/valid_id.npy \
        --mode "$MODE"
