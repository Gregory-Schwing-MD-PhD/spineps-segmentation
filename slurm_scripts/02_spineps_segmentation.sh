#!/bin/bash
#SBATCH -q gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=v100
#SBATCH --time=36:00:00
#SBATCH --job-name=spineps_run
#SBATCH -o logs/spineps_%j.out
#SBATCH -e logs/spineps_%j.err

set -euo pipefail

MODE=${MODE:-prod}

echo "================================================================"
echo "WORKER: SPINEPS SEGMENTATION"
echo "Mode: $MODE"
echo "Job ID: $SLURM_JOB_ID | GPU: $CUDA_VISIBLE_DEVICES"
echo "================================================================"

export CONDA_PREFIX="${HOME}/mambaforge/envs/nextflow"
export PATH="${CONDA_PREFIX}/bin:$PATH"
unset JAVA_HOME
which singularity || echo "WARNING: singularity not found"
export XDG_RUNTIME_DIR="${HOME}/xdr"
export NXF_SINGULARITY_CACHEDIR="${HOME}/singularity_cache"
mkdir -p $XDG_RUNTIME_DIR $NXF_SINGULARITY_CACHEDIR
export NXF_SINGULARITY_HOME_MOUNT=true
unset LD_LIBRARY_PATH PYTHONPATH R_LIBS R_LIBS_USER R_LIBS_SITE

# ============================================================================
# PATHS
# ============================================================================

PROJECT_DIR="$(pwd)"
DATA_DIR="${PROJECT_DIR}/data/raw/train_images"
SERIES_CSV="${PROJECT_DIR}/data/raw/train_series_descriptions.csv"
OUTPUT_DIR="${PROJECT_DIR}/results/spineps_segmentation"
NIFTI_DIR="${OUTPUT_DIR}/nifti"
MODELS_CACHE="${PROJECT_DIR}/models/spineps_cache"

mkdir -p logs "$OUTPUT_DIR" "$NIFTI_DIR" "$MODELS_CACHE"

CONTAINER="docker://go2432/spineps-segmentation:latest"
IMG_PATH="${NXF_SINGULARITY_CACHEDIR}/spineps-segmentation.sif"
if [[ ! -f "$IMG_PATH" ]]; then
    singularity pull "$IMG_PATH" "$CONTAINER"
fi

# ============================================================================
# EXECUTION
# ============================================================================

singularity exec --nv \
    --bind "$PROJECT_DIR":/work \
    --bind "$DATA_DIR":/data/input \
    --bind "$OUTPUT_DIR":/data/output \
    --bind "$NIFTI_DIR":/data/output/nifti \
    --bind "$MODELS_CACHE":/app/models \
    --bind "$(dirname $SERIES_CSV)":/data/raw \
    --env SPINEPS_SEGMENTOR_MODELS=/app/models \
    --env SPINEPS_ENVIRONMENT_DIR=/app/models \
    --pwd /work \
    "$IMG_PATH" \
    python /work/scripts/run_spineps_segmentation.py \
        --input_dir  /data/input \
        --series_csv /data/raw/train_series_descriptions.csv \
        --output_dir /data/output \
        --valid_ids  /work/models/valid_id.npy \
        --mode "$MODE"

echo "Segmentation Complete."
