#!/bin/bash
#SBATCH -q gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=v100
#SBATCH --time=12:00:00
#SBATCH --job-name=spineps_run
#SBATCH -o logs/spineps_%j.out
#SBATCH -e logs/spineps_%j.err

set -euo pipefail

# Mode is passed via --export=MODE=... from master script
MODE=${MODE:-production}

echo "================================================================"
echo "WORKER: SPINEPS SEGMENTATION"
echo "Mode: $MODE"
echo "Job ID: $SLURM_JOB_ID | GPU: $CUDA_VISIBLE_DEVICES"
echo "================================================================"

# --- STRICT ENVIRONMENT SETUP ---
export CONDA_PREFIX="${HOME}/mambaforge/envs/nextflow"
export PATH="${CONDA_PREFIX}/bin:$PATH"
unset JAVA_HOME

which singularity || echo "WARNING: singularity not found"

export XDG_RUNTIME_DIR="${HOME}/xdr"
export NXF_SINGULARITY_CACHEDIR="${HOME}/singularity_cache"
mkdir -p $XDG_RUNTIME_DIR $NXF_SINGULARITY_CACHEDIR

export NXF_SINGULARITY_HOME_MOUNT=true
unset LD_LIBRARY_PATH
unset PYTHONPATH
unset R_LIBS
unset R_LIBS_USER
unset R_LIBS_SITE
# --------------------------------

PROJECT_DIR="$(pwd)"
mkdir -p logs

CONTAINER="docker://go2432/spineps-segmentation:latest"
IMG_PATH="${NXF_SINGULARITY_CACHEDIR}/spineps-segmentation.sif"

if [[ ! -f "$IMG_PATH" ]]; then
    singularity pull "$IMG_PATH" "$CONTAINER"
fi

# Execution
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

echo "Segmentation Complete."
