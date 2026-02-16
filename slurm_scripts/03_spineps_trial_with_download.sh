#!/usr/bin/bash
#SBATCH -q gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=v100
#SBATCH --time=04:00:00
#SBATCH --job-name=spineps_trial_dl
#SBATCH -o logs/spineps_trial_dl_%j.out
#SBATCH -e logs/spineps_trial_dl_%j.err

set -euo pipefail

echo "================================================================"
echo "SPINEPS SEGMENTATION - TRIAL MODE (WITH DOWNLOAD)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo "================================================================"

nvidia-smi

# Environment setup (same as 01_spineps_trial.sh)
export SINGULARITY_TMPDIR="/tmp/${USER}_job_${SLURM_JOB_ID}"
export XDG_RUNTIME_DIR="$SINGULARITY_TMPDIR/runtime"
export NXF_SINGULARITY_CACHEDIR="${HOME}/singularity_cache"
mkdir -p "$SINGULARITY_TMPDIR" "$XDG_RUNTIME_DIR" "$NXF_SINGULARITY_CACHEDIR"

trap 'rm -rf "$SINGULARITY_TMPDIR"' EXIT

export CONDA_PREFIX="${HOME}/mambaforge/envs/nextflow"
export PATH="${CONDA_PREFIX}/bin:$PATH"
unset JAVA_HOME LD_LIBRARY_PATH PYTHONPATH R_LIBS R_LIBS_USER R_LIBS_SITE
export NXF_SINGULARITY_HOME_MOUNT=true

PROJECT_DIR="$(pwd)"
DATA_DIR="${PROJECT_DIR}/data/raw"

# ============================================================================
# STEP 0: DOWNLOAD DATA
# ============================================================================

echo ""
echo "================================================================"
echo "STEP 0: DATA DOWNLOAD"
echo "================================================================"

if [[ ! -d "${DATA_DIR}/train_images" ]] || [[ $(ls -A "${DATA_DIR}/train_images" 2>/dev/null | wc -l) -eq 0 ]]; then
    echo "Downloading RSNA dataset..."
    
    python scripts/download_rsna_data.py --output_dir "$DATA_DIR"
    
    if [ $? -ne 0 ]; then
        echo "ERROR: Data download failed"
        exit 1
    fi
else
    echo "✓ Data already exists, skipping download"
fi

# ============================================================================
# STEP 1-2: Run SPINEPS segmentation + centroid extraction
# ============================================================================

# Use same steps as 01_spineps_trial.sh
# (Implementation identical to trial script)

echo "Proceeding to segmentation..."
bash slurm_scripts/01_spineps_trial.sh

echo "================================================================"
echo "COMPLETE"
echo "================================================================"
