#!/bin/bash
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --job-name=download_valid_ids
#SBATCH -o logs/download_valid_ids_%j.out
#SBATCH -e logs/download_valid_ids_%j.err

set -euo pipefail

echo "================================================================"
echo "Download Validation IDs (valid_id.npy)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo "================================================================"

# Environment setup
export CONDA_PREFIX="${HOME}/mambaforge/envs/nextflow"
export PATH="${CONDA_PREFIX}/bin:$PATH"
unset JAVA_HOME LD_LIBRARY_PATH PYTHONPATH R_LIBS R_LIBS_USER R_LIBS_SITE

export XDG_RUNTIME_DIR="${HOME}/xdr"
export NXF_SINGULARITY_CACHEDIR="${HOME}/singularity_cache"
mkdir -p "$XDG_RUNTIME_DIR" "$NXF_SINGULARITY_CACHEDIR"

export NXF_SINGULARITY_HOME_MOUNT=true

# Project setup
PROJECT_DIR="$(pwd)"
MODELS_DIR="${PROJECT_DIR}/models"
TMP_DIR="${PROJECT_DIR}/.tmp_download"
mkdir -p "$MODELS_DIR" "$TMP_DIR"

# Cleanup function
cleanup() {
    rm -rf "$TMP_DIR"
    rm -rf "${PROJECT_DIR}/.kaggle_tmp"
}
trap cleanup EXIT

# Container setup
DOCKER_USERNAME="go2432"
CONTAINER="docker://${DOCKER_USERNAME}/spineps-segmentation:latest"
IMG_PATH="${NXF_SINGULARITY_CACHEDIR}/spineps-segmentation.sif"

# Check Kaggle credentials
KAGGLE_JSON="${HOME}/.kaggle/kaggle.json"
if [[ ! -f "$KAGGLE_JSON" ]]; then
    echo "ERROR: Kaggle credentials not found at $KAGGLE_JSON"
    echo ""
    echo "Setup instructions:"
    echo "  1. Go to: https://www.kaggle.com/settings/account"
    echo "  2. Click 'Create New Token' under API section"
    echo "  3. Save kaggle.json to ~/.kaggle/"
    echo "  4. Run: chmod 600 ~/.kaggle/kaggle.json"
    exit 1
fi

echo "✓ Kaggle credentials found"

# Pull container if needed
if [[ ! -f "$IMG_PATH" ]]; then
    echo "Pulling container..."
    singularity pull "$IMG_PATH" "$CONTAINER"
fi

echo "✓ Container ready"

# Setup Kaggle credentials for container
mkdir -p "${PROJECT_DIR}/.kaggle_tmp"
cp "${HOME}/.kaggle/kaggle.json" "${PROJECT_DIR}/.kaggle_tmp/"
chmod 600 "${PROJECT_DIR}/.kaggle_tmp/kaggle.json"

echo ""
echo "================================================================"
echo "Downloading valid_id.npy (283 validation study IDs)"
echo "Dataset: rsna2024-demo-workflow (by hengck23)"
echo "File size: ~3 KB"
echo "================================================================"

# Download using Kaggle API
singularity exec \
    --bind "$PROJECT_DIR":/work \
    --bind "${PROJECT_DIR}/.kaggle_tmp":/root/.kaggle \
    --pwd /work \
    "$IMG_PATH" \
    bash -c "
        cd /work/.tmp_download
        
        echo 'Downloading dataset...'
        kaggle datasets download -d hengck23/rsna2024-demo-workflow
        
        echo 'Extracting valid_id.npy...'
        unzip -j rsna2024-demo-workflow.zip valid_id.npy
        
        echo 'Moving to models directory...'
        mv valid_id.npy /work/models/valid_id.npy
        
        echo 'Done!'
    "

exit_code=$?

if [ $exit_code -ne 0 ]; then
    echo "ERROR: Download failed"
    exit $exit_code
fi

# Verify file exists
if [ -f "${MODELS_DIR}/valid_id.npy" ]; then
    echo ""
    echo "================================================================"
    echo "DOWNLOAD COMPLETE"
    echo "End: $(date)"
    echo "================================================================"
    
    ls -lh "${MODELS_DIR}/valid_id.npy"
    
    # Show validation set info
    echo ""
    echo "Validation set information:"
    python3 << PYEOF
import numpy as np
try:
    valid_ids = np.load('${MODELS_DIR}/valid_id.npy')
    print(f"  Total studies: {len(valid_ids)}")
    print(f"  Sample IDs: {list(valid_ids[:5])}")
except Exception as e:
    print(f"  Error loading file: {e}")
PYEOF
    
    echo ""
    echo "✓ Validation IDs ready: ${MODELS_DIR}/valid_id.npy"
    echo ""
    echo "CRITICAL: Pipeline will ONLY process these 283 studies"
    echo "This ensures NO DATA LEAKAGE from training set!"
    echo ""
    echo "Next steps:"
    echo "  1. Run trial segmentation:"
    echo "     sbatch slurm_scripts/01_spineps_trial.sh"
    echo ""
    echo "  2. If trial succeeds, run production:"
    echo "     sbatch slurm_scripts/02_spineps_production.sh"
    echo "================================================================"
else
    echo "ERROR: valid_id.npy not found after download"
    exit 1
fi
