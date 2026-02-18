#!/bin/bash
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --job-name=download_model
#SBATCH -o logs/download_model_%j.out
#SBATCH -e logs/download_model_%j.err

set -euo pipefail
set -x

echo "================================================================"
echo "Download Point Net Model Checkpoint"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "================================================================"

# Environment setup
export CONDA_PREFIX="${HOME}/mambaforge/envs/nextflow"
export PATH="${CONDA_PREFIX}/bin:$PATH"
unset JAVA_HOME

which singularity || echo "WARNING: singularity not found"

export XDG_RUNTIME_DIR="${HOME}/xdr"
export NXF_SINGULARITY_CACHEDIR="${HOME}/singularity_cache"
mkdir -p $XDG_RUNTIME_DIR $NXF_SINGULARITY_CACHEDIR

export NXF_SINGULARITY_HOME_MOUNT=true

# Clean environment
unset LD_LIBRARY_PATH
unset PYTHONPATH
unset R_LIBS
unset R_LIBS_USER
unset R_LIBS_SITE

# Project setup
PROJECT_DIR="$(pwd)"
MODELS_DIR="${PROJECT_DIR}/models"
TMP_DIR="${PROJECT_DIR}/.tmp_download"
mkdir -p "$MODELS_DIR"
mkdir -p "$TMP_DIR"

# Cleanup function
cleanup() {
    rm -rf "$TMP_DIR"
    rm -rf ${PROJECT_DIR}/.kaggle_tmp
}
trap cleanup EXIT

# Container setup
DOCKER_USERNAME="go2432"
CONTAINER="docker://${DOCKER_USERNAME}/lstv-uncertainty:latest"
IMG_PATH="${NXF_SINGULARITY_CACHEDIR}/lstv-uncertainty.sif"

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

echo "Kaggle credentials found: $KAGGLE_JSON"

# Pull container if needed
if [[ ! -f "$IMG_PATH" ]]; then
    echo "Pulling container..."
    singularity pull "$IMG_PATH" "$CONTAINER"
fi

echo "Container ready: $IMG_PATH"

# Setup Kaggle credentials for container
mkdir -p ${PROJECT_DIR}/.kaggle_tmp
cp ${HOME}/.kaggle/kaggle.json ${PROJECT_DIR}/.kaggle_tmp/
chmod 600 ${PROJECT_DIR}/.kaggle_tmp/kaggle.json

echo "================================================================"
echo "Downloading Point Net Model Checkpoint & Validation IDs..."
echo "Dataset: rsna2024-demo-workflow (by hengck23)"
echo "Files: 00002484.pth (130 MB), valid_id.npy"
echo "================================================================"

# Download using Kaggle API
# All operations happen in writable project directory
singularity exec \
    --bind $PROJECT_DIR:/work \
    --bind ${PROJECT_DIR}/.kaggle_tmp:/root/.kaggle \
    --pwd /work \
    "$IMG_PATH" \
    bash -c "
        # Download to writable temp directory
        cd /work/.tmp_download
        kaggle datasets download -d hengck23/rsna2024-demo-workflow
        
        # Extract just the model file and validation IDs
        unzip -j rsna2024-demo-workflow.zip 00002484.pth valid_id.npy
        
        # Move to models directory with proper names
        mv 00002484.pth /work/models/point_net_checkpoint.pth
        mv valid_id.npy /work/models/valid_id.npy
    "

exit_code=$?

if [ $exit_code -ne 0 ]; then
    echo "ERROR: Download failed"
    exit $exit_code
fi

# Verify files exist
if [ -f "${MODELS_DIR}/point_net_checkpoint.pth" ] && [ -f "${MODELS_DIR}/valid_id.npy" ]; then
    echo "================================================================"
    echo "Download complete!"
    echo "End time: $(date)"
    echo "================================================================"
    
    ls -lh ${MODELS_DIR}/point_net_checkpoint.pth
    ls -lh ${MODELS_DIR}/valid_id.npy
    
    echo ""
    echo "Model checkpoint ready at: ${MODELS_DIR}/point_net_checkpoint.pth"
    echo "Validation IDs ready at: ${MODELS_DIR}/valid_id.npy"
    echo ""
    echo "IMPORTANT: Inference will ONLY use validation set studies to avoid data leakage!"
    echo ""
    echo "Next step: Run inference"
    echo "  sbatch slurm_scripts/02_trial_inference.sh"
    echo "================================================================"
else
    echo "ERROR: Files not found after download"
    exit 1
fi
