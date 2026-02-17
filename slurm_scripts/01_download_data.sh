#!/bin/bash
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --job-name=dl_rsna
#SBATCH -o logs/dl_rsna_%j.out
#SBATCH -e logs/dl_rsna_%j.err
#SBATCH --mail-user=go2432@wayne.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

echo "================================================================"
echo "WORKER: DOWNLOAD DATA SUBSET"
echo "Job ID: $SLURM_JOB_ID"
echo "================================================================"

# --- ENVIRONMENT SETUP ---
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
# -------------------------

PROJECT_DIR="$(pwd)"
DATA_DIR="${PROJECT_DIR}/data/raw"
MODELS_DIR="${PROJECT_DIR}/models"
# MOUNT FIX: Create the persistent download dir on HOST first
mkdir -p "$DATA_DIR" "$MODELS_DIR" logs "${PROJECT_DIR}/.tmp_dl"

# --- KAGGLE CHECK ---
KAGGLE_JSON="${HOME}/.kaggle/kaggle.json"
if [[ ! -f "$KAGGLE_JSON" ]]; then
    echo "ERROR: Kaggle credentials not found at $KAGGLE_JSON"
    exit 1
fi
echo "Kaggle credentials found: $KAGGLE_JSON"

# --- CONTAINER SETUP ---
CONTAINER="docker://go2432/spineps-preprocessing:latest"
IMG_PATH="${NXF_SINGULARITY_CACHEDIR}/spineps-preprocessing.sif"

if [[ ! -f "$IMG_PATH" ]]; then
    singularity pull "$IMG_PATH" "$CONTAINER"
fi

# --- PREPARE KAGGLE MOUNT ---
mkdir -p ${PROJECT_DIR}/.kaggle_tmp
cp ${HOME}/.kaggle/kaggle.json ${PROJECT_DIR}/.kaggle_tmp/
chmod 600 ${PROJECT_DIR}/.kaggle_tmp/kaggle.json

echo "Starting download logic..."

singularity exec \
    --bind $PROJECT_DIR:/work \
    --bind ${PROJECT_DIR}/.kaggle_tmp:/root/.kaggle \
    --pwd /work \
    "$IMG_PATH" \
    bash -c "
        set -e
        cd .tmp_dl

        echo '--- 1. Downloading Metadata ---'
        
        # IDEMPOTENCE FIX: Check if workflow zip exists
        if [ ! -f rsna2024-demo-workflow.zip ]; then
            kaggle datasets download -d hengck23/rsna2024-demo-workflow
        else
            echo 'Metadata zip found, skipping download.'
        fi
        
        unzip -j -o rsna2024-demo-workflow.zip valid_id.npy
        mv valid_id.npy /work/models/

        echo 'Downloading CSV...'
        kaggle competitions download -c rsna-2024-lumbar-spine-degenerative-classification -f train_series_descriptions.csv

        if [ -f 'train_series_descriptions.csv.zip' ]; then
            echo 'Unzipping CSV...'
            unzip -o train_series_descriptions.csv.zip
        elif [ -f 'train_series_descriptions.csv' ]; then
            echo 'CSV downloaded uncompressed (skipping unzip).'
        else
            echo 'ERROR: CSV file not found after download.'
            ls -lh
            exit 1
        fi

        mv train_series_descriptions.csv /work/data/raw/

        echo '--- 2. Downloading & Extracting Images ---'
        
        # Define variable inside inner shell
        ZIP_FILE='rsna-2024-lumbar-spine-degenerative-classification.zip'
        
        # Note: Escaped \$ZIP_FILE so outer shell doesn't expand it
        if [ ! -f \"\$ZIP_FILE\" ]; then
            echo 'Zip not found, downloading...'
            kaggle competitions download -c rsna-2024-lumbar-spine-degenerative-classification
        else
            echo 'Zip found in .tmp_dl, skipping download.'
        fi

        python3 -c \"
import zipfile, numpy as np, os
valid_ids = set(str(x) for x in np.load('/work/models/valid_id.npy'))
print(f'Extracting {len(valid_ids)} studies...')

# BUG FIX: Escaped the dollar sign here: \\\$ZIP_FILE
with zipfile.ZipFile('\$ZIP_FILE', 'r') as z:
    to_ext = [f for f in z.namelist() if f.startswith('train_images/') and len(f.split('/')) > 1 and f.split('/')[1] in valid_ids]
    z.extractall('/work/data/raw', members=to_ext)
\"
    "

exit_code=$?

# Cleanup Kaggle creds, BUT KEEP .tmp_dl
rm -rf ${PROJECT_DIR}/.kaggle_tmp

if [ $exit_code -ne 0 ]; then
    echo "ERROR: Download failed"
    exit $exit_code
fi

echo "Download complete!"
