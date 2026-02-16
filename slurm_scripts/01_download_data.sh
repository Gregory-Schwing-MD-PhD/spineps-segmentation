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

set -euo pipefail

echo "================================================================"
echo "STEP 1: DOWNLOAD DATA SUBSET"
echo "Job ID: $SLURM_JOB_ID"
echo "================================================================"

# Paths
PROJECT_DIR="$(pwd)"
DATA_DIR="${PROJECT_DIR}/data/raw"
MODELS_DIR="${PROJECT_DIR}/models"
mkdir -p "$DATA_DIR" "$MODELS_DIR" logs

# Container
export NXF_SINGULARITY_CACHEDIR="${HOME}/singularity_cache"
export NXF_SINGULARITY_HOME_MOUNT=true
CONTAINER="docker://go2432/lstv-uncertainty:latest"
IMG_PATH="${NXF_SINGULARITY_CACHEDIR}/lstv-uncertainty.sif"

if [[ ! -f "$IMG_PATH" ]]; then
    singularity pull "$IMG_PATH" "$CONTAINER"
fi

# Setup Kaggle
mkdir -p "${PROJECT_DIR}/.kaggle_tmp"
cp "${HOME}/.kaggle/kaggle.json" "${PROJECT_DIR}/.kaggle_tmp/"
chmod 600 "${PROJECT_DIR}/.kaggle_tmp/kaggle.json"

# Download Logic
singularity exec \
    --bind "$PROJECT_DIR":/work \
    --bind "${PROJECT_DIR}/.kaggle_tmp":/root/.kaggle \
    --pwd /work \
    "$IMG_PATH" \
    bash -c "
        set -e
        mkdir -p .tmp_dl
        cd .tmp_dl
        
        echo '--- 1. Downloading Metadata (Valid IDs & CSV) ---'
        kaggle datasets download -d hengck23/rsna2024-demo-workflow
        unzip -j -o rsna2024-demo-workflow.zip valid_id.npy
        mv valid_id.npy /work/models/
        
        kaggle competitions download -c rsna-2024-lumbar-spine-degenerative-classification -f train_series_descriptions.csv
        unzip -o train_series_descriptions.csv.zip
        mv train_series_descriptions.csv /work/data/raw/
        
        echo '--- 2. Downloading & Extracting Validation Images ---'
        kaggle competitions download -c rsna-2024-lumbar-spine-degenerative-classification
        
        python3 -c \"
import zipfile, numpy as np, os
valid_ids = set(str(x) for x in np.load('/work/models/valid_id.npy'))
print(f'Extracting {len(valid_ids)} studies...')
with zipfile.ZipFile('rsna-2024-lumbar-spine-degenerative-classification.zip', 'r') as z:
    to_ext = [f for f in z.namelist() if f.split('/')[1] in valid_ids if f.startswith('train_images/')]
    z.extractall('/work/data/raw', members=to_ext)
\"
    "

rm -rf "${PROJECT_DIR}/.tmp_dl" "${PROJECT_DIR}/.kaggle_tmp"
echo "Download Complete."
