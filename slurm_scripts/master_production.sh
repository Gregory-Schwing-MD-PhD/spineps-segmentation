#!/bin/bash
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --job-name=master_prod
#SBATCH -o logs/master_prod_%j.out
#SBATCH -e logs/master_prod_%j.err

set -euo pipefail

echo "================================================================"
echo "MASTER: PRODUCTION PIPELINE (All Valid Studies)"
echo "Job ID: $SLURM_JOB_ID"
echo "================================================================"

# --- STRICT ENVIRONMENT SETUP ---
export CONDA_PREFIX="${HOME}/mambaforge/envs/nextflow"
export PATH="${CONDA_PREFIX}/bin:$PATH"
unset JAVA_HOME

which singularity || echo "WARNING: singularity not found"
# --------------------------------

PROJECT_DIR="$(pwd)"
DATA_CHECK="${PROJECT_DIR}/data/raw/train_series_descriptions.csv"

# Slurm scripts location
DL_SCRIPT="${PROJECT_DIR}/slurm_scripts/01_download_data.sh"
SEG_SCRIPT="${PROJECT_DIR}/slurm_scripts/02_spineps_segmentation.sh"

DEPENDENCY=""

# 1. CHECK DATA
if [ ! -f "$DATA_CHECK" ]; then
    echo "[!] Data missing. Submitting Download Job..."
    JOB_ID_DL=$(sbatch --parsable "$DL_SCRIPT")
    echo "    -> Download Job: $JOB_ID_DL"
    DEPENDENCY="--dependency=afterok:$JOB_ID_DL"
else
    echo "[✓] Data found. Skipping Download."
fi

# 2. SUBMIT SEGMENTATION (Production Mode)
echo "[*] Submitting Segmentation Job..."
JOB_ID_SEG=$(sbatch --parsable $DEPENDENCY --export=MODE=prod "$SEG_SCRIPT")
echo "    -> Segmentation Job: $JOB_ID_SEG"

echo "================================================================"
echo "Pipeline Submitted."
echo "================================================================"
