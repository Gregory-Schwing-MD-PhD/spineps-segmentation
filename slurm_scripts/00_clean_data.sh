#!/bin/bash
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --job-name=clean_data
#SBATCH -o logs/clean_%j.out
#SBATCH -e logs/clean_%j.err

set -euo pipefail

echo "================================================================"
echo "CLEAN DATA DIRECTORY"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo "================================================================"

PROJECT_DIR="$(pwd)"
DATA_DIR="${PROJECT_DIR}/data/raw"

echo ""
echo "WARNING: This will DELETE all data in:"
echo "  $DATA_DIR/train_images/"
echo ""

if [[ -d "${DATA_DIR}/train_images" ]]; then
    echo "Calculating current size..."
    du -sh "${DATA_DIR}/train_images"
    
    echo ""
    echo "Deleting..."
    rm -rf "${DATA_DIR}/train_images"
    
    echo "✓ Deleted ${DATA_DIR}/train_images"
else
    echo "Directory doesn't exist, nothing to delete."
fi

echo ""
echo "================================================================"
echo "CLEAN COMPLETE"
echo "================================================================"
echo ""
echo "Next step:"
echo "  sbatch slurm_scripts/01_download_data.sh"
echo "================================================================"
