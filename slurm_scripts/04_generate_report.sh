#!/bin/bash
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --job-name=lstv_report
#SBATCH -o logs/report_%j.out
#SBATCH -e logs/report_%j.err

set -euo pipefail
set -x

echo "================================================================"
echo "LSTV Report Generation"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo "================================================================"

# --- Environment ---
export CONDA_PREFIX="${HOME}/mambaforge/envs/nextflow"
export PATH="${CONDA_PREFIX}/bin:$PATH"
unset JAVA_HOME
which singularity
export NXF_SINGULARITY_HOME_MOUNT=true
unset LD_LIBRARY_PATH PYTHONPATH R_LIBS R_LIBS_USER R_LIBS_SITE

# --- Paths ---
PROJECT_DIR="$(pwd)"
DATA_DIR="${PROJECT_DIR}/data/raw/train_images"
SERIES_CSV="${PROJECT_DIR}/data/raw/train_series_descriptions.csv"
SPINEPS_SEG_DIR="${PROJECT_DIR}/results/spineps_segmentation/segmentations"
INFERENCE_DIR="${PROJECT_DIR}/results/centroid_inference"
CSV_PATH="${INFERENCE_DIR}/lstv_uncertainty_metrics.csv"
OUTPUT_HTML="${INFERENCE_DIR}/report.html"

# --- Container ---
export NXF_SINGULARITY_CACHEDIR="${HOME}/singularity_cache"
mkdir -p "$NXF_SINGULARITY_CACHEDIR"
CONTAINER="docker://go2432/lstv-uncertainty:latest"
IMG_PATH="${NXF_SINGULARITY_CACHEDIR}/lstv-uncertainty.sif"

if [[ ! -f "$IMG_PATH" ]]; then
    echo "Pulling container..."
    singularity pull "$IMG_PATH" "$CONTAINER"
fi

# --- Prerequisites ---
if [[ ! -f "$CSV_PATH" ]]; then
    echo "ERROR: CSV not found: $CSV_PATH"
    echo "Run centroid inference first: sbatch slurm_scripts/03_centroid_inference.sh"
    exit 1
fi

echo "Found CSV with $(tail -n +2 $CSV_PATH | wc -l) studies"

# --- Run report generation ---
echo "Generating HTML report with embedded visualizations..."

singularity exec \
    --bind "$PROJECT_DIR":/work \
    --bind "$DATA_DIR":/data/input \
    --bind "$INFERENCE_DIR":/data/output \
    --bind "$SPINEPS_SEG_DIR":/data/spineps \
    --bind "$(dirname $SERIES_CSV)":/data/raw \
    --pwd /work \
    "$IMG_PATH" \
    python /work/scripts/generate_report.py \
        --csv /data/output/lstv_uncertainty_metrics.csv \
        --output /data/output/report.html \
        --data_dir /data/input \
        --series_csv /data/raw/train_series_descriptions.csv \
        --seg_dir /data/spineps \
        --relabeled_dir /data/output/relabeled_masks

echo ""
echo "================================================================"
echo "Complete! End: $(date)"
echo "================================================================"
echo ""
echo "Report saved to: $OUTPUT_HTML"
echo ""
echo "To view:"
echo "  firefox $OUTPUT_HTML"
echo "  # or copy to local machine:"
echo "  scp warrior:$OUTPUT_HTML ."
echo "================================================================"
