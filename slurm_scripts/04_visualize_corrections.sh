#!/bin/bash
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --job-name=viz_corrections
#SBATCH -o logs/viz_%j.out
#SBATCH -e logs/viz_%j.err

set -euo pipefail

echo "================================================================"
echo "VISUALIZE ANATOMICAL CORRECTIONS"
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
SPINEPS_SEG_DIR="${PROJECT_DIR}/results/spineps_segmentation/segmentations"
PROPAGATION_DIR="${PROJECT_DIR}/results/anatomical_propagation"
OUTPUT_DIR="${PROPAGATION_DIR}/visualizations"

mkdir -p "$OUTPUT_DIR"

# --- Container ---
export NXF_SINGULARITY_CACHEDIR="${HOME}/singularity_cache"
mkdir -p "$NXF_SINGULARITY_CACHEDIR"
CONTAINER="docker://go2432/lstv-uncertainty:latest"
IMG_PATH="${NXF_SINGULARITY_CACHEDIR}/lstv-uncertainty.sif"

if [[ ! -f "$IMG_PATH" ]]; then
    echo "Pulling container..."
    singularity pull "$IMG_PATH" "$CONTAINER"
fi

# --- Run visualization ---
singularity exec \
    --bind "$PROJECT_DIR":/work \
    --bind "$SPINEPS_SEG_DIR":/data/original \
    --bind "$PROPAGATION_DIR":/data/propagation \
    --bind "$OUTPUT_DIR":/data/output \
    --pwd /work \
    "$IMG_PATH" \
    python /work/scripts/visualize_corrections.py \
        --original_dir  /data/original \
        --corrected_dir /data/propagation/anatomically_corrected \
        --reports_dir   /data/propagation/correction_reports \
        --audit_json    /data/propagation/audit_queue/high_priority_audit.json \
        --output_dir    /data/output

echo ""
echo "================================================================"
echo "Complete! End: $(date)"
echo "================================================================"
echo ""
echo "Visualizations saved to: $OUTPUT_DIR"
echo ""
echo "Next step: Review the comparison images to verify corrections"
echo "================================================================"
