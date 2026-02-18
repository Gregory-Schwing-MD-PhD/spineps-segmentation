#!/bin/bash
#SBATCH -q gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=v100
#SBATCH --time=36:00:00
#SBATCH --job-name=lstv_propagation
#SBATCH -o logs/propagation_%j.out
#SBATCH -e logs/propagation_%j.err

set -euo pipefail
set -x

echo "================================================================"
echo "ANATOMICAL LABEL PROPAGATION PIPELINE"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "================================================================"

nvidia-smi

# --- Singularity setup ---
export SINGULARITY_TMPDIR="/tmp/${USER}_job_${SLURM_JOB_ID}"
export XDG_RUNTIME_DIR="$SINGULARITY_TMPDIR/runtime"
export NXF_SINGULARITY_CACHEDIR="${HOME}/singularity_cache"
mkdir -p "$SINGULARITY_TMPDIR" "$XDG_RUNTIME_DIR" "$NXF_SINGULARITY_CACHEDIR"
trap 'rm -rf "$SINGULARITY_TMPDIR"' EXIT

# --- Environment ---
export CONDA_PREFIX="${HOME}/mambaforge/envs/nextflow"
export PATH="${CONDA_PREFIX}/bin:$PATH"
unset JAVA_HOME
which singularity
export NXF_SINGULARITY_HOME_MOUNT=true
unset LD_LIBRARY_PATH PYTHONPATH R_LIBS R_LIBS_USER R_LIBS_SITE

# --- Paths ---
PROJECT_DIR="$(pwd)"
MODELS_DIR="${PROJECT_DIR}/models"
SPINEPS_NIFTI_DIR="${PROJECT_DIR}/results/spineps_segmentation/nifti"
SPINEPS_SEG_DIR="${PROJECT_DIR}/results/spineps_segmentation/segmentations"
OUTPUT_DIR="${PROJECT_DIR}/results/anatomical_propagation"

mkdir -p "$OUTPUT_DIR/logs"

# --- Container ---
CONTAINER="docker://go2432/lstv-uncertainty:latest"
IMG_PATH="${NXF_SINGULARITY_CACHEDIR}/lstv-uncertainty.sif"
if [[ ! -f "$IMG_PATH" ]]; then
    echo "Pulling container..."
    singularity pull "$IMG_PATH" "$CONTAINER"
fi

# --- Prerequisites ---
echo ""
echo "Checking prerequisites..."
echo "  NIfTI:       $SPINEPS_NIFTI_DIR"
echo "  Centroids:   $SPINEPS_SEG_DIR"
echo "  Models:      $MODELS_DIR"
echo ""

if [[ ! -d "$SPINEPS_NIFTI_DIR" ]]; then
    echo "ERROR: SPINEPS NIfTI not found: $SPINEPS_NIFTI_DIR"
    echo "Run SPINEPS first: sbatch slurm_scripts/02_spineps_segmentation.sh"
    exit 1
fi

N_NIFTI=$(ls "$SPINEPS_NIFTI_DIR"/*_T2w.nii.gz 2>/dev/null | wc -l)
N_CTD=$(ls "$SPINEPS_SEG_DIR"/*_ctd.json 2>/dev/null | wc -l)
N_MASKS=$(ls "$SPINEPS_SEG_DIR"/*_seg-vert_msk.nii.gz 2>/dev/null | wc -l)
echo "✓ Found $N_NIFTI NIfTI files, $N_CTD centroids, $N_MASKS masks"

CHECKPOINT="${MODELS_DIR}/point_net_checkpoint.pth"
if [[ ! -f "$CHECKPOINT" ]]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT"
    echo "Download with: sbatch slurm_scripts/01_download_data.sh"
    exit 1
fi

VALID_IDS="${MODELS_DIR}/valid_id.npy"
if [[ ! -f "$VALID_IDS" ]]; then
    echo "WARNING: valid_id.npy not found — will process ALL studies"
    VALID_IDS=""
fi

echo "================================================================"

# --- Run propagation ---
singularity exec --nv \
    --bind "$PROJECT_DIR":/work \
    --bind "$OUTPUT_DIR":/data/output \
    --bind "$MODELS_DIR":/app/models \
    --bind "$SPINEPS_NIFTI_DIR":/data/nifti \
    --bind "$SPINEPS_SEG_DIR":/data/spineps \
    --pwd /work \
    "$IMG_PATH" \
    python /work/scripts/inference_centroid_propagation.py \
        --nifti_dir    /data/nifti \
        --centroid_dir /data/spineps \
        --seg_dir      /data/spineps \
        --output_dir   /data/output \
        --checkpoint   /app/models/point_net_checkpoint.pth \
        ${VALID_IDS:+--valid_ids /app/models/valid_id.npy} \
        --mode prod

echo ""
echo "================================================================"
echo "Complete! End: $(date)"
echo "================================================================"
echo ""
echo "Outputs:"
echo "  Corrected masks: $OUTPUT_DIR/anatomically_corrected/"
echo "  Reports:         $OUTPUT_DIR/correction_reports/"
echo "  Audit queue:     $OUTPUT_DIR/audit_queue/high_priority_audit.json"
echo "  Metrics CSV:     $OUTPUT_DIR/anatomical_correction_metrics.csv"
echo ""
echo "Next step:"
echo "  sbatch slurm_scripts/04_visualize_corrections.sh"
echo "================================================================"
