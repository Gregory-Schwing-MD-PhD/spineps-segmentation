#!/usr/bin/bash
#SBATCH -q gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=v100
#SBATCH --time=02:00:00
#SBATCH --job-name=spineps_trial
#SBATCH -o logs/spineps_trial_%j.out
#SBATCH -e logs/spineps_trial_%j.err

set -euo pipefail

# ============================================================================
# SPINEPS SEGMENTATION - TRIAL MODE
# ============================================================================
# Processes 3 studies from valid_id.npy
# Assumes data already downloaded
# ============================================================================

echo "================================================================"
echo "SPINEPS SEGMENTATION - TRIAL MODE"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "================================================================"

nvidia-smi

# ============================================================================
# ENVIRONMENT
# ============================================================================

export SINGULARITY_TMPDIR="/tmp/${USER}_job_${SLURM_JOB_ID}"
export XDG_RUNTIME_DIR="$SINGULARITY_TMPDIR/runtime"
export NXF_SINGULARITY_CACHEDIR="${HOME}/singularity_cache"
mkdir -p "$SINGULARITY_TMPDIR" "$XDG_RUNTIME_DIR" "$NXF_SINGULARITY_CACHEDIR"

trap 'rm -rf "$SINGULARITY_TMPDIR"' EXIT

export CONDA_PREFIX="${HOME}/mambaforge/envs/nextflow"
export PATH="${CONDA_PREFIX}/bin:$PATH"
unset JAVA_HOME LD_LIBRARY_PATH PYTHONPATH R_LIBS R_LIBS_USER R_LIBS_SITE

export NXF_SINGULARITY_HOME_MOUNT=true

# ============================================================================
# PATHS
# ============================================================================

PROJECT_DIR="$(pwd)"
DATA_DIR="${PROJECT_DIR}/data/raw/train_images"
SERIES_CSV="${PROJECT_DIR}/data/raw/train_series_descriptions.csv"
VALID_IDS="${PROJECT_DIR}/models/valid_id.npy"

# Output structure
OUTPUT_BASE="${PROJECT_DIR}/results/spineps_segmentation"
NIFTI_DIR="${OUTPUT_BASE}/nifti"
SEG_DIR="${OUTPUT_BASE}/segmentations"
CENTROID_DIR="${OUTPUT_BASE}/centroids"
METADATA_DIR="${OUTPUT_BASE}/metadata"

mkdir -p "$OUTPUT_BASE" "$NIFTI_DIR" "$SEG_DIR" "$CENTROID_DIR" "$METADATA_DIR"
mkdir -p logs

MODELS_CACHE="${PROJECT_DIR}/models/spineps_cache"
mkdir -p "$MODELS_CACHE"

# Container
DOCKER_USERNAME="go2432"
CONTAINER="docker://${DOCKER_USERNAME}/spineps-segmentation:latest"
IMG_PATH="${NXF_SINGULARITY_CACHEDIR}/spineps-segmentation.sif"

if [[ ! -f "$IMG_PATH" ]]; then
    echo "Pulling SPINEPS container..."
    singularity pull "$IMG_PATH" "$CONTAINER"
fi

echo "✓ Container: $IMG_PATH"
echo "✓ Data: $DATA_DIR"
echo "✓ Output: $OUTPUT_BASE"
echo "✓ Mode: TRIAL (3 studies from valid_id.npy)"

# ============================================================================
# STEP 1: SPINEPS SEGMENTATION
# ============================================================================

echo ""
echo "================================================================"
echo "STEP 1: SPINEPS SEGMENTATION"
echo "================================================================"

singularity exec --nv \
    --bind "$PROJECT_DIR":/work \
    --bind "$DATA_DIR":/data/input \
    --bind "$OUTPUT_BASE":/data/output \
    --bind "$MODELS_CACHE":/app/models \
    --bind "$(dirname $SERIES_CSV)":/data/raw \
    --env SPINEPS_SEGMENTOR_MODELS=/app/models \
    --env SPINEPS_ENVIRONMENT_DIR=/app/models \
    --pwd /work \
    "$IMG_PATH" \
    python /work/scripts/run_spineps_segmentation.py \
        --input_dir /data/input \
        --series_csv /data/raw/train_series_descriptions.csv \
        --nifti_dir /data/output/nifti \
        --seg_dir /data/output/segmentations \
        --metadata_dir /data/output/metadata \
        --valid_ids /work/models/valid_id.npy \
        --limit 3 \
        --mode trial

segmentation_exit=$?

if [ $segmentation_exit -ne 0 ]; then
    echo "ERROR: SPINEPS segmentation failed"
    exit $segmentation_exit
fi

echo "✓ SPINEPS segmentation complete"

# ============================================================================
# STEP 2: EXTRACT CENTROIDS
# ============================================================================

echo ""
echo "================================================================"
echo "STEP 2: CENTROID EXTRACTION"
echo "================================================================"

singularity exec \
    --bind "$PROJECT_DIR":/work \
    --bind "$SEG_DIR":/data/segmentations \
    --bind "$CENTROID_DIR":/data/centroids \
    --pwd /work \
    "$IMG_PATH" \
    python /work/scripts/extract_centroids.py \
        --seg_dir /data/segmentations \
        --output_dir /data/centroids \
        --mode trial

centroid_exit=$?

if [ $centroid_exit -ne 0 ]; then
    echo "ERROR: Centroid extraction failed"
    exit $centroid_exit
fi

echo "✓ Centroid extraction complete"

# ============================================================================
# SUMMARY
# ============================================================================

echo ""
echo "================================================================"
echo "PIPELINE COMPLETE"
echo "================================================================"
echo "End: $(date)"
echo ""
echo "Output structure:"
echo "  NIfTI files:       ${NIFTI_DIR}/"
echo "  Segmentations:     ${SEG_DIR}/"
echo "  Centroids:         ${CENTROID_DIR}/"
echo "  Metadata:          ${METADATA_DIR}/"
echo ""
echo "Files created (3 studies):"
echo "  {study_id}_T2w.nii.gz"
echo "  {study_id}_seg-vert_msk.nii.gz        (instance segmentation)"
echo "  {study_id}_seg-spine_msk.nii.gz       (semantic segmentation)"
echo "  {study_id}_seg-subreg_msk.nii.gz      (sub-region masks)"
echo "  {study_id}_ctd.json                    (SPINEPS centroids)"
echo "  {study_id}_centroids.json              (fusion centroids)"
echo "  {study_id}_vertebra_labels.json        (label mapping)"
echo ""
echo "Verification:"
echo "  ls -lh ${SEG_DIR}/*_seg-vert_msk.nii.gz | wc -l"
echo "  ls -lh ${CENTROID_DIR}/*_centroids.json | wc -l"
echo ""
echo "Next: Pass centroids to lstv-uncertainty-detection pipeline"
echo "  --centroid_dir ${CENTROID_DIR}"
echo "================================================================"
