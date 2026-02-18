#!/bin/bash
#SBATCH -q gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=v100
#SBATCH --time=36:00:00
#SBATCH --job-name=lstv_centroid
#SBATCH -o logs/centroid_%j.out
#SBATCH -e logs/centroid_%j.err
#SBATCH --mail-user=go2432@wayne.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail
set -x

echo "================================================================"
echo "LSTV Centroid-Guided Uncertainty Inference"
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

# --- Paths (all in one repo now!) ---
PROJECT_DIR="$(pwd)"
DATA_DIR="${PROJECT_DIR}/data/raw/train_images"
SERIES_CSV="${PROJECT_DIR}/data/raw/train_series_descriptions.csv"
MODELS_DIR="${PROJECT_DIR}/models"
SPINEPS_SEG_DIR="${PROJECT_DIR}/results/spineps_segmentation/segmentations"
OUTPUT_DIR="${PROJECT_DIR}/results/centroid_inference"

mkdir -p "$OUTPUT_DIR/logs" "$OUTPUT_DIR/relabeled_masks" "$OUTPUT_DIR/audit_queue"

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
echo "  Data:          $DATA_DIR"
echo "  Series CSV:    $SERIES_CSV"
echo "  SPINEPS segs:  $SPINEPS_SEG_DIR"
echo "  Models:        $MODELS_DIR"
echo ""

if [[ ! -d "$DATA_DIR" ]]; then
    echo "ERROR: DICOM data not found: $DATA_DIR"
    exit 1
fi

if [[ ! -f "$SERIES_CSV" ]]; then
    echo "ERROR: Series CSV not found: $SERIES_CSV"
    exit 1
fi

if [[ ! -d "$SPINEPS_SEG_DIR" ]]; then
    echo "ERROR: SPINEPS segmentations not found: $SPINEPS_SEG_DIR"
    echo "Run SPINEPS first: sbatch slurm_scripts/02_spineps_segmentation.sh"
    exit 1
fi

N_CTD=$(ls "$SPINEPS_SEG_DIR"/*_ctd.json 2>/dev/null | wc -l)
N_MASKS=$(ls "$SPINEPS_SEG_DIR"/*_seg-vert_msk.nii.gz 2>/dev/null | wc -l)
echo "✓ Found $N_CTD centroid files, $N_MASKS instance masks"

CHECKPOINT="${MODELS_DIR}/point_net_checkpoint.pth"
if [[ ! -f "$CHECKPOINT" ]]; then
    echo "WARNING: Checkpoint not found — will run in MOCK mode"
    echo "Download with: sbatch slurm_scripts/01b_download_model.sh"
fi

echo "================================================================"

# --- Run inference ---
singularity exec --nv \
    --bind "$PROJECT_DIR":/work \
    --bind "$DATA_DIR":/data/input \
    --bind "$OUTPUT_DIR":/data/output \
    --bind "$MODELS_DIR":/app/models \
    --bind "$(dirname $SERIES_CSV)":/data/raw \
    --bind "$SPINEPS_SEG_DIR":/data/spineps \
    --pwd /work \
    "$IMG_PATH" \
    python /work/scripts/inference_centroid.py \
        --input_dir    /data/input \
        --series_csv   /data/raw/train_series_descriptions.csv \
        --centroid_dir /data/spineps \
        --seg_dir      /data/spineps \
        --output_dir   /data/output \
        --checkpoint   /app/models/point_net_checkpoint.pth \
        --valid_ids    /app/models/valid_id.npy \
        --mode prod

echo ""
echo "================================================================"
echo "Complete! End: $(date)"
echo "================================================================"
echo ""
echo "Outputs:"
echo "  CSV:         $OUTPUT_DIR/lstv_uncertainty_metrics.csv"
echo "  Re-labeled:  $OUTPUT_DIR/relabeled_masks/"
echo "  Audit queue: $OUTPUT_DIR/audit_queue/high_priority_audit.json"
echo ""
echo "Next step:"
echo "  sbatch slurm_scripts/04_generate_report.sh"
echo "================================================================"
