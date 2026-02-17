#!/usr/bin/env python3
"""
SPINEPS Segmentation Pipeline
Runs SPINEPS on validation set studies and saves ALL outputs

Usage:
    python run_spineps_segmentation.py \
        --input_dir /work/data/raw/train_images \
        --series_csv /work/data/raw/train_series_descriptions.csv \
        --output_dir /work/results/spineps_segmentation \
        --valid_ids /work/models/valid_id.npy \
        --mode trial

Output structure under --output_dir:
    nifti/         {study_id}_T2w.nii.gz
    segmentations/ {study_id}_seg-vert_msk.nii.gz   (instance)
                   {study_id}_seg-spine_msk.nii.gz   (semantic)
                   {study_id}_seg-subreg_msk.nii.gz  (sub-region)
                   {study_id}_ctd.json               (SPINEPS centroids)
    metadata/      {study_id}_metadata.json
    progress.json  (resume state - processed/failed/success counts)
"""

import argparse
import json
import subprocess
import shutil
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)



# ============================================================================
# PROGRESS TRACKING  (enables resume after failed jobs)
# ============================================================================

def load_progress(progress_file: Path) -> dict:
    """Load progress from previous run, or start fresh."""
    if progress_file.exists():
        try:
            with open(progress_file, 'r') as f:
                progress = json.load(f)
            logger.info(
                f"Resuming: {len(progress['success'])} done, "
                f"{len(progress['failed'])} failed previously"
            )
            return progress
        except Exception as e:
            logger.warning(f"Could not load progress file: {e} — starting fresh")
    return {'processed': [], 'success': [], 'failed': []}


def save_progress(progress_file: Path, progress: dict):
    """Atomically save progress (tmp → rename so a crash can't corrupt it)."""
    try:
        tmp = progress_file.with_suffix('.json.tmp')
        with open(tmp, 'w') as f:
            json.dump(progress, f, indent=2)
        tmp.replace(progress_file)
    except Exception as e:
        logger.warning(f"Could not save progress: {e}")


# ============================================================================
# DATA LOADING
# ============================================================================

def load_valid_ids(valid_ids_path: Path) -> set:
    try:
        valid_ids = np.load(valid_ids_path)
        valid_ids = set([str(id) for id in valid_ids])
        logger.info(f"Loaded {len(valid_ids)} validation study IDs")
        return valid_ids
    except Exception as e:
        logger.error(f"Failed to load valid_ids: {e}")
        return set()


def load_series_descriptions(csv_path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} series descriptions")
        return df
    except Exception as e:
        logger.error(f"Failed to load series CSV: {e}")
        return None


def select_best_series(study_dir: Path, series_df: pd.DataFrame, study_id: str) -> Path:
    if series_df is not None:
        try:
            study_series = series_df[series_df['study_id'] == int(study_id)]
            if len(study_series) > 0:
                priorities = ['Sagittal T2', 'Sagittal T2/STIR', 'SAG T2', 'Sagittal T1', 'SAG T1']
                for priority in priorities:
                    matching = study_series[
                        study_series['series_description'].str.contains(priority, case=False, na=False)
                    ]
                    if len(matching) > 0:
                        series_id = str(matching.iloc[0]['series_id'])
                        series_path = study_dir / series_id
                        if series_path.exists():
                            return series_path
        except Exception:
            pass

    series_dirs = sorted([d for d in study_dir.iterdir() if d.is_dir()])
    return series_dirs[0] if series_dirs else None


# ============================================================================
# DICOM → NIFTI
# ============================================================================

def convert_dicom_to_nifti(dicom_dir: Path, output_path: Path, study_id: str) -> Path:
    """
    Convert DICOM series to NIfTI using dcm2niix.
    study_id is passed explicitly to avoid deriving it from the filename
    (Path.stem on a .nii.gz returns 'name.nii', not 'name').
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        bids_base = f"sub-{study_id}_T2w"

        cmd = [
            'dcm2niix',
            '-z', 'y',
            '-f', bids_base,
            '-o', str(output_path.parent),
            '-m', 'y',
            '-b', 'n',
            str(dicom_dir)
        ]
        result = subprocess.run(cmd, stdout=None, stderr=subprocess.PIPE, text=True, timeout=120)
        sys.stdout.flush()

        if result.returncode != 0:
            logger.error(f"dcm2niix failed: {result.stderr}")
            return None

        expected = output_path.parent / f"{bids_base}.nii.gz"
        if not expected.exists():
            # dcm2niix sometimes appends extra suffixes; take first match
            files = sorted(output_path.parent.glob(f"{bids_base}*.nii.gz"))
            if not files:
                return None
            if files[0] != expected:
                # Remove any stale file at expected path before renaming
                if expected.exists():
                    expected.unlink()
                shutil.move(str(files[0]), str(expected))

        return expected

    except Exception as e:
        logger.error(f"DICOM conversion failed: {e}")
        return None


# ============================================================================
# SPINEPS
# ============================================================================

def run_spineps(nifti_path: Path, seg_dir: Path, study_id: str) -> dict:
    """
    Run SPINEPS via python -m spineps.entrypoint (the CLI binary is broken).
    Env vars passed directly to subprocess so they exist at import time —
    spineps/utils/filepaths.py tries to mkdir its models dir on import,
    before any wrapper script could set them.
    Returns dict of {output_type: Path} or None on failure.
    """
    try:
        seg_dir.mkdir(parents=True, exist_ok=True)

        # Pass env vars via subprocess env so they exist before spineps imports
        import os
        env = os.environ.copy()
        env['SPINEPS_SEGMENTOR_MODELS'] = '/app/models'
        env['SPINEPS_ENVIRONMENT_DIR'] = '/app/models'

        # Call python -m spineps.entrypoint directly — no wrapper needed
        cmd = [
            'python', '-m', 'spineps.entrypoint', 'sample',
            '-i', str(nifti_path),
            '-model_semantic',  't2w',
            '-model_instance',  'instance',
            '-model_labeling',  't2w_labeling',
            '-override_semantic',
            '-override_instance',
            '-override_ctd'
        ]

        logger.info("  Running SPINEPS...")
        sys.stdout.flush()
        # stderr=PIPE captures errors for logging; stdout streams live to SLURM log
        result = subprocess.run(cmd, stdout=None, stderr=subprocess.PIPE, text=True, timeout=600, env=env)
        sys.stdout.flush()

        if result.returncode != 0:
            logger.error(f"  SPINEPS failed:\n{result.stderr}")
            return None

        # SPINEPS writes to derivatives_seg/ next to the input NIfTI
        derivatives_base = nifti_path.parent / "derivatives_seg"
        if not derivatives_base.exists():
            logger.error(f"  derivatives_seg not found at: {derivatives_base}")
            return None

        def find_file(exact_name: str, glob_pattern: str) -> Path:
            """Try exact name first, fall back to glob."""
            f = derivatives_base / exact_name
            if not f.exists():
                matches = list(derivatives_base.glob(glob_pattern))
                f = matches[0] if matches else None
            return f if (f and f.exists()) else None

        outputs = {}

        # Instance segmentation — vertebra-level labels (L1=20 … L6=25, Sacrum=26)
        f = find_file(f"sub-{study_id}_mod-T2w_seg-vert_msk.nii.gz", "*_seg-vert_msk.nii.gz")
        if f:
            dest = seg_dir / f"{study_id}_seg-vert_msk.nii.gz"
            shutil.copy(f, dest)
            outputs['instance_mask'] = dest
            logger.info("  ✓ Instance mask (seg-vert)")
        else:
            logger.warning("  ⚠ Instance mask not found — SPINEPS may have failed silently")

        # Semantic segmentation — broad region labels
        f = find_file(f"sub-{study_id}_mod-T2w_seg-spine_msk.nii.gz", "*_seg-spine_msk.nii.gz")
        if f:
            dest = seg_dir / f"{study_id}_seg-spine_msk.nii.gz"
            shutil.copy(f, dest)
            outputs['semantic_mask'] = dest
            logger.info("  ✓ Semantic mask (seg-spine)")

        # Sub-region segmentation — endplates, bodies, spinous processes, etc.
        f = find_file(f"sub-{study_id}_mod-T2w_seg-subreg_msk.nii.gz", "*_seg-subreg_msk.nii.gz")
        if f:
            dest = seg_dir / f"{study_id}_seg-subreg_msk.nii.gz"
            shutil.copy(f, dest)
            outputs['subreg_mask'] = dest
            logger.info("  ✓ Sub-region mask (seg-subreg)")

        # Centroids JSON (SPINEPS native format)
        f = find_file(f"sub-{study_id}_mod-T2w_ctd.json", "*_ctd.json")
        if f:
            dest = seg_dir / f"{study_id}_ctd.json"
            shutil.copy(f, dest)
            outputs['centroid_json'] = dest
            logger.info("  ✓ Centroids JSON (ctd)")

        # Must have at least the instance mask to be useful downstream
        if 'instance_mask' not in outputs:
            logger.error("  Instance mask missing — treating as failure")
            return None

        return outputs

    except subprocess.TimeoutExpired:
        logger.error("  SPINEPS timed out (>600s)")
        sys.stdout.flush()
        return None
    except Exception as e:
        logger.error(f"  SPINEPS error: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        sys.stdout.flush()
        return None


# ============================================================================
# METADATA
# ============================================================================

def save_metadata(study_id: str, outputs: dict, metadata_dir: Path):
    metadata = {
        'study_id':  study_id,
        'outputs':   {k: str(v) for k, v in outputs.items()},
        'timestamp': pd.Timestamp.now().isoformat()
    }
    with open(metadata_dir / f"{study_id}_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='SPINEPS Segmentation Pipeline')
    parser.add_argument('--input_dir',  required=True, help='DICOM input directory (train_images/)')
    parser.add_argument('--series_csv', required=True, help='train_series_descriptions.csv')
    parser.add_argument('--output_dir', required=True,
                        help='Root output dir. Subdirs nifti/, segmentations/, metadata/ created automatically.')
    parser.add_argument('--valid_ids',  required=True, help='Path to valid_id.npy')
    parser.add_argument('--limit', type=int, default=None, help='Cap number of studies')
    parser.add_argument('--mode', choices=['trial', 'debug', 'prod'], default='prod')
    args = parser.parse_args()

    # All subdirectories derived from --output_dir
    output_dir   = Path(args.output_dir)
    nifti_dir    = output_dir / 'nifti'
    seg_dir      = output_dir / 'segmentations'
    metadata_dir = output_dir / 'metadata'
    progress_file = output_dir / 'progress.json'

    for d in [nifti_dir, seg_dir, metadata_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Load progress for resume support
    progress = load_progress(progress_file)
    already_processed = set(progress['processed'])

    # Load inputs
    valid_ids = load_valid_ids(Path(args.valid_ids))
    if not valid_ids:
        logger.error("No validation IDs loaded — aborting")
        return 1

    series_df = load_series_descriptions(Path(args.series_csv))

    # Filter to validation set only, then apply mode/limit
    input_dir  = Path(args.input_dir)
    study_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])
    study_dirs = [d for d in study_dirs if d.name in valid_ids]
    logger.info(f"Filtered to {len(study_dirs)} validation studies")

    if args.mode == 'debug':
        study_dirs = study_dirs[:1]
    elif args.mode == 'trial':
        study_dirs = study_dirs[:3]
    elif args.limit:
        study_dirs = study_dirs[:args.limit]

    # Exclude already-processed studies (resume support)
    remaining = [d for d in study_dirs if d.name not in already_processed]
    skipped   = len(study_dirs) - len(remaining)

    logger.info("=" * 70)
    logger.info("SPINEPS SEGMENTATION PIPELINE")
    logger.info("=" * 70)
    logger.info(f"Mode:             {args.mode}")
    logger.info(f"Total studies:    {len(study_dirs)}")
    logger.info(f"Already done:     {skipped}")
    logger.info(f"To process:       {len(remaining)}")
    logger.info(f"Output root:      {output_dir}")
    logger.info("=" * 70)
    sys.stdout.flush()

    success_count = len(progress['success'])
    error_count   = len(progress['failed'])

    for study_dir in tqdm(remaining, desc="Studies"):
        study_id = study_dir.name
        logger.info(f"\n[{study_id}]")
        sys.stdout.flush()

        try:
            # Series selection
            series_dir = select_best_series(study_dir, series_df, study_id)
            if series_dir is None:
                logger.warning("  ✗ No suitable series found")
                progress['processed'].append(study_id)
                progress['failed'].append(study_id)
                save_progress(progress_file, progress)
                error_count += 1
                continue
            logger.info(f"  Series: {series_dir.name}")
            sys.stdout.flush()

            # DICOM → NIfTI
            # Pass study_id explicitly — do NOT derive from filename stem
            # (Path.stem on .nii.gz returns 'name.nii', not 'name')
            nifti_path = nifti_dir / f"{study_id}_T2w.nii.gz"
            if not nifti_path.exists():
                logger.info("  Converting DICOM → NIfTI...")
                sys.stdout.flush()
                nifti_path = convert_dicom_to_nifti(series_dir, nifti_path, study_id)
                if nifti_path is None:
                    logger.warning("  ✗ DICOM conversion failed")
                    progress['processed'].append(study_id)
                    progress['failed'].append(study_id)
                    save_progress(progress_file, progress)
                    error_count += 1
                    continue

            # SPINEPS segmentation
            outputs = run_spineps(nifti_path, seg_dir, study_id)
            if outputs is None:
                logger.warning("  ✗ SPINEPS failed")
                progress['processed'].append(study_id)
                progress['failed'].append(study_id)
                save_progress(progress_file, progress)
                error_count += 1
                continue

            # Save metadata and mark success
            save_metadata(study_id, outputs, metadata_dir)
            progress['processed'].append(study_id)
            progress['success'].append(study_id)
            save_progress(progress_file, progress)
            success_count += 1
            logger.info(f"  ✓ Done ({len(outputs)} outputs)")
            sys.stdout.flush()

        except KeyboardInterrupt:
            logger.warning("\n⚠ Interrupted — progress saved, safe to resubmit")
            sys.stdout.flush()
            break
        except Exception as e:
            logger.error(f"  ✗ Unexpected error: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            progress['processed'].append(study_id)
            progress['failed'].append(study_id)
            save_progress(progress_file, progress)
            error_count += 1
            sys.stdout.flush()

    logger.info("\n" + "=" * 70)
    logger.info("DONE")
    logger.info("=" * 70)
    logger.info(f"Success:  {success_count}")
    logger.info(f"Failed:   {error_count}")
    logger.info(f"Total:    {success_count + error_count}")
    if progress['failed']:
        logger.info(f"Failed study IDs: {progress['failed']}")
    logger.info(f"Progress saved: {progress_file}")
    logger.info("")
    logger.info("Next step:")
    logger.info(f"  python scripts/extract_centroids.py \\")
    logger.info(f"    --seg_dir {seg_dir} \\")
    logger.info(f"    --output_dir {output_dir}/centroids \\")
    logger.info(f"    --mode {args.mode}")
    sys.stdout.flush()

    return 0 if error_count == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
