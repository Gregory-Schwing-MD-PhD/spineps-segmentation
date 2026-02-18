#!/usr/bin/env python3
"""
SPINEPS Segmentation Pipeline with Disc Centroid Computation

This version automatically computes disc centroids after SPINEPS runs,
so the *_ctd.json files contain BOTH vertebrae AND disc centroids.
"""

import argparse
import json
import subprocess
import shutil
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.ndimage import center_of_mass
from tqdm import tqdm
import logging

try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# DISC CENTROID COMPUTATION (THE MISSING PIECE)
# ============================================================================

def compute_and_add_disc_centroids(seg_file: Path, ctd_path: Path) -> int:
    """
    Compute disc centroids from instance mask and add to SPINEPS centroid JSON.
    Returns number of disc centroids added.
    """
    if not HAS_NIBABEL:
        logger.warning("nibabel not available - skipping disc centroid computation")
        return 0
    
    try:
        # Load instance mask
        seg_nii = nib.load(seg_file)
        seg_data = seg_nii.get_fdata().astype(int)
        
        # Find disc instances (119-126)
        disc_centroids = {}
        for disc_label in range(119, 127):
            mask = (seg_data == disc_label)
            if mask.sum() > 0:
                centroid_voxel = center_of_mass(mask)
                disc_centroids[disc_label] = list(centroid_voxel)
        
        if not disc_centroids:
            return 0
        
        # Load existing SPINEPS centroid JSON
        with open(ctd_path) as f:
            data = json.load(f)
        
        if len(data) < 2:
            logger.warning(f"Unexpected centroid JSON structure: {ctd_path}")
            return 0
        
        # Add disc centroids to POI data (second element)
        for disc_label, coords in disc_centroids.items():
            data[1][str(disc_label)] = {'50': coords}
        
        # Save updated JSON
        with open(ctd_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        return len(disc_centroids)
    
    except Exception as e:
        logger.warning(f"Error computing disc centroids: {e}")
        return 0


# ============================================================================
# PROGRESS TRACKING
# ============================================================================

def load_progress(progress_file: Path) -> dict:
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
            files = sorted(output_path.parent.glob(f"{bids_base}*.nii.gz"))
            if not files:
                return None
            if files[0] != expected:
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
    try:
        seg_dir.mkdir(parents=True, exist_ok=True)

        import os
        env = os.environ.copy()
        env['SPINEPS_SEGMENTOR_MODELS'] = '/app/models'
        env['SPINEPS_ENVIRONMENT_DIR'] = '/app/models'

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
        result = subprocess.run(cmd, stdout=None, stderr=subprocess.PIPE, text=True, timeout=600, env=env)
        sys.stdout.flush()

        if result.returncode != 0:
            logger.error(f"  SPINEPS failed:\n{result.stderr}")
            return None

        derivatives_base = nifti_path.parent / "derivatives_seg"
        if not derivatives_base.exists():
            logger.error(f"  derivatives_seg not found at: {derivatives_base}")
            return None

        def find_file(exact_name: str, glob_pattern: str) -> Path:
            f = derivatives_base / exact_name
            if not f.exists():
                matches = list(derivatives_base.glob(glob_pattern))
                f = matches[0] if matches else None
            return f if (f and f.exists()) else None

        outputs = {}

        # Instance segmentation
        f = find_file(f"sub-{study_id}_mod-T2w_seg-vert_msk.nii.gz", "*_seg-vert_msk.nii.gz")
        if f:
            dest = seg_dir / f"{study_id}_seg-vert_msk.nii.gz"
            shutil.copy(f, dest)
            outputs['instance_mask'] = dest
            logger.info("  ✓ Instance mask (seg-vert)")
        else:
            logger.warning("  ⚠ Instance mask not found")

        # Semantic segmentation
        f = find_file(f"sub-{study_id}_mod-T2w_seg-spine_msk.nii.gz", "*_seg-spine_msk.nii.gz")
        if f:
            dest = seg_dir / f"{study_id}_seg-spine_msk.nii.gz"
            shutil.copy(f, dest)
            outputs['semantic_mask'] = dest
            logger.info("  ✓ Semantic mask (seg-spine)")

        # Sub-region segmentation
        f = find_file(f"sub-{study_id}_mod-T2w_seg-subreg_msk.nii.gz", "*_seg-subreg_msk.nii.gz")
        if f:
            dest = seg_dir / f"{study_id}_seg-subreg_msk.nii.gz"
            shutil.copy(f, dest)
            outputs['subreg_mask'] = dest
            logger.info("  ✓ Sub-region mask (seg-subreg)")

        # Centroids JSON
        f = find_file(f"sub-{study_id}_mod-T2w_ctd.json", "*_ctd.json")
        if f:
            dest = seg_dir / f"{study_id}_ctd.json"
            shutil.copy(f, dest)
            outputs['centroid_json'] = dest
            logger.info("  ✓ Centroids JSON (ctd)")
            
            # **NEW: Compute and add disc centroids**
            if 'instance_mask' in outputs:
                n_discs = compute_and_add_disc_centroids(outputs['instance_mask'], dest)
                if n_discs > 0:
                    logger.info(f"  ✓ Added {n_discs} disc centroids to JSON")

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
    parser.add_argument('--input_dir',  required=True)
    parser.add_argument('--series_csv', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--valid_ids',  required=True)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--mode', choices=['trial', 'debug', 'prod'], default='prod')
    args = parser.parse_args()

    output_dir   = Path(args.output_dir)
    nifti_dir    = output_dir / 'nifti'
    seg_dir      = output_dir / 'segmentations'
    metadata_dir = output_dir / 'metadata'
    progress_file = output_dir / 'progress.json'

    for d in [nifti_dir, seg_dir, metadata_dir]:
        d.mkdir(parents=True, exist_ok=True)

    progress = load_progress(progress_file)
    already_processed = set(progress['processed'])

    valid_ids = load_valid_ids(Path(args.valid_ids))
    if not valid_ids:
        logger.error("No validation IDs loaded")
        return 1

    series_df = load_series_descriptions(Path(args.series_csv))

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

    remaining = [d for d in study_dirs if d.name not in already_processed]
    skipped   = len(study_dirs) - len(remaining)

    logger.info("=" * 70)
    logger.info("SPINEPS SEGMENTATION PIPELINE (WITH DISC CENTROIDS)")
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

            outputs = run_spineps(nifti_path, seg_dir, study_id)
            if outputs is None:
                logger.warning("  ✗ SPINEPS failed")
                progress['processed'].append(study_id)
                progress['failed'].append(study_id)
                save_progress(progress_file, progress)
                error_count += 1
                continue

            save_metadata(study_id, outputs, metadata_dir)
            progress['processed'].append(study_id)
            progress['success'].append(study_id)
            save_progress(progress_file, progress)
            success_count += 1
            logger.info(f"  ✓ Done ({len(outputs)} outputs)")
            sys.stdout.flush()

        except KeyboardInterrupt:
            logger.warning("\n⚠ Interrupted — progress saved")
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
        logger.info(f"Failed IDs: {progress['failed']}")
    logger.info(f"Progress: {progress_file}")
    logger.info("")
    logger.info("Centroid JSONs now contain BOTH vertebrae AND disc centroids!")
    logger.info("Next: sbatch slurm_scripts/03_anatomical_propagation.sh")
    sys.stdout.flush()

    return 0 if error_count == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
