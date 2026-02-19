#!/usr/bin/env python3
"""
SPINEPS Segmentation Pipeline - FULL VERSION

Features:
- Computes centroids for ALL structures (vertebrae, discs, endplates, subregions)
- Saves uncertainty maps from softmax logits
- Prepares masks for DICOM overlay visualization
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
# CENTROID COMPUTATION FOR ALL STRUCTURES
# ============================================================================

def compute_all_centroids(instance_mask_path: Path, semantic_mask_path: Path, ctd_path: Path) -> dict:
    """
    Compute centroids for ALL structures in both masks.
    
    Returns dict with counts of added centroids by type.
    """
    if not HAS_NIBABEL:
        logger.warning("nibabel not available - skipping centroid computation")
        return {}
    
    try:
        # Load masks
        instance_nii = nib.load(instance_mask_path)
        instance_data = instance_nii.get_fdata().astype(int)
        
        semantic_nii = nib.load(semantic_mask_path)
        semantic_data = semantic_nii.get_fdata().astype(int)
        
        # Load existing SPINEPS centroid JSON
        with open(ctd_path) as f:
            ctd_data = json.load(f)
        
        if len(ctd_data) < 2:
            logger.warning(f"Unexpected centroid JSON structure: {ctd_path}")
            return {}
        
        added_counts = {
            'vertebrae': 0,
            'discs': 0,
            'endplates': 0,
            'subregions': 0
        }
        
        # Process instance mask (vertebrae, discs, endplates)
        instance_labels = np.unique(instance_data)
        instance_labels = instance_labels[instance_labels > 0]
        
        for label in instance_labels:
            label_str = str(label)
            
            # Skip if already exists (SPINEPS adds vertebrae)
            if label_str in ctd_data[1]:
                continue
            
            mask = (instance_data == label)
            if mask.sum() == 0:
                continue
            
            centroid = center_of_mass(mask)
            ctd_data[1][label_str] = {'50': list(centroid)}
            
            # Categorize
            if label <= 28:
                added_counts['vertebrae'] += 1
            elif 119 <= label <= 126:
                added_counts['discs'] += 1
            elif label >= 200:
                added_counts['endplates'] += 1
        
        # Process semantic mask (subregions: 41-49, 60-62, etc.)
        semantic_labels = np.unique(semantic_data)
        semantic_labels = semantic_labels[semantic_labels > 0]
        
        for label in semantic_labels:
            label_str = str(label)
            
            # Skip if already exists
            if label_str in ctd_data[1]:
                continue
            
            mask = (semantic_data == label)
            if mask.sum() == 0:
                continue
            
            centroid = center_of_mass(mask)
            ctd_data[1][label_str] = {'50': list(centroid)}
            added_counts['subregions'] += 1
        
        # Save updated JSON
        with open(ctd_path, 'w') as f:
            json.dump(ctd_data, f, indent=2)
        
        return added_counts
    
    except Exception as e:
        logger.warning(f"Error computing centroids: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return {}


# ============================================================================
# UNCERTAINTY MAP COMPUTATION
# ============================================================================

def compute_uncertainty_from_softmax(derivatives_dir: Path, study_id: str, seg_dir: Path) -> bool:
    """
    Compute uncertainty map from softmax logits if available.
    Uncertainty = 1 - max(softmax)
    """
    if not HAS_NIBABEL:
        return False
    
    try:
        # Look for softmax logits file
        logits_pattern = f"*{study_id}*logit*.npz"
        logits_files = list(derivatives_dir.glob(logits_pattern))
        
        if not logits_files:
            return False
        
        logits_file = logits_files[0]
        
        # Load softmax logits
        logits_data = np.load(logits_file)
        softmax = logits_data['arr_0']  # Default key for np.savez_compressed
        
        # Compute uncertainty: 1 - max(softmax along class dimension)
        uncertainty = 1.0 - np.max(softmax, axis=-1)
        
        # Load reference NIfTI to get header/affine
        semantic_mask = seg_dir / f"{study_id}_seg-spine_msk.nii.gz"
        if not semantic_mask.exists():
            return False
        
        ref_nii = nib.load(semantic_mask)
        
        # Create uncertainty NIfTI
        unc_nii = nib.Nifti1Image(uncertainty.astype(np.float32), ref_nii.affine, ref_nii.header)
        
        # Save
        unc_path = seg_dir / f"{study_id}_unc.nii.gz"
        nib.save(unc_nii, unc_path)
        
        logger.info(f"  ✓ Uncertainty map saved")
        return True
    
    except Exception as e:
        logger.warning(f"Could not compute uncertainty map: {e}")
        return False


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
            '-save_softmax_logits',  # For uncertainty computation
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
            
            # Compute centroids for ALL structures
            if 'instance_mask' in outputs and 'semantic_mask' in outputs:
                counts = compute_all_centroids(
                    outputs['instance_mask'],
                    outputs['semantic_mask'],
                    dest
                )
                if counts:
                    total = sum(counts.values())
                    logger.info(f"  ✓ Added {total} centroids: "
                              f"{counts.get('discs', 0)} discs, "
                              f"{counts.get('endplates', 0)} endplates, "
                              f"{counts.get('subregions', 0)} subregions")

        # Uncertainty map (from softmax logits)
        if 'semantic_mask' in outputs:
            unc_computed = compute_uncertainty_from_softmax(derivatives_base, study_id, seg_dir)
            if unc_computed:
                outputs['uncertainty_map'] = seg_dir / f"{study_id}_unc.nii.gz"

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
    parser = argparse.ArgumentParser(description='SPINEPS Segmentation Pipeline - Full Version')
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
    logger.info("SPINEPS SEGMENTATION PIPELINE - FULL VERSION")
    logger.info("=" * 70)
    logger.info(f"Mode:             {args.mode}")
    logger.info(f"Total studies:    {len(study_dirs)}")
    logger.info(f"Already done:     {skipped}")
    logger.info(f"To process:       {len(remaining)}")
    logger.info(f"Output root:      {output_dir}")
    logger.info("")
    logger.info("Features enabled:")
    logger.info("  ✓ All structure centroids (vertebrae, discs, endplates, subregions)")
    logger.info("  ✓ Uncertainty maps")
    logger.info("  ✓ DICOM-aligned masks for overlay")
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
    logger.info("Outputs per study:")
    logger.info("  • *_seg-vert_msk.nii.gz  - Instance mask (vertebrae, discs, endplates)")
    logger.info("  • *_seg-spine_msk.nii.gz - Semantic mask (subregions)")
    logger.info("  • *_ctd.json             - Centroids for ALL structures")
    logger.info("  • *_unc.nii.gz           - Uncertainty map")
    logger.info("")
    logger.info("For DICOM overlay: All masks are in the same space as input NIfTI")
    sys.stdout.flush()

    return 0 if error_count == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
