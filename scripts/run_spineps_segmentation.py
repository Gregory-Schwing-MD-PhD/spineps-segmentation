#!/usr/bin/env python3
"""
SPINEPS Segmentation Pipeline
Runs SPINEPS on validation set studies and saves ALL outputs

Outputs per study:
  - NIfTI: {study_id}_T2w.nii.gz
  - Instance mask: {study_id}_seg-vert_msk.nii.gz
  - Semantic mask: {study_id}_seg-spine_msk.nii.gz  
  - Sub-region mask: {study_id}_seg-subreg_msk.nii.gz
  - Centroids: {study_id}_ctd.json
  - Metadata: {study_id}_metadata.json
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


def load_valid_ids(valid_ids_path: Path) -> set:
    """Load validation study IDs from .npy file"""
    try:
        valid_ids = np.load(valid_ids_path)
        valid_ids = set([str(id) for id in valid_ids])
        logger.info(f"Loaded {len(valid_ids)} validation study IDs")
        return valid_ids
    except Exception as e:
        logger.error(f"Failed to load valid_ids: {e}")
        return set()


def load_series_descriptions(csv_path: Path) -> pd.DataFrame:
    """Load series descriptions CSV"""
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} series descriptions")
        return df
    except Exception as e:
        logger.error(f"Failed to load series CSV: {e}")
        return None


def select_best_series(study_dir: Path, series_df: pd.DataFrame, study_id: str) -> Path:
    """Select best T2 sagittal series for a study"""
    # Try to use series descriptions
    if series_df is not None:
        study_series = series_df[series_df['study_id'] == int(study_id)]
        if len(study_series) > 0:
            priorities = [
                'Sagittal T2',
                'Sagittal T2/STIR',
                'SAG T2',
                'Sagittal T1',
                'SAG T1'
            ]
            for priority in priorities:
                matching = study_series[
                    study_series['series_description'].str.contains(
                        priority, case=False, na=False
                    )
                ]
                if len(matching) > 0:
                    series_id = str(matching.iloc[0]['series_id'])
                    series_path = study_dir / series_id
                    if series_path.exists():
                        return series_path
    
    # Fallback: use first available series
    series_dirs = sorted([d for d in study_dir.iterdir() if d.is_dir()])
    if series_dirs:
        return series_dirs[0]
    
    return None


def convert_dicom_to_nifti(dicom_dir: Path, output_path: Path) -> Path:
    """Convert DICOM to NIfTI using dcm2niix"""
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        study_id = output_path.stem.replace('_T2w', '')
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
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode != 0:
            logger.error(f"dcm2niix failed: {result.stderr}")
            return None
        
        expected_output = output_path.parent / f"{bids_base}.nii.gz"
        
        if not expected_output.exists():
            # Find any generated file
            nifti_files = sorted(output_path.parent.glob(f"{bids_base}*.nii.gz"))
            if not nifti_files:
                return None
            generated_file = nifti_files[0]
            if generated_file != expected_output:
                shutil.move(str(generated_file), str(expected_output))
        
        return expected_output
        
    except Exception as e:
        logger.error(f"DICOM conversion failed: {e}")
        return None


def run_spineps_segmentation(nifti_path: Path, seg_dir: Path, study_id: str) -> dict:
    """
    Run SPINEPS segmentation and collect ALL outputs
    
    Returns dict with paths to all output files
    """
    try:
        seg_dir.mkdir(parents=True, exist_ok=True)
        
        # Run SPINEPS
        cmd = [
            'bash', '/work/src/screening/spineps_wrapper.sh', 'sample',
            '-i', str(nifti_path),
            '-model_semantic', 't2w',
            '-model_instance', 'instance',
            '-model_labeling', 't2w_labeling',
            '-override_semantic',
            '-override_instance',
            '-override_ctd'
        ]
        
        logger.debug(f"  Running SPINEPS for {study_id}...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode != 0:
            logger.error(f"  SPINEPS failed: {result.stderr}")
            return None
        
        # Collect outputs from derivatives directory
        input_parent = nifti_path.parent
        derivatives_base = input_parent / "derivatives_seg"
        
        if not derivatives_base.exists():
            logger.error(f"  Derivatives directory not found: {derivatives_base}")
            return None
        
        # Expected SPINEPS output files
        outputs = {}
        
        # Instance segmentation (vertebrae labels)
        instance_pattern = f"sub-{study_id}_mod-T2w_seg-vert_msk.nii.gz"
        instance_file = derivatives_base / instance_pattern
        if not instance_file.exists():
            # Try to find any instance file
            instance_files = list(derivatives_base.glob("*_seg-vert_msk.nii.gz"))
            if instance_files:
                instance_file = instance_files[0]
        
        if instance_file.exists():
            output_instance = seg_dir / f"{study_id}_seg-vert_msk.nii.gz"
            shutil.copy(instance_file, output_instance)
            outputs['instance_mask'] = output_instance
            logger.debug(f"  ✓ Instance mask: {output_instance.name}")
        
        # Semantic segmentation (spine regions)
        semantic_pattern = f"sub-{study_id}_mod-T2w_seg-spine_msk.nii.gz"
        semantic_file = derivatives_base / semantic_pattern
        if not semantic_file.exists():
            semantic_files = list(derivatives_base.glob("*_seg-spine_msk.nii.gz"))
            if semantic_files:
                semantic_file = semantic_files[0]
        
        if semantic_file.exists():
            output_semantic = seg_dir / f"{study_id}_seg-spine_msk.nii.gz"
            shutil.copy(semantic_file, output_semantic)
            outputs['semantic_mask'] = output_semantic
            logger.debug(f"  ✓ Semantic mask: {output_semantic.name}")
        
        # Sub-region segmentation
        subreg_pattern = f"sub-{study_id}_mod-T2w_seg-subreg_msk.nii.gz"
        subreg_file = derivatives_base / subreg_pattern
        if not subreg_file.exists():
            subreg_files = list(derivatives_base.glob("*_seg-subreg_msk.nii.gz"))
            if subreg_files:
                subreg_file = subreg_files[0]
        
        if subreg_file.exists():
            output_subreg = seg_dir / f"{study_id}_seg-subreg_msk.nii.gz"
            shutil.copy(subreg_file, output_subreg)
            outputs['subreg_mask'] = output_subreg
            logger.debug(f"  ✓ Sub-region mask: {output_subreg.name}")
        
        # Centroids JSON (SPINEPS native format)
        ctd_pattern = f"sub-{study_id}_mod-T2w_ctd.json"
        ctd_file = derivatives_base / ctd_pattern
        if not ctd_file.exists():
            ctd_files = list(derivatives_base.glob("*_ctd.json"))
            if ctd_files:
                ctd_file = ctd_files[0]
        
        if ctd_file.exists():
            output_ctd = seg_dir / f"{study_id}_ctd.json"
            shutil.copy(ctd_file, output_ctd)
            outputs['centroid_json'] = output_ctd
            logger.debug(f"  ✓ Centroids: {output_ctd.name}")
        
        if not outputs:
            logger.error(f"  No SPINEPS outputs found in {derivatives_base}")
            return None
        
        return outputs
        
    except Exception as e:
        logger.error(f"  SPINEPS segmentation failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None


def save_metadata(study_id: str, outputs: dict, metadata_dir: Path):
    """Save processing metadata for a study"""
    metadata = {
        'study_id': study_id,
        'outputs': {k: str(v) for k, v in outputs.items()},
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    metadata_path = metadata_dir / f"{study_id}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description='SPINEPS Segmentation Pipeline'
    )
    
    parser.add_argument('--input_dir', type=str, required=True,
                       help='DICOM input directory')
    parser.add_argument('--series_csv', type=str, required=True,
                       help='Series descriptions CSV')
    parser.add_argument('--nifti_dir', type=str, required=True,
                       help='Output directory for NIfTI files')
    parser.add_argument('--seg_dir', type=str, required=True,
                       help='Output directory for segmentations')
    parser.add_argument('--metadata_dir', type=str, required=True,
                       help='Output directory for metadata')
    parser.add_argument('--valid_ids', type=str, required=True,
                       help='Path to valid_id.npy (validation set only)')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of studies (for trial mode)')
    parser.add_argument('--mode', type=str, choices=['trial', 'debug', 'prod'],
                       default='prod',
                       help='Execution mode')
    
    args = parser.parse_args()
    
    # Setup paths
    input_dir = Path(args.input_dir)
    nifti_dir = Path(args.nifti_dir)
    seg_dir = Path(args.seg_dir)
    metadata_dir = Path(args.metadata_dir)
    
    nifti_dir.mkdir(parents=True, exist_ok=True)
    seg_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    
    # Load validation IDs
    valid_ids_path = Path(args.valid_ids)
    valid_ids = load_valid_ids(valid_ids_path)
    
    if not valid_ids:
        logger.error("No validation IDs loaded - aborting")
        return 1
    
    # Load series descriptions
    series_csv = Path(args.series_csv)
    series_df = load_series_descriptions(series_csv)
    
    # Get all study directories
    study_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])
    
    # Filter to validation set ONLY
    study_dirs = [d for d in study_dirs if d.name in valid_ids]
    logger.info(f"Filtered to {len(study_dirs)} validation studies")
    
    # Apply limit if specified
    if args.limit:
        study_dirs = study_dirs[:args.limit]
        logger.info(f"Limited to {len(study_dirs)} studies ({args.mode} mode)")
    
    logger.info("="*70)
    logger.info("SPINEPS SEGMENTATION PIPELINE")
    logger.info("="*70)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Studies to process: {len(study_dirs)}")
    logger.info(f"Validation set only: YES")
    logger.info("="*70)
    
    # Process each study
    success_count = 0
    error_count = 0
    
    for study_dir in tqdm(study_dirs, desc="Processing studies"):
        study_id = study_dir.name
        logger.info(f"\n[{study_id}]")
        
        try:
            # Check if already processed
            expected_instance = seg_dir / f"{study_id}_seg-vert_msk.nii.gz"
            if expected_instance.exists():
                logger.info(f"  ✓ Already processed (skipping)")
                success_count += 1
                continue
            
            # Select best series
            series_dir = select_best_series(study_dir, series_df, study_id)
            if series_dir is None:
                logger.warning(f"  ✗ No suitable series found")
                error_count += 1
                continue
            
            logger.info(f"  Series: {series_dir.name}")
            
            # Convert DICOM to NIfTI
            nifti_path = nifti_dir / f"{study_id}_T2w.nii.gz"
            if not nifti_path.exists():
                logger.info(f"  Converting DICOM → NIfTI...")
                nifti_path = convert_dicom_to_nifti(series_dir, nifti_path)
                if nifti_path is None:
                    logger.warning(f"  ✗ DICOM conversion failed")
                    error_count += 1
                    continue
            
            # Run SPINEPS segmentation
            logger.info(f"  Running SPINEPS segmentation...")
            outputs = run_spineps_segmentation(nifti_path, seg_dir, study_id)
            
            if outputs is None:
                logger.warning(f"  ✗ SPINEPS segmentation failed")
                error_count += 1
                continue
            
            # Save metadata
            save_metadata(study_id, outputs, metadata_dir)
            
            logger.info(f"  ✓ Complete ({len(outputs)} outputs)")
            success_count += 1
            
        except KeyboardInterrupt:
            logger.warning("\n⚠ Interrupted by user")
            break
        except Exception as e:
            logger.error(f"  ✗ Error: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            error_count += 1
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("PIPELINE COMPLETE")
    logger.info("="*70)
    logger.info(f"Successfully processed: {success_count}")
    logger.info(f"Errors: {error_count}")
    logger.info(f"Total: {success_count + error_count}")
    logger.info("")
    logger.info("Next step: Extract centroids")
    logger.info(f"  python scripts/extract_centroids_spineps.py \\")
    logger.info(f"    --seg_dir {seg_dir} \\")
    logger.info(f"    --output_dir results/spineps_segmentation/centroids \\")
    logger.info(f"    --mode {args.mode}")
    
    return 0 if error_count == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
