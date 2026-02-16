#!/usr/bin/env python3
"""
Extract Centroids from SPINEPS Segmentations
Processes instance masks and creates centroid JSON files for fusion pipeline

Output per study:
  - {study_id}_centroids.json       (3D coordinates + anatomical labels)
  - {study_id}_vertebra_labels.json (instance ID → anatomical label mapping)
"""

import argparse
import json
import numpy as np
import nibabel as nib
from pathlib import Path
from scipy.ndimage import center_of_mass
from tqdm import tqdm
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# SPINEPS anatomical label mapping
SPINEPS_LABELS = {
    1: 'C1', 2: 'C2', 3: 'C3', 4: 'C4', 5: 'C5', 6: 'C6', 7: 'C7',
    8: 'T1', 9: 'T2', 10: 'T3', 11: 'T4', 12: 'T5', 13: 'T6',
    14: 'T7', 15: 'T8', 16: 'T9', 17: 'T10', 18: 'T11', 19: 'T12',
    20: 'L1', 21: 'L2', 22: 'L3', 23: 'L4', 24: 'L5', 25: 'L6',
    26: 'Sacrum',
    # Discs (100 + vertebra ID)
    119: 'T12_L1_disc',
    120: 'L1_L2_disc', 121: 'L2_L3_disc', 122: 'L3_L4_disc',
    123: 'L4_L5_disc', 124: 'L5_S1_disc', 125: 'L6_S1_disc',
    126: 'S1_S2_disc',
}


def extract_centroids_and_labels(seg_path: Path) -> tuple:
    """
    Extract centroids and anatomical labels from SPINEPS instance segmentation
    
    Args:
        seg_path: Path to *_seg-vert_msk.nii.gz file
        
    Returns:
        centroids (dict): Centroid data for each structure
        label_mapping (dict): Instance ID → anatomical label
    """
    # Load segmentation
    seg_nii = nib.load(seg_path)
    seg_data = seg_nii.get_fdata().astype(int)
    affine = seg_nii.affine
    
    unique_labels = np.unique(seg_data)
    unique_labels = unique_labels[unique_labels > 0]  # Exclude background
    
    logger.debug(f"Loaded: {seg_path.name}")
    logger.debug(f"  Shape: {seg_data.shape}")
    logger.debug(f"  Unique labels: {len(unique_labels)} - {list(unique_labels[:10])}")
    
    centroids = {}
    label_mapping = {}
    
    # Extract all structures (vertebrae, discs, sacrum)
    for instance_id in unique_labels:
        if instance_id not in SPINEPS_LABELS:
            logger.debug(f"  Skipping unknown label: {instance_id}")
            continue
        
        mask = (seg_data == instance_id)
        
        if mask.sum() == 0:
            continue
        
        # Calculate centroid in voxel space
        centroid_voxel = center_of_mass(mask)
        
        # Transform to world coordinates
        centroid_voxel_homo = np.array([*centroid_voxel, 1.0])
        centroid_world = (affine @ centroid_voxel_homo)[:3]
        
        # Volume
        volume_voxels = int(mask.sum())
        
        # Anatomical label
        anatomical_label = SPINEPS_LABELS[instance_id]
        
        # Key naming: vertebra_X for vertebrae, disc_X for discs
        if instance_id <= 26:
            structure_key = f"vertebra_{instance_id}"
        else:
            structure_key = f"disc_{instance_id}"
        
        centroids[structure_key] = {
            'instance_id': int(instance_id),
            'anatomical_label': anatomical_label,
            'centroid_voxel': [float(c) for c in centroid_voxel],  # [i, j, k]
            'centroid_world': [float(c) for c in centroid_world],  # [x, y, z] mm
            'volume_voxels': volume_voxels
        }
        
        label_mapping[int(instance_id)] = anatomical_label
    
    logger.debug(f"  Extracted {len(centroids)} structures")
    
    return centroids, label_mapping


def save_json(data: dict, output_path: Path):
    """Save data as JSON"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    logger.debug(f"  Saved: {output_path}")


def process_all_studies(seg_dir: Path, output_dir: Path, mode: str = 'prod'):
    """
    Process all SPINEPS segmentations to extract centroids
    
    Expected input:
      seg_dir/
        ├── {study_id}_seg-vert_msk.nii.gz
        └── ...
    
    Output:
      output_dir/
        ├── {study_id}_centroids.json
        ├── {study_id}_vertebra_labels.json
        └── ...
    """
    logger.info("="*70)
    logger.info("SPINEPS CENTROID EXTRACTION")
    logger.info("="*70)
    logger.info(f"Segmentation directory: {seg_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Mode: {mode}")
    
    # Find all SPINEPS instance segmentation files
    seg_files = sorted(seg_dir.glob("*_seg-vert_msk.nii.gz"))
    
    if len(seg_files) == 0:
        logger.error("No segmentation files found!")
        logger.error(f"Expected: {seg_dir}/*_seg-vert_msk.nii.gz")
        logger.error("Ensure SPINEPS segmentation completed successfully")
        return
    
    logger.info(f"Found {len(seg_files)} segmentation files")
    
    # Select based on mode
    if mode == 'trial':
        seg_files = seg_files[:3]
        logger.info(f"TRIAL mode: Processing {len(seg_files)} files")
    elif mode == 'debug':
        seg_files = seg_files[:1]
        logger.info(f"DEBUG mode: Processing {len(seg_files)} file")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each study
    success_count = 0
    error_count = 0
    
    for seg_file in tqdm(seg_files, desc="Extracting centroids"):
        # Extract study ID from filename
        # Format: {study_id}_seg-vert_msk.nii.gz
        study_id = seg_file.stem.replace("_seg-vert_msk", "")
        
        try:
            # Extract centroids and labels
            centroids, label_mapping = extract_centroids_and_labels(seg_file)
            
            if len(centroids) == 0:
                logger.warning(f"No structures found in {study_id}")
                error_count += 1
                continue
            
            # Save centroids
            centroid_path = output_dir / f"{study_id}_centroids.json"
            save_json(centroids, centroid_path)
            
            # Save label mapping
            label_path = output_dir / f"{study_id}_vertebra_labels.json"
            save_json(label_mapping, label_path)
            
            success_count += 1
            
        except Exception as e:
            logger.error(f"Error processing {study_id}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            error_count += 1
    
    # Summary
    logger.info("="*70)
    logger.info("SUMMARY")
    logger.info("="*70)
    logger.info(f"Successfully processed: {success_count}")
    logger.info(f"Errors: {error_count}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("")
    logger.info("Files created per study:")
    logger.info("  {study_id}_centroids.json       - 3D coordinates + labels")
    logger.info("  {study_id}_vertebra_labels.json - Instance ID → label")
    logger.info("")
    logger.info("Next step: Run lstv-uncertainty-detection with --centroid_dir")


def main():
    parser = argparse.ArgumentParser(
        description='Extract centroids from SPINEPS segmentations'
    )
    
    parser.add_argument('--seg_dir', type=str, required=True,
                       help='Directory with SPINEPS instance masks (*_seg-vert_msk.nii.gz)')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for centroids')
    parser.add_argument('--mode', type=str, choices=['trial', 'debug', 'prod'],
                       default='prod',
                       help='Processing mode')
    
    args = parser.parse_args()
    
    seg_dir = Path(args.seg_dir)
    output_dir = Path(args.output_dir)
    
    if not seg_dir.exists():
        logger.error(f"Segmentation directory not found: {seg_dir}")
        return 1
    
    process_all_studies(seg_dir, output_dir, args.mode)
    return 0


if __name__ == '__main__':
    exit(main())
