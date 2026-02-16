#!/usr/bin/env python3
"""
Download RSNA 2024 Lumbar Spine Dataset from Kaggle

Usage:
    python download_rsna_data.py --output_dir /data/raw
"""

import argparse
import subprocess
import sys
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_kaggle_credentials():
    """Verify Kaggle credentials are configured"""
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_json = kaggle_dir / 'kaggle.json'
    
    if not kaggle_json.exists():
        logger.error("Kaggle credentials not found!")
        logger.error(f"Expected: {kaggle_json}")
        logger.error("")
        logger.error("Please set up Kaggle credentials:")
        logger.error("1. Go to https://www.kaggle.com/settings/account")
        logger.error("2. Create New API Token")
        logger.error("3. Move kaggle.json to ~/.kaggle/")
        logger.error("4. chmod 600 ~/.kaggle/kaggle.json")
        return False
    
    logger.info(f"✓ Kaggle credentials found: {kaggle_json}")
    return True


def download_dataset(output_dir: Path):
    """Download RSNA 2024 dataset from Kaggle"""
    
    competition = "rsna-2024-lumbar-spine-degenerative-classification"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*70)
    logger.info("RSNA 2024 DATASET DOWNLOAD")
    logger.info("="*70)
    logger.info(f"Competition: {competition}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("")
    
    # Download train_images.zip (~190GB)
    logger.info("Downloading train_images.zip (this will take 30-60 minutes)...")
    logger.info("File size: ~190GB")
    
    try:
        cmd = [
            'kaggle', 'competitions', 'download',
            '-c', competition,
            '-f', 'train_images.zip',
            '-p', str(output_dir)
        ]
        
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info("✓ Download complete: train_images.zip")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Download failed: {e}")
        logger.error(f"stdout: {e.stdout}")
        logger.error(f"stderr: {e.stderr}")
        return False
    
    # Download train_series_descriptions.csv
    logger.info("")
    logger.info("Downloading train_series_descriptions.csv...")
    
    try:
        cmd = [
            'kaggle', 'competitions', 'download',
            '-c', competition,
            '-f', 'train_series_descriptions.csv',
            '-p', str(output_dir)
        ]
        
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info("✓ Download complete: train_series_descriptions.csv")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Download failed: {e}")
        return False
    
    # Extract train_images.zip
    logger.info("")
    logger.info("Extracting train_images.zip (this will take 20-30 minutes)...")
    
    zip_file = output_dir / 'train_images.zip'
    
    if not zip_file.exists():
        logger.error(f"ZIP file not found: {zip_file}")
        return False
    
    try:
        cmd = ['unzip', '-q', str(zip_file), '-d', str(output_dir)]
        result = subprocess.run(cmd, check=True, timeout=3600)
        logger.info("✓ Extraction complete")
        
    except subprocess.TimeoutExpired:
        logger.error("Extraction timed out (>1 hour)")
        return False
    except subprocess.CalledProcessError as e:
        logger.error(f"Extraction failed: {e}")
        return False
    
    # Verify extraction
    train_images_dir = output_dir / 'train_images'
    
    if not train_images_dir.exists():
        logger.error(f"Extraction failed: {train_images_dir} not found")
        return False
    
    study_count = len(list(train_images_dir.iterdir()))
    logger.info(f"✓ Found {study_count} studies in train_images/")
    
    logger.info("")
    logger.info("="*70)
    logger.info("DOWNLOAD COMPLETE")
    logger.info("="*70)
    logger.info(f"Dataset location: {output_dir}")
    logger.info(f"Studies: {study_count}")
    logger.info("")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Download RSNA 2024 Lumbar Spine Dataset'
    )
    
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for dataset')
    
    args = parser.parse_args()
    
    # Check credentials
    if not check_kaggle_credentials():
        return 1
    
    # Download dataset
    output_dir = Path(args.output_dir)
    
    success = download_dataset(output_dir)
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
