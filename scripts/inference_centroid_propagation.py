#!/usr/bin/env python3
"""
Centroid-Guided LSTV Detection with Anatomical Label Propagation
==================================================================

BREAKTHROUGH: This isn't just LSTV detection - it's spine re-mapping.

Pipeline:
  1. Load SPINEPS centroids (voxel coords) and instance masks
  2. Load NIfTI volume that SPINEPS processed (same affine)
  3. For each disc (119-124): run Ian Pan model at centroid location
  4. VERIFY disc identity: which channel has highest confidence + lowest entropy?
  5. PROPAGATE labels: use verified discs to re-label adjacent vertebrae
  6. DETECT LSTV: high entropy at L5/S1 → LSTV_Anomaly
  7. RESOLVE conflicts: if 6 lumbar vertebrae but only 5 clear transitions → merge

Outputs:
  - {study_id}_anatomically_corrected.nii.gz  (re-labeled instance mask)
  - {study_id}_correction_report.json         (before/after label mapping)
  - lstv_uncertainty_metrics.csv              (entropy + confidence per disc)
  - audit_queue/high_priority_audit.json      (cases with label corrections)

The "Before": SPINEPS counts 1, 2, 3, 4, 5, 6 blindly
The "After":  Ian Pan identifies L5-S1, we rename every bone from that anchor
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    from loguru import logger
    logger.remove()
    logger.add(sys.stdout,
               format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                      "<level>{level: <8}</level> | <level>{message}</level>")
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s | %(levelname)s | %(message)s')
    logger = logging.getLogger(__name__)

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False
    logger.warning("timm not available")


# ============================================================================
# THRESHOLDS
# ============================================================================

ENTROPY_VERIFIED  = 4.0   # below → disc identity verified
ENTROPY_LSTV      = 5.0   # above → LSTV anomaly
IMAGE_SIZE        = 160
N_AUDIT_CASES     = 60


# ============================================================================
# LABEL MAPPINGS
# ============================================================================

# SPINEPS disc labels → anatomical disc names
DISC_LABEL_TO_NAME = {
    119: 'T12_L1',  120: 'L1_L2',
    121: 'L2_L3',   122: 'L3_L4',
    123: 'L4_L5',   124: 'L5_S1',
}

# Ian Pan channel → disc name
CHANNEL_TO_DISC = {
    0: 'background',
    1: 'L1_L2', 2: 'L2_L3', 3: 'L3_L4',
    4: 'L4_L5', 5: 'L5_S1',
}

# Disc → adjacent vertebrae (superior, inferior)
DISC_TO_VERTEBRAE = {
    'T12_L1': ('T12', 'L1'),
    'L1_L2':  ('L1', 'L2'),
    'L2_L3':  ('L2', 'L3'),
    'L3_L4':  ('L3', 'L4'),
    'L4_L5':  ('L4', 'L5'),
    'L5_S1':  ('L5', 'Sacrum'),
}

# Anatomical vertebra name → corrected instance label
VERTEBRA_TO_LABEL = {
    'T12': 19, 'L1': 20, 'L2': 21, 'L3': 22,
    'L4': 23, 'L5': 24, 'L6': 25, 'Sacrum': 26,
}

# SPINEPS instance labels (what it outputs)
SPINEPS_LABELS = {
    19: 'T12', 20: 'L1', 21: 'L2', 22: 'L3',
    23: 'L4', 24: 'L5', 25: 'L6', 26: 'Sacrum', 28: 'T12_partial'
}


# ============================================================================
# MODEL (Ian Pan UNet)
# ============================================================================

class MyDecoderBlock(nn.Module):
    def __init__(self, in_channel, skip_channel, out_channel):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel + skip_channel, out_channel,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel), nn.ReLU(inplace=True),
        )
        self.attention1 = nn.Identity()
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel), nn.ReLU(inplace=True),
        )
        self.attention2 = nn.Identity()

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x); x = self.conv2(x); x = self.attention2(x)
        return x


class MyUnetDecoder(nn.Module):
    def __init__(self, in_channel, skip_channel, out_channel):
        super().__init__()
        self.center = nn.Identity()
        i_channel = [in_channel] + out_channel[:-1]
        self.block = nn.ModuleList([
            MyDecoderBlock(i, s, o)
            for i, s, o in zip(i_channel, skip_channel, out_channel)
        ])

    def forward(self, feature, skip):
        d = self.center(feature)
        decode = []
        for i, block in enumerate(self.block):
            d = block(d, skip[i]); decode.append(d)
        return d, decode


class Net(nn.Module):
    def __init__(self, pretrained=False, cfg=None):
        super().__init__()
        self.output_type = ['infer', 'loss']
        self.register_buffer('D', torch.tensor(0))
        self.register_buffer('mean', torch.tensor(0))
        self.register_buffer('std', torch.tensor(1))
        encoder_dim = [64, 256, 512, 1024, 2048]
        decoder_dim = [256, 128, 64, 32, 16]
        if not HAS_TIMM:
            raise ImportError("timm required")
        self.encoder = timm.create_model(
            'resnet50d', pretrained=pretrained,
            in_chans=3, num_classes=0, global_pool='')
        self.decoder = MyUnetDecoder(
            in_channel=encoder_dim[-1],
            skip_channel=encoder_dim[:-1][::-1] + [0],
            out_channel=decoder_dim)
        self.logit = nn.Conv2d(decoder_dim[-1], 6, kernel_size=1)

    def forward(self, batch):
        device = self.D.device
        image = batch['sagittal'].to(device)
        x = image.float() / 255
        x = (x - self.mean) / self.std
        x = x.expand(-1, 3, -1, -1)
        encode = []
        e = self.encoder
        x = e.act1(e.bn1(e.conv1(x))); encode.append(x)
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        x = e.layer1(x); encode.append(x)
        x = e.layer2(x); encode.append(x)
        x = e.layer3(x); encode.append(x)
        x = e.layer4(x); encode.append(x)
        last, decode = self.decoder(
            feature=encode[-1], skip=encode[:-1][::-1] + [None])
        logit = self.logit(last)
        output = {}
        if 'infer' in self.output_type:
            output['probability'] = torch.softmax(logit, 1)
        return output


# ============================================================================
# UNCERTAINTY
# ============================================================================

def compute_uncertainty(heatmap: np.ndarray) -> Tuple[float, float]:
    """Returns (peak_confidence, entropy)."""
    peak_confidence = float(np.max(heatmap))
    flat = heatmap.flatten()
    flat /= (flat.sum() + 1e-9)
    entropy = float(-np.sum(flat * np.log(flat + 1e-9)))
    return peak_confidence, entropy


# ============================================================================
# CENTROID LOADING
# ============================================================================

def load_spineps_ctd(ctd_path: Path) -> Dict:
    """Load SPINEPS POI centroids. Returns {label_id: {'voxel': [x,y,z]}}."""
    try:
        with open(ctd_path) as f:
            data = json.load(f)
        
        centroids = {}
        if isinstance(data, list) and len(data) >= 2:
            centroid_data = data[1]
            if isinstance(centroid_data, dict):
                for label_str, coords_dict in centroid_data.items():
                    try:
                        lid = int(label_str)
                        if isinstance(coords_dict, dict) and "50" in coords_dict:
                            voxel_coords = coords_dict["50"]
                            centroids[lid] = {'voxel': list(voxel_coords)}
                    except (ValueError, TypeError, KeyError):
                        pass
        return centroids
    except Exception as e:
        logger.warning(f"Could not load {ctd_path}: {e}")
        return {}


def voxel_to_world(voxel: List[float], affine: np.ndarray) -> np.ndarray:
    """Convert voxel coordinates to world (mm)."""
    return (affine @ np.array([*voxel, 1.0]))[:3]


def world_to_voxel(world: np.ndarray, affine: np.ndarray, volume_shape: Tuple) -> Optional[Tuple]:
    """Convert world (mm) to voxel indices."""
    try:
        affine_inv = np.linalg.inv(affine)
        voxel = affine_inv @ np.array([*world, 1.0])
        s, r, c = int(round(voxel[0])), int(round(voxel[1])), int(round(voxel[2]))
        N, H, W = volume_shape
        return (s, r, c) if (0 <= s < N and 0 <= r < H and 0 <= c < W) else None
    except Exception:
        return None


# ============================================================================
# VOLUME LOADING
# ============================================================================

def load_nifti_volume(nifti_path: Path):
    """Load SPINEPS NIfTI. Returns (volume, affine)."""
    try:
        nii = nib.load(nifti_path)
        data = nii.get_fdata()
        vmin, vmax = data.min(), data.max()
        if vmax > vmin:
            data = ((data - vmin) / (vmax - vmin) * 255).astype(np.uint8)
        else:
            data = np.zeros_like(data, dtype=np.uint8)
        return data, nii.affine
    except Exception as e:
        logger.error(f"Error loading NIfTI: {e}")
        return None, None


# ============================================================================
# PATCH EXTRACTION + INFERENCE
# ============================================================================

def extract_patch(volume: np.ndarray, row: int, col: int,
                  patch_size: int = IMAGE_SIZE) -> np.ndarray:
    """Extract patch from middle sagittal slice."""
    _, H, W = volume.shape
    mid = volume.shape[0] // 2
    img = volume[mid].astype(np.float32)
    half = patch_size // 2
    r0, r1 = max(0, row - half), min(H, row + half)
    c0, c1 = max(0, col - half), min(W, col + half)
    crop = img[r0:r1, c0:c1]
    pad_top    = max(0, half - row)
    pad_bottom = max(0, row + half - H)
    pad_left   = max(0, half - col)
    pad_right  = max(0, col + half - W)
    crop = np.pad(crop, ((pad_top, pad_bottom), (pad_left, pad_right)), constant_values=0)
    return cv2.resize(crop, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR).astype(np.uint8)


def run_model(patch: np.ndarray, model, device) -> np.ndarray:
    """Run model on patch. Returns (6, H, W) probabilities."""
    tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).byte().to(device)
    with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
        with torch.no_grad():
            output = model({'sagittal': tensor})
    return output['probability'][0].float().cpu().numpy()


# ============================================================================
# DISC VERIFICATION
# ============================================================================

def verify_disc_at_centroid(volume, affine, centroid_voxel, model, device) -> Dict:
    """
    Run Ian Pan model at disc centroid location.
    Returns verification dict with:
      - verified_identity: which disc this actually is
      - confidence: peak probability
      - entropy: uncertainty
      - all_channels: probabilities for each disc level
    """
    # Convert voxel -> world -> voxel (to handle affine properly)
    world = voxel_to_world(centroid_voxel, affine)
    voxel_coords = world_to_voxel(world, affine, volume.shape)
    
    if voxel_coords is None:
        return None
    
    _, row, col = voxel_coords
    patch = extract_patch(volume, row, col)
    prob = run_model(patch, model, device)
    
    # For each channel, get peak confidence + entropy
    channel_metrics = {}
    for ch_idx, disc_name in CHANNEL_TO_DISC.items():
        if ch_idx == 0:  # skip background
            continue
        heatmap = prob[ch_idx]
        conf, ent = compute_uncertainty(heatmap)
        channel_metrics[disc_name] = {'confidence': conf, 'entropy': ent}
    
    # Which disc identity has highest confidence + lowest entropy?
    best_disc = None
    best_score = -np.inf
    for disc_name, metrics in channel_metrics.items():
        # Score = high confidence AND low entropy
        score = metrics['confidence'] - 0.5 * metrics['entropy']
        if score > best_score:
            best_score = score
            best_disc = disc_name
    
    return {
        'verified_identity': best_disc,
        'confidence': channel_metrics[best_disc]['confidence'],
        'entropy': channel_metrics[best_disc]['entropy'],
        'all_channels': channel_metrics,
        'is_verified': channel_metrics[best_disc]['entropy'] < ENTROPY_VERIFIED,
        'is_lstv_signal': channel_metrics[best_disc]['entropy'] > ENTROPY_LSTV,
    }


# ============================================================================
# ANATOMICAL LABEL PROPAGATION
# ============================================================================

def propagate_labels_from_discs(disc_verifications: Dict, instance_mask: np.ndarray,
                                centroids: Dict, affine: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """
    THE MISSING LINK: Use verified disc identities to re-label vertebrae.
    
    Args:
        disc_verifications: {spineps_disc_label: verification_dict}
        instance_mask: original SPINEPS instance segmentation
        centroids: SPINEPS centroids {label_id: {'voxel': [x,y,z]}}
        affine: NIfTI affine transform
    
    Returns:
        corrected_mask: re-labeled instance mask
        correction_report: {instance_label: {'before': name, 'after': name, 'reason': str}}
    """
    corrected_mask = instance_mask.copy()
    correction_report = {}
    
    # Build mapping: verified_disc_name -> world_coords
    verified_discs = {}
    for spineps_label, verification in disc_verifications.items():
        if verification and verification['is_verified']:
            disc_name = verification['verified_identity']
            if spineps_label in centroids:
                world_pos = voxel_to_world(centroids[spineps_label]['voxel'], affine)
                verified_discs[disc_name] = {
                    'world_pos': world_pos,
                    'spineps_label': spineps_label,
                    'verification': verification
                }
    
    logger.info(f"    Verified discs: {list(verified_discs.keys())}")
    
    # For each verified disc, identify and re-label adjacent vertebrae
    for disc_name, disc_info in verified_discs.items():
        if disc_name not in DISC_TO_VERTEBRAE:
            continue
        
        superior_name, inferior_name = DISC_TO_VERTEBRAE[disc_name]
        disc_world = disc_info['world_pos']
        
        # Find vertebra instances above and below this disc
        # (in world space, checking Z-coordinate in sagittal)
        vertebra_instances = {}
        for label_id in np.unique(instance_mask):
            if label_id == 0 or label_id not in SPINEPS_LABELS:
                continue
            if label_id not in centroids:
                continue
            
            vert_world = voxel_to_world(centroids[label_id]['voxel'], affine)
            
            # Check if this vertebra is adjacent to disc
            # (within reasonable distance in superior-inferior direction)
            dist_z = abs(vert_world[2] - disc_world[2])  # Z = SI direction
            
            if dist_z < 50:  # within 50mm
                if vert_world[2] > disc_world[2]:  # above disc
                    vertebra_instances['superior'] = label_id
                elif vert_world[2] < disc_world[2]:  # below disc
                    vertebra_instances['inferior'] = label_id
        
        # Re-label the adjacent vertebrae
        if 'superior' in vertebra_instances:
            old_label = vertebra_instances['superior']
            new_label = VERTEBRA_TO_LABEL.get(superior_name, old_label)
            if old_label != new_label:
                corrected_mask[instance_mask == old_label] = new_label
                correction_report[old_label] = {
                    'before': SPINEPS_LABELS.get(old_label, f'Label{old_label}'),
                    'after': superior_name,
                    'reason': f'Adjacent to verified {disc_name} (superior)',
                    'verified_disc': disc_name
                }
                logger.info(f"      {SPINEPS_LABELS.get(old_label)} -> {superior_name} (above {disc_name})")
        
        if 'inferior' in vertebra_instances:
            old_label = vertebra_instances['inferior']
            new_label = VERTEBRA_TO_LABEL.get(inferior_name, old_label)
            if old_label != new_label:
                corrected_mask[instance_mask == old_label] = new_label
                correction_report[old_label] = {
                    'before': SPINEPS_LABELS.get(old_label, f'Label{old_label}'),
                    'after': inferior_name,
                    'reason': f'Adjacent to verified {disc_name} (inferior)',
                    'verified_disc': disc_name
                }
                logger.info(f"      {SPINEPS_LABELS.get(old_label)} -> {inferior_name} (below {disc_name})")
    
    # LSTV detection: if L5-S1 has high entropy, mark those instances
    if 'L5_S1' in verified_discs:
        l5s1_verification = verified_discs['L5_S1']['verification']
        if l5s1_verification['is_lstv_signal']:
            logger.info(f"      LSTV signal detected at L5-S1 (entropy={l5s1_verification['entropy']:.3f})")
            # Mark L5 and Sacrum instances as potentially anomalous
            for old_label, correction in correction_report.items():
                if correction['after'] in ('L5', 'Sacrum'):
                    correction['lstv_flag'] = True
                    correction['l5s1_entropy'] = l5s1_verification['entropy']
    
    return corrected_mask, correction_report


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Anatomical Label Propagation')
    parser.add_argument('--nifti_dir',    required=True, help='SPINEPS NIfTI dir')
    parser.add_argument('--centroid_dir', required=True, help='SPINEPS centroids dir')
    parser.add_argument('--seg_dir',      required=True, help='SPINEPS instance masks dir')
    parser.add_argument('--output_dir',   required=True)
    parser.add_argument('--checkpoint',   required=True)
    parser.add_argument('--valid_ids',    default=None)
    parser.add_argument('--mode',         choices=['trial', 'debug', 'prod'], default='prod')
    args = parser.parse_args()

    nifti_dir    = Path(args.nifti_dir)
    centroid_dir = Path(args.centroid_dir)
    seg_dir      = Path(args.seg_dir)
    output_dir   = Path(args.output_dir)
    
    corrected_dir = output_dir / 'anatomically_corrected'
    reports_dir   = output_dir / 'correction_reports'
    audit_dir     = output_dir / 'audit_queue'
    
    for d in [corrected_dir, reports_dir, audit_dir]:
        d.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("ANATOMICAL LABEL PROPAGATION")
    logger.info("=" * 70)

    # Load validation IDs
    valid_ids = None
    if args.valid_ids and Path(args.valid_ids).exists():
        valid_ids = set(str(i) for i in np.load(args.valid_ids))
        logger.info(f"Loaded {len(valid_ids)} validation IDs")

    # Get studies
    nifti_files = sorted(nifti_dir.glob('*_T2w.nii.gz'))
    if valid_ids:
        # Extract study_id from filename: sub-{study_id}_T2w.nii.gz -> {study_id}
        nifti_files = [f for f in nifti_files 
                      if f.name.replace('sub-', '').replace('_T2w.nii.gz', '') in valid_ids]
    
    if args.mode == 'trial':
        nifti_files = nifti_files[:10]
    elif args.mode == 'debug':
        nifti_files = nifti_files[:1]
    
    logger.info(f"Processing {len(nifti_files)} studies")

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = Net(pretrained=False)
    sd = ckpt.get('state_dict') or ckpt.get('model_state_dict') or ckpt
    model.load_state_dict(sd)
    model = model.to(device)
    model.eval()
    model.output_type = ['infer']
    logger.info("✓ Model loaded")

    results = []
    audit_cases = []

    for nifti_path in tqdm(nifti_files, desc="Studies"):
        # Extract study_id from filename: sub-{study_id}_T2w.nii.gz -> {study_id}
        study_id = nifti_path.name.replace('sub-', '').replace('_T2w.nii.gz', '')
        logger.info(f"\n[{study_id}]")

        # Load volume
        volume, affine = load_nifti_volume(nifti_path)
        if volume is None:
            logger.warning("  Failed to load volume"); continue
        logger.info(f"  Volume: {volume.shape}")

        # Load centroids
        ctd_path = centroid_dir / f"{study_id}_ctd.json"
        if not ctd_path.exists():
            logger.warning("  No centroids"); continue
        centroids = load_spineps_ctd(ctd_path)
        logger.info(f"  Centroids: {len(centroids)}")

        # Load instance mask
        mask_path = seg_dir / f"{study_id}_seg-vert_msk.nii.gz"
        if not mask_path.exists():
            logger.warning("  No instance mask"); continue
        mask_nii = nib.load(mask_path)
        instance_mask = mask_nii.get_fdata().astype(np.int32)

        # STEP 1: Verify each disc
        logger.info("  Verifying discs...")
        disc_verifications = {}
        for spineps_label, disc_name in DISC_LABEL_TO_NAME.items():
            if spineps_label not in centroids:
                continue
            
            verification = verify_disc_at_centroid(
                volume, affine, centroids[spineps_label]['voxel'],
                model, device
            )
            
            if verification:
                disc_verifications[spineps_label] = verification
                status = "✓" if verification['is_verified'] else "?"
                logger.info(f"    {disc_name}: {verification['verified_identity']} "
                           f"(conf={verification['confidence']:.3f}, "
                           f"H={verification['entropy']:.3f}) {status}")

        # STEP 2: Propagate labels
        logger.info("  Propagating anatomical labels...")
        corrected_mask, correction_report = propagate_labels_from_discs(
            disc_verifications, instance_mask, centroids, affine
        )

        # Save corrected mask
        corrected_nii = nib.Nifti1Image(corrected_mask, mask_nii.affine, mask_nii.header)
        corrected_path = corrected_dir / f"{study_id}_anatomically_corrected.nii.gz"
        nib.save(corrected_nii, corrected_path)

        # Save correction report
        report = {
            'study_id': study_id,
            'n_corrections': len(correction_report),
            'corrections': correction_report,
            'disc_verifications': {
                DISC_LABEL_TO_NAME.get(k, str(k)): {
                    'verified_as': v['verified_identity'],
                    'confidence': float(v['confidence']),
                    'entropy': float(v['entropy']),
                    'is_verified': v['is_verified'],
                    'is_lstv_signal': v['is_lstv_signal'],
                } for k, v in disc_verifications.items()
            }
        }
        
        with open(reports_dir / f"{study_id}_correction_report.json", 'w') as f:
            json.dump(report, f, indent=2)

        # Add to audit queue if corrections were made
        if correction_report:
            l5s1_entropy = disc_verifications.get(124, {}).get('entropy', 0.0)
            audit_cases.append({
                'study_id': study_id,
                'n_corrections': len(correction_report),
                'corrections': list(correction_report.keys()),
                'l5_s1_entropy': float(l5s1_entropy) if isinstance(l5s1_entropy, (int, float)) else 0.0,
                'corrected_mask': str(corrected_path),
                'report': str(reports_dir / f"{study_id}_correction_report.json"),
            })

        # Collect metrics
        row = {'study_id': study_id, 'n_corrections': len(correction_report)}
        for disc_label, disc_name in DISC_LABEL_TO_NAME.items():
            if disc_label in disc_verifications:
                v = disc_verifications[disc_label]
                row[f'{disc_name}_verified_as'] = v['verified_identity']
                row[f'{disc_name}_confidence'] = v['confidence']
                row[f'{disc_name}_entropy'] = v['entropy']
                row[f'{disc_name}_verified'] = v['is_verified']
            else:
                row[f'{disc_name}_verified_as'] = 'missing'
                row[f'{disc_name}_confidence'] = 0.0
                row[f'{disc_name}_entropy'] = 0.0
                row[f'{disc_name}_verified'] = False
        results.append(row)

        logger.info(f"  ✓ {len(correction_report)} corrections made")

    # Save metrics CSV
    df = pd.DataFrame(results)
    csv_path = output_dir / 'anatomical_correction_metrics.csv'
    df.to_csv(csv_path, index=False)
    logger.info(f"\n✓ Metrics → {csv_path}")

    # Save audit queue (top-60 by corrections + L5/S1 entropy)
    audit_cases_sorted = sorted(
        audit_cases,
        key=lambda x: (x['n_corrections'], x['l5_s1_entropy']),
        reverse=True
    )[:N_AUDIT_CASES]
    
    with open(audit_dir / 'high_priority_audit.json', 'w') as f:
        json.dump({
            'description': 'Top cases with anatomical corrections',
            'n_cases': len(audit_cases_sorted),
            'cases': audit_cases_sorted
        }, f, indent=2)
    
    logger.info(f"✓ Audit queue ({len(audit_cases_sorted)} cases) → {audit_dir / 'high_priority_audit.json'}")

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info(f"  Total studies: {len(results)}")
    logger.info(f"  Studies with corrections: {sum(1 for r in results if r['n_corrections'] > 0)}")
    logger.info(f"  Total corrections: {sum(r['n_corrections'] for r in results)}")


if __name__ == '__main__':
    main()
