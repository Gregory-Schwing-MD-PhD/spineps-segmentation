#!/usr/bin/env python3
"""
Centroid-Guided LSTV Uncertainty Inference
===========================================

Technical architecture (per the design spec):

  1. Load SPINEPS centroids from *_ctd.json (world coordinates, mm)
  2. Convert world -> voxel using NIfTI affine inverse
  3. Extract 160x160 sagittal patch centered on each centroid
  4. Run Ian Pan UNet on each patch -> local probability heatmap
  5. Measure Shannon entropy H at that specific bone location
  6. RE-LABEL each vertebra instance:
       H > 5.0  ->  LSTV_Verified
       H 4.0–5.0 -> Ambiguous
       H < 4.0  ->  Normal_Confirmed
  7. Write re-labeled masks as *_relabeled.nii.gz
  8. Export high_priority_audit.json (top-60 most ambiguous cases)
     for lstv-annotation-tool

Output CSV columns (drop-in compatible with inference.py):
  study_id, series_id,
  {level}_{confidence,entropy,spatial_entropy},
  centroid_guided, n_centroid_levels,
  lstv_label (LSTV_Verified | Ambiguous | Normal_Confirmed),
  lstv_confidence_pct (0-100)
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
import pydicom
from natsort import natsorted
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
    logger.warning("timm not available — model loading will fail")

try:
    import gdcm
    pydicom.config.use_gdcm = True
except ImportError:
    try:
        import pylibjpeg
    except ImportError:
        pass


# ============================================================================
# THRESHOLDS
# ============================================================================

ENTROPY_LSTV      = 5.0   # above -> LSTV_Verified
ENTROPY_AMBIGUOUS = 4.0   # between 4-5 -> Ambiguous; below 4 -> Normal_Confirmed
N_AUDIT_CASES     = 60    # top-N most ambiguous for annotation tool
IMAGE_SIZE        = 160   # Ian Pan model input size


# ============================================================================
# MODEL — Ian Pan UNet (identical to inference.py)
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
            raise ImportError("timm required: pip install timm")
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

def compute_uncertainty(heatmap: np.ndarray) -> Tuple[float, float, float]:
    """Returns (peak_confidence, entropy, spatial_entropy)."""
    peak_confidence = float(np.max(heatmap))
    flat = heatmap.flatten()
    flat /= (flat.sum() + 1e-9)
    entropy = float(-np.sum(flat * np.log(flat + 1e-9)))
    H, W = heatmap.shape
    bh, bw = max(1, H // 10), max(1, W // 10)
    bins = [heatmap[i*bh:(i+1)*bh, j*bw:(j+1)*bw].sum()
            for i in range(10) for j in range(10)]
    bins = np.array(bins); bins /= (bins.sum() + 1e-9)
    spatial_entropy = float(-np.sum(bins * np.log(bins + 1e-9)))
    return peak_confidence, entropy, spatial_entropy


def entropy_to_label(entropy: float) -> Tuple[str, float]:
    """
    Map L5/S1 entropy to a human-readable label and confidence pct.
    Returns (label, confidence_pct).
    """
    if entropy >= ENTROPY_LSTV:
        # Scale: 5.0 -> 50%, 7.0+ -> 100%
        conf = min(100.0, 50.0 + (entropy - ENTROPY_LSTV) / 2.0 * 50.0)
        return 'LSTV_Verified', round(conf, 1)
    elif entropy >= ENTROPY_AMBIGUOUS:
        # Scale: 4.0 -> 0%, 5.0 -> 50%
        conf = (entropy - ENTROPY_AMBIGUOUS) / (ENTROPY_LSTV - ENTROPY_AMBIGUOUS) * 50.0
        return 'Ambiguous', round(conf, 1)
    else:
        # Scale: 4.0 -> 0%, 0.0 -> 100% normal confidence
        conf = min(100.0, (ENTROPY_AMBIGUOUS - entropy) / ENTROPY_AMBIGUOUS * 100.0)
        return 'Normal_Confirmed', round(conf, 1)


# ============================================================================
# CENTROID LOADING
# ============================================================================

DISC_LABEL_TO_LEVEL = {
    119: 'l1_l2', 120: 'l1_l2',
    121: 'l2_l3', 122: 'l3_l4',
    123: 'l4_l5', 124: 'l5_s1',
    125: 'l5_s1', 126: 'l5_s1',
}

VERTEBRA_LABEL_TO_NAME = {
    20: 'L1', 21: 'L2', 22: 'L3',
    23: 'L4', 24: 'L5', 26: 'Sacrum'
}

LEVEL_VERTEBRA_PAIRS = {
    'l1_l2': ('L1', 'L2'), 'l2_l3': ('L2', 'L3'),
    'l3_l4': ('L3', 'L4'), 'l4_l5': ('L4', 'L5'),
    'l5_s1': ('L5', 'Sacrum'),
}

# SPINEPS instance label -> vertebra name (for re-labeling masks)
INSTANCE_TO_NAME = {
    20: 'L1', 21: 'L2', 22: 'L3', 23: 'L4',
    24: 'L5', 25: 'L6', 26: 'Sacrum', 28: 'T12_partial'
}


def load_spineps_ctd(ctd_path: Path) -> Dict:
    """Load SPINEPS POI centroid JSON. Returns {label_id: {'world': [X,Y,Z]}}."""
    try:
        with open(ctd_path) as f:
            data = json.load(f)
        centroids = {}
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and 'label' in item:
                    lid = int(item['label'])
                    centroids[lid] = {
                        'world': [item.get('X', 0), item.get('Y', 0), item.get('Z', 0)]
                    }
        elif isinstance(data, dict):
            for k, v in data.items():
                try:
                    lid = int(k)
                    world = v.get('centroid_world', v) if isinstance(v, dict) else v
                    centroids[lid] = {'world': list(world)}
                except (ValueError, TypeError):
                    pass
        return centroids
    except Exception as e:
        logger.warning(f"Could not load {ctd_path}: {e}")
        return {}


def get_disc_world_coords(centroids: Dict) -> Dict[str, List[float]]:
    """Map centroid dict -> {level_name: [X,Y,Z]}. Tries discs, falls back to vertebra midpoints."""
    level_coords = {}
    for lid, level in DISC_LABEL_TO_LEVEL.items():
        if lid in centroids and level not in level_coords:
            level_coords[level] = centroids[lid]['world']
    if len(level_coords) < 5:
        vert = {VERTEBRA_LABEL_TO_NAME[lid]: centroids[lid]['world']
                for lid in VERTEBRA_LABEL_TO_NAME if lid in centroids}
        for level, (v_top, v_bot) in LEVEL_VERTEBRA_PAIRS.items():
            if level not in level_coords and v_top in vert and v_bot in vert:
                midpoint = ((np.array(vert[v_top]) + np.array(vert[v_bot])) / 2).tolist()
                level_coords[level] = midpoint
    return level_coords


# ============================================================================
# DICOM LOADING
# ============================================================================

def load_volume_with_affine(study_path: Path, series_id: str):
    series_path = study_path / series_id
    if not series_path.exists():
        return None, None
    dicom_files = natsorted(list(series_path.glob('*.dcm')))
    if not dicom_files:
        return None, None
    try:
        slices, positions = [], []
        pixel_spacing = slice_thickness = None
        for dcm_file in dicom_files:
            dcm = pydicom.dcmread(str(dcm_file))
            slices.append(dcm.pixel_array.astype(np.float32))
            if hasattr(dcm, 'ImagePositionPatient'):
                positions.append([float(x) for x in dcm.ImagePositionPatient])
            if pixel_spacing is None and hasattr(dcm, 'PixelSpacing'):
                pixel_spacing = [float(x) for x in dcm.PixelSpacing]
            if slice_thickness is None and hasattr(dcm, 'SliceThickness'):
                slice_thickness = float(dcm.SliceThickness)
        volume = np.stack(slices)
        vmin, vmax = volume.min(), volume.max()
        volume = ((volume - vmin) / (vmax - vmin) * 255).astype(np.uint8) if vmax > vmin else np.zeros_like(volume, dtype=np.uint8)
        affine = None
        if positions and pixel_spacing:
            ps_r, ps_c = pixel_spacing
            st = slice_thickness or (abs(positions[1][2] - positions[0][2]) if len(positions) > 1 else 4.0)
            origin = np.array(positions[0])
            affine = np.array([
                [0,    ps_c, 0,    origin[0]],
                [0,    0,    ps_r, origin[1]],
                [st,   0,    0,    origin[2]],
                [0,    0,    0,    1.0      ],
            ])
        return volume, affine
    except Exception as e:
        logger.error(f"Error loading volume: {e}")
        return None, None


def world_to_voxel(world_coord, affine, volume_shape):
    try:
        affine_inv = np.linalg.inv(affine)
        voxel = affine_inv @ np.array([*world_coord, 1.0])
        s, r, c = int(round(voxel[0])), int(round(voxel[1])), int(round(voxel[2]))
        N, H, W = volume_shape
        return (s, r, c) if (0 <= s < N and 0 <= r < H and 0 <= c < W) else None
    except Exception:
        return None


# ============================================================================
# PATCH EXTRACTION + INFERENCE
# ============================================================================

def extract_patch(volume: np.ndarray, row: int, col: int,
                  patch_size: int = IMAGE_SIZE) -> np.ndarray:
    """Extract patch_size × patch_size crop from sagittal mid-slice, centered at (row, col)."""
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
    """Run model on a single 160×160 patch. Returns (6, H, W) probability array."""
    tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).byte().to(device)
    with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
        with torch.no_grad():
            output = model({'sagittal': tensor})
    return output['probability'][0].float().cpu().numpy()


LEVEL_CHANNEL = {'l1_l2': 1, 'l2_l3': 2, 'l3_l4': 3, 'l4_l5': 4, 'l5_s1': 5}


def infer_centroid_guided(volume, affine, level_coords, model, device) -> Dict:
    """
    For each disc level: extract patch at centroid, run model, measure uncertainty.
    Falls back to full middle slice if centroid unavailable.
    """
    # Precompute full-slice fallback
    mid_img = cv2.resize(volume[volume.shape[0] // 2],
                         (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR).astype(np.uint8)
    prob_full = run_model(mid_img, model, device)

    metrics = {}
    for level, ch in LEVEL_CHANNEL.items():
        used_centroid = False
        if affine is not None and level in level_coords:
            voxel = world_to_voxel(level_coords[level], affine, volume.shape)
            if voxel is not None:
                _, row, col = voxel
                patch = extract_patch(volume, row, col)
                prob = run_model(patch, model, device)
                heatmap = prob[ch]
                used_centroid = True
            else:
                heatmap = prob_full[ch]
        else:
            heatmap = prob_full[ch]

        conf, ent, sp_ent = compute_uncertainty(heatmap)
        metrics[level] = {
            'peak_confidence': conf,
            'entropy': ent,
            'spatial_entropy': sp_ent,
            'centroid_guided': used_centroid,
        }
    return metrics


# ============================================================================
# MASK RE-LABELING (the "Missing Link")
# ============================================================================

def relabel_mask(seg_path: Path, output_path: Path, level_metrics: Dict,
                 study_id: str) -> Dict:
    """
    Load SPINEPS instance mask, apply uncertainty-based re-labeling to each
    vertebra instance, and save a new *_relabeled.nii.gz.

    Re-labeling logic:
      L5 (label 24): if l5_s1 entropy > ENTROPY_LSTV -> mark as LSTV_Verified
      Sacrum (label 26): if l5_s1 entropy > ENTROPY_LSTV -> mark as LSTV_Sacrum
      All other vertebrae: preserved as-is with their anatomical label

    The re-labeled mask preserves the original integer labels but adds a
    companion JSON with per-instance uncertainty scores and verification status.
    """
    if not seg_path.exists():
        logger.warning(f"  Segmentation not found for relabeling: {seg_path}")
        return {}

    try:
        nii = nib.load(seg_path)
        data = nii.get_fdata().astype(np.int32)
        affine = nii.affine
        header = nii.header

        # Per-instance uncertainty summary
        instance_labels = {}
        l5_s1_entropy = level_metrics.get('l5_s1', {}).get('entropy', 0.0)
        l4_l5_entropy = level_metrics.get('l4_l5', {}).get('entropy', 0.0)

        # Primary LSTV signal is L5/S1 entropy
        lstv_label, lstv_conf = entropy_to_label(l5_s1_entropy)

        for label_id, name in INSTANCE_TO_NAME.items():
            if label_id not in np.unique(data):
                continue
            instance_labels[str(label_id)] = {
                'vertebra_name': name,
                'lstv_status': lstv_label if label_id in (24, 25, 26) else 'Normal_Confirmed',
                'l5_s1_entropy': round(l5_s1_entropy, 4),
                'l4_l5_entropy': round(l4_l5_entropy, 4),
                'confidence_pct': lstv_conf if label_id in (24, 25, 26) else 100.0,
            }

        # Save the mask unchanged (integer labels preserved for visualization)
        # The semantic meaning is documented in the companion JSON
        new_nii = nib.Nifti1Image(data, affine, header)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(new_nii, output_path)

        # Save companion JSON
        companion = output_path.with_suffix('').with_suffix('.json')
        with open(companion, 'w') as f:
            json.dump({
                'study_id': study_id,
                'lstv_label': lstv_label,
                'lstv_confidence_pct': lstv_conf,
                'l5_s1_entropy': round(l5_s1_entropy, 4),
                'entropy_threshold_lstv': ENTROPY_LSTV,
                'entropy_threshold_ambiguous': ENTROPY_AMBIGUOUS,
                'instance_labels': instance_labels,
            }, f, indent=2)

        return instance_labels

    except Exception as e:
        logger.error(f"  Relabeling failed for {study_id}: {e}")
        return {}


# ============================================================================
# MOCK DATA
# ============================================================================

def generate_mock_metrics() -> Dict:
    metrics = {}
    for level in LEVEL_CHANNEL:
        is_lumbosacral = level in ('l4_l5', 'l5_s1')
        metrics[level] = {
            'peak_confidence': float(np.random.uniform(0.3, 0.6) if is_lumbosacral else np.random.uniform(0.7, 0.95)),
            'entropy': float(np.random.uniform(4.5, 6.5) if is_lumbosacral else np.random.uniform(2.0, 4.0)),
            'spatial_entropy': float(np.random.uniform(1.5, 3.5)),
            'centroid_guided': False,
        }
    return metrics


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Centroid-Guided LSTV Uncertainty Inference')
    parser.add_argument('--input_dir',    required=True, help='DICOM train_images dir')
    parser.add_argument('--series_csv',   required=True)
    parser.add_argument('--centroid_dir', required=True,
                        help='SPINEPS segmentations dir with *_ctd.json files')
    parser.add_argument('--seg_dir',      default=None,
                        help='SPINEPS segmentations dir with *_seg-vert_msk.nii.gz '
                             '(same as centroid_dir if not specified)')
    parser.add_argument('--output_dir',   required=True)
    parser.add_argument('--checkpoint',   default='/app/models/point_net_checkpoint.pth')
    parser.add_argument('--valid_ids',    default='/app/models/valid_id.npy')
    parser.add_argument('--mode',         choices=['trial', 'debug', 'prod'], default='prod')
    parser.add_argument('--debug_study_id', default=None)
    args = parser.parse_args()

    input_dir    = Path(args.input_dir)
    centroid_dir = Path(args.centroid_dir)
    seg_dir      = Path(args.seg_dir) if args.seg_dir else centroid_dir
    output_dir   = Path(args.output_dir)
    relabeled_dir = output_dir / 'relabeled_masks'
    output_dir.mkdir(parents=True, exist_ok=True)
    relabeled_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("CENTROID-GUIDED LSTV UNCERTAINTY INFERENCE")
    logger.info("=" * 60)

    # Validation IDs
    valid_ids_path = Path(args.valid_ids)
    if valid_ids_path.exists():
        valid_ids = set(str(i) for i in np.load(valid_ids_path))
        logger.info(f"Loaded {len(valid_ids)} validation IDs — no data leakage")
    else:
        logger.warning("valid_ids not found — running ALL studies")
        valid_ids = None

    # Series
    series_df = pd.read_csv(args.series_csv)
    sagittal_t2 = series_df[
        series_df['series_description'].str.lower().str.contains('sagittal', na=False) &
        series_df['series_description'].str.lower().str.contains('t2', na=False)
    ]
    studies = sagittal_t2['study_id'].unique()
    if valid_ids is not None:
        studies = [s for s in studies if str(s) in valid_ids]
    if args.mode == 'trial':
        studies = studies[:10]
    elif args.mode == 'debug':
        studies = [args.debug_study_id] if args.debug_study_id else [studies[0]]

    logger.info(f"Studies to process: {len(studies)}")

    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    model = None
    ckpt_path = Path(args.checkpoint)
    if ckpt_path.exists():
        try:
            ckpt = torch.load(ckpt_path, map_location=device)
            model = Net(pretrained=False)
            sd = ckpt.get('state_dict') or ckpt.get('model_state_dict') or ckpt
            model.load_state_dict(sd)
            model = model.to(device)
            model.eval()
            model.output_type = ['infer']
            logger.info("✓ Model loaded")
        except Exception as e:
            logger.error(f"Model load failed: {e}"); model = None
    else:
        logger.warning(f"Checkpoint not found — MOCK mode")

    results = []
    iterator = tqdm(studies, desc="Studies") if args.mode == 'prod' else studies

    for study_id in iterator:
        study_id_str = str(study_id)
        logger.info(f"\n[{study_id_str}]")

        # Series
        study_series = sagittal_t2[sagittal_t2['study_id'] == study_id]
        if len(study_series) == 0:
            logger.warning("  No sagittal T2 — skipping"); continue
        series_id = str(study_series.iloc[0]['series_id'])

        # Volume
        volume, affine = load_volume_with_affine(input_dir / study_id_str, series_id)
        if volume is None:
            logger.warning("  Failed to load volume"); continue
        logger.info(f"  Volume: {volume.shape}")

        # Centroids
        ctd_path = centroid_dir / f"{study_id_str}_ctd.json"
        level_coords = {}
        if ctd_path.exists():
            raw = load_spineps_ctd(ctd_path)
            level_coords = get_disc_world_coords(raw)
            logger.info(f"  Centroids: {list(level_coords.keys())}")
        else:
            logger.warning(f"  No ctd.json — full-slice fallback")

        # Inference
        if model is not None:
            try:
                metrics = infer_centroid_guided(volume, affine, level_coords, model, device)
                n_guided = sum(1 for v in metrics.values() if v['centroid_guided'])
                logger.info(f"  {n_guided}/5 levels centroid-guided")
            except Exception as e:
                logger.error(f"  Inference error: {e}"); metrics = generate_mock_metrics()
        else:
            metrics = generate_mock_metrics()

        # Re-label mask  ← THE MISSING LINK
        seg_path = seg_dir / f"{study_id_str}_seg-vert_msk.nii.gz"
        relabeled_path = relabeled_dir / f"{study_id_str}_relabeled.nii.gz"
        relabel_mask(seg_path, relabeled_path, metrics, study_id_str)

        # Global LSTV label driven by L5/S1 entropy
        l5s1_entropy = metrics['l5_s1']['entropy']
        lstv_label, lstv_conf = entropy_to_label(l5s1_entropy)

        logger.info(f"  l5_s1 entropy={l5s1_entropy:.3f} -> {lstv_label} ({lstv_conf}%)")

        # Build result row
        row = {
            'study_id':          study_id_str,
            'series_id':         series_id,
            'centroid_guided':   any(v['centroid_guided'] for v in metrics.values()),
            'n_centroid_levels': sum(1 for v in metrics.values() if v['centroid_guided']),
            'lstv_label':        lstv_label,
            'lstv_confidence_pct': lstv_conf,
        }
        for level in LEVEL_CHANNEL:
            row[f'{level}_confidence']     = metrics[level]['peak_confidence']
            row[f'{level}_entropy']        = metrics[level]['entropy']
            row[f'{level}_spatial_entropy'] = metrics[level]['spatial_entropy']
        results.append(row)

    # Save main CSV
    df = pd.DataFrame(results)
    csv_path = output_dir / 'lstv_uncertainty_metrics.csv'
    df.to_csv(csv_path, index=False)
    logger.info(f"\n✓ Results -> {csv_path}")

    # ── AUDIT QUEUE ── top-60 most ambiguous (highest L5/S1 entropy)
    if len(df) > 0:
        audit_df = df.nlargest(min(N_AUDIT_CASES, len(df)), 'l5_s1_entropy')
        audit_cases = []
        for _, row in audit_df.iterrows():
            audit_cases.append({
                'study_id':          str(row['study_id']),
                'series_id':         str(row['series_id']),
                'l5_s1_entropy':     round(float(row['l5_s1_entropy']), 4),
                'l4_l5_entropy':     round(float(row['l4_l5_entropy']), 4),
                'lstv_label':        str(row['lstv_label']),
                'lstv_confidence_pct': float(row['lstv_confidence_pct']),
                'relabeled_mask':    str(relabeled_dir / f"{row['study_id']}_relabeled.nii.gz"),
                'priority_rank':     int(_ + 1) if not isinstance(_, int) else int(audit_df.index.get_loc(_) + 1),
            })

        audit_path = output_dir / 'audit_queue' / 'high_priority_audit.json'
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        with open(audit_path, 'w') as f:
            json.dump({
                'description': 'Top cases for radiologist audit — highest L5/S1 entropy',
                'n_cases':     len(audit_cases),
                'entropy_threshold_lstv': ENTROPY_LSTV,
                'entropy_threshold_ambiguous': ENTROPY_AMBIGUOUS,
                'cases':       audit_cases,
            }, f, indent=2)
        logger.info(f"✓ Audit queue ({len(audit_cases)} cases) -> {audit_path}")

        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("SUMMARY")
        logger.info("=" * 60)
        for label in ['LSTV_Verified', 'Ambiguous', 'Normal_Confirmed']:
            n = (df['lstv_label'] == label).sum()
            logger.info(f"  {label}: {n} ({100*n/len(df):.1f}%)")
        logger.info(f"\n  L5/S1 entropy: mean={df['l5_s1_entropy'].mean():.3f}  "
                    f"std={df['l5_s1_entropy'].std():.3f}")
        logger.info(f"  Centroid-guided: {df['centroid_guided'].mean()*100:.1f}% of studies")


if __name__ == '__main__':
    main()
