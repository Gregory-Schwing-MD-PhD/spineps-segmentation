#!/usr/bin/env python3
"""
Visual Comparison: Original SPINEPS vs Anatomically Corrected Labels
=====================================================================

Creates side-by-side visualizations showing:
  - Left: Original SPINEPS instance mask
  - Right: Anatomically corrected mask
  - Annotations showing which labels changed and why

For the audit queue (top-60 cases with corrections)
"""

import argparse
import json
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path
from tqdm import tqdm

SPINEPS_LABELS = {
    19: 'T12', 20: 'L1', 21: 'L2', 22: 'L3',
    23: 'L4', 24: 'L5', 25: 'L6', 26: 'Sacrum', 28: 'T12_partial'
}

LABEL_COLORS = {
    19: '#FFB6C1',  # T12 - light pink
    20: '#FF69B4',  # L1 - hot pink
    21: '#FF1493',  # L2 - deep pink
    22: '#C71585',  # L3 - medium violet red
    23: '#8B008B',  # L4 - dark magenta
    24: '#4B0082',  # L5 - indigo
    25: '#9370DB',  # L6 - medium purple
    26: '#7B68EE',  # Sacrum - medium slate blue
}


def create_comparison_figure(original_path, corrected_path, report_path, output_path):
    """Create side-by-side comparison with annotations."""
    
    # Load masks
    original_nii = nib.load(original_path)
    corrected_nii = nib.load(corrected_path)
    
    original = original_nii.get_fdata().astype(np.int32)
    corrected = corrected_nii.get_fdata().astype(np.int32)
    
    # Load report
    with open(report_path) as f:
        report = json.load(f)
    
    # Get middle sagittal slice
    mid_slice = original.shape[0] // 2
    original_slice = original[mid_slice, :, :]
    corrected_slice = corrected[mid_slice, :, :]
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    
    # Panel 1: Original SPINEPS
    ax = axes[0]
    ax.imshow(original_slice, cmap='gray', alpha=0.3)
    
    # Color-code each instance
    colored_original = np.zeros((*original_slice.shape, 4))
    for label_id in np.unique(original_slice):
        if label_id == 0:
            continue
        mask = original_slice == label_id
        color = LABEL_COLORS.get(label_id, [0.5, 0.5, 0.5, 0.5])
        if isinstance(color, str):
            from matplotlib.colors import to_rgba
            color = to_rgba(color, alpha=0.5)
        colored_original[mask] = color
    
    ax.imshow(colored_original)
    ax.set_title('Original SPINEPS Instance Mask', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Add legend for original
    y_pos = 0.95
    for label_id in sorted(np.unique(original_slice)):
        if label_id == 0:
            continue
        name = SPINEPS_LABELS.get(label_id, f'Label {label_id}')
        color = LABEL_COLORS.get(label_id, '#999999')
        ax.text(0.02, y_pos, f'■ {name}', transform=ax.transAxes,
                fontsize=10, color=color, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        y_pos -= 0.06
    
    # Panel 2: Corrected
    ax = axes[1]
    ax.imshow(corrected_slice, cmap='gray', alpha=0.3)
    
    colored_corrected = np.zeros((*corrected_slice.shape, 4))
    for label_id in np.unique(corrected_slice):
        if label_id == 0:
            continue
        mask = corrected_slice == label_id
        color = LABEL_COLORS.get(label_id, [0.5, 0.5, 0.5, 0.5])
        if isinstance(color, str):
            from matplotlib.colors import to_rgba
            color = to_rgba(color, alpha=0.5)
        colored_corrected[mask] = color
    
    ax.imshow(colored_corrected)
    ax.set_title('Anatomically Corrected', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Panel 3: Correction legend
    ax = axes[2]
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    
    y_pos = 9.5
    ax.text(0.5, y_pos, f"Study: {report['study_id']}", fontsize=14, fontweight='bold')
    y_pos -= 0.8
    
    ax.text(0.5, y_pos, f"Corrections: {report['n_corrections']}", fontsize=12, fontweight='bold')
    y_pos -= 1.0
    
    # List each correction
    if report['corrections']:
        ax.text(0.5, y_pos, "Label Changes:", fontsize=11, fontweight='bold')
        y_pos -= 0.6
        
        for old_label_str, correction in report['corrections'].items():
            old_label = int(old_label_str)
            before = correction['before']
            after = correction['after']
            reason = correction['reason']
            
            # Color boxes
            old_color = LABEL_COLORS.get(old_label, '#999999')
            new_label = next((k for k, v in SPINEPS_LABELS.items() if v == after), None)
            new_color = LABEL_COLORS.get(new_label, '#00FF00') if new_label else '#00FF00'
            
            rect1 = Rectangle((0.3, y_pos - 0.15), 0.3, 0.3, facecolor=old_color, 
                            edgecolor='black', alpha=0.7)
            rect2 = Rectangle((1.0, y_pos - 0.15), 0.3, 0.3, facecolor=new_color,
                            edgecolor='black', alpha=0.7)
            ax.add_patch(rect1)
            ax.add_patch(rect2)
            
            ax.text(1.5, y_pos, f"{before} → {after}", fontsize=9, va='bottom')
            y_pos -= 0.4
            ax.text(0.7, y_pos, f"  {reason}", fontsize=8, style='italic', color='#666')
            y_pos -= 0.6
            
            if y_pos < 1.0:
                ax.text(0.5, y_pos, "...(see JSON for full report)", fontsize=8, style='italic')
                break
    
    # Add verification info
    if 'disc_verifications' in report:
        y_pos -= 0.5
        ax.text(0.5, y_pos, "Disc Verifications:", fontsize=11, fontweight='bold')
        y_pos -= 0.6
        
        for disc_name, verification in list(report['disc_verifications'].items())[:5]:
            verified = "✓" if verification['is_verified'] else "?"
            lstv = " [LSTV]" if verification.get('is_lstv_signal', False) else ""
            
            ax.text(0.5, y_pos,
                   f"{disc_name}: {verification['verified_as']} {verified}{lstv}",
                   fontsize=9)
            y_pos -= 0.3
            ax.text(0.7, y_pos,
                   f"H={verification['entropy']:.2f}, conf={verification['confidence']:.2f}",
                   fontsize=8, color='#666')
            y_pos -= 0.5
    
    plt.suptitle('Anatomical Label Propagation — Before vs After',
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize anatomical corrections')
    parser.add_argument('--original_dir',  required=True, help='Original SPINEPS masks')
    parser.add_argument('--corrected_dir', required=True, help='Corrected masks')
    parser.add_argument('--reports_dir',   required=True, help='Correction reports')
    parser.add_argument('--audit_json',    required=True, help='Audit queue JSON')
    parser.add_argument('--output_dir',    required=True, help='Output visualizations')
    args = parser.parse_args()

    original_dir  = Path(args.original_dir)
    corrected_dir = Path(args.corrected_dir)
    reports_dir   = Path(args.reports_dir)
    output_dir    = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load audit queue
    with open(args.audit_json) as f:
        audit_data = json.load(f)
    
    print(f"Generating visualizations for {len(audit_data['cases'])} cases...")

    for case in tqdm(audit_data['cases'], desc="Visualizations"):
        study_id = case['study_id']
        
        original_path  = original_dir / f"{study_id}_seg-vert_msk.nii.gz"
        corrected_path = corrected_dir / f"{study_id}_anatomically_corrected.nii.gz"
        report_path    = reports_dir / f"{study_id}_correction_report.json"
        output_path    = output_dir / f"{study_id}_comparison.png"
        
        if not all([original_path.exists(), corrected_path.exists(), report_path.exists()]):
            print(f"Skipping {study_id} — missing files")
            continue
        
        try:
            create_comparison_figure(original_path, corrected_path, report_path, output_path)
        except Exception as e:
            print(f"Error processing {study_id}: {e}")
    
    print(f"\n✓ Visualizations saved to: {output_dir}")
    print(f"\nTo create an HTML gallery:")
    print(f"  python scripts/create_audit_gallery.py \\")
    print(f"    --viz_dir {output_dir} \\")
    print(f"    --audit_json {args.audit_json} \\")
    print(f"    --output audit_gallery.html")


if __name__ == '__main__':
    main()
