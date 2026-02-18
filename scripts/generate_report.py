#!/usr/bin/env python3
"""
LSTV Report Generator with Relabeled Mask Visualization
========================================================
Visualizes:
  - Top/bottom LSTV confidence cases
  - Embedded DICOM images with segmentation mask overlay
  - LSTV labels (LSTV_Verified, Ambiguous, Normal_Confirmed) color-coded
  - Per-vertebra instance labels from relabeled JSON
"""
import argparse
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path
import base64
from io import BytesIO
from jinja2 import Template
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from loguru import logger
import sys
import pydicom
import nibabel as nib
import cv2
from natsort import natsorted

logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | <level>{message}</level>")


def encode_image_base64(fig):
    """Convert matplotlib figure to base64"""
    buffer = BytesIO()
    fig.savefig(buffer, format='png', dpi=120, bbox_inches='tight', facecolor='white')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()
    plt.close(fig)
    return f"data:image/png;base64,{image_base64}"


def normalise_to_8bit(x):
    """Normalize to 8-bit range"""
    lower, upper = np.percentile(x, (0.5, 99.5))
    x = np.clip(x, lower, upper)
    x = x - np.min(x)
    if np.max(x) > 0:
        x = x / np.max(x)
    return (x * 255).astype(np.uint8)


def load_dicom_middle_slice(study_path, series_id):
    """Load middle DICOM slice"""
    series_path = study_path / str(series_id)
    if not series_path.exists():
        return None
    dicom_files = natsorted(list(series_path.glob('*.dcm')))
    if not dicom_files:
        return None
    try:
        mid_idx = len(dicom_files) // 2
        dcm = pydicom.dcmread(str(dicom_files[mid_idx]))
        img = dcm.pixel_array
        return normalise_to_8bit(img)
    except Exception as e:
        logger.warning(f"Error loading {series_id}: {e}")
        return None


def load_mask_middle_slice(mask_path):
    """Load middle slice of NIfTI mask"""
    try:
        nii = nib.load(mask_path)
        data = nii.get_fdata().astype(np.int32)
        mid_slice = data.shape[0] // 2
        return data[mid_slice, :, :]
    except Exception as e:
        logger.warning(f"Error loading mask {mask_path}: {e}")
        return None


def create_mask_overlay_viz(study_id, series_id, data_dir, seg_dir, relabeled_dir):
    """
    Create visualization with DICOM + mask overlay + LSTV labels
    Returns base64-encoded image
    """
    study_path = Path(data_dir) / str(int(study_id))
    
    # Load DICOM
    dicom_img = load_dicom_middle_slice(study_path, series_id)
    if dicom_img is None:
        return None
    
    # Load relabeled mask
    relabeled_mask_path = Path(relabeled_dir) / f"{int(study_id)}_relabeled.nii.gz"
    relabeled_json_path = Path(relabeled_dir) / f"{int(study_id)}_relabeled.json"
    
    mask = None
    instance_labels = {}
    lstv_label = "Unknown"
    lstv_conf = 0.0
    
    if relabeled_mask_path.exists():
        mask = load_mask_middle_slice(relabeled_mask_path)
    
    if relabeled_json_path.exists():
        with open(relabeled_json_path) as f:
            relabeled_data = json.load(f)
            instance_labels = relabeled_data.get('instance_labels', {})
            lstv_label = relabeled_data.get('lstv_label', 'Unknown')
            lstv_conf = relabeled_data.get('lstv_confidence_pct', 0.0)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Panel 1: DICOM only
    axes[0].imshow(dicom_img, cmap='gray')
    axes[0].set_title('Sagittal T2 (Middle Slice)', fontsize=13, fontweight='bold')
    axes[0].axis('off')
    
    # Panel 2: DICOM + mask overlay
    axes[1].imshow(dicom_img, cmap='gray')
    if mask is not None:
        # Color map: L5 (24) = red, Sacrum (26) = blue, others = green
        mask_colored = np.zeros((*mask.shape, 4))
        for label_id_str, info in instance_labels.items():
            label_id = int(label_id_str)
            if label_id not in np.unique(mask):
                continue
            
            vert_name = info.get('vertebra_name', f'L{label_id}')
            status = info.get('lstv_status', 'Normal_Confirmed')
            
            # Color by LSTV status
            if status == 'LSTV_Verified':
                color = [1.0, 0.0, 0.0, 0.4]  # Red
            elif status == 'Ambiguous':
                color = [1.0, 0.65, 0.0, 0.4]  # Orange
            else:
                color = [0.0, 1.0, 0.0, 0.3]  # Green
            
            mask_colored[mask == label_id] = color
        
        axes[1].imshow(mask_colored, interpolation='nearest')
    axes[1].set_title('DICOM + Segmentation Overlay', fontsize=13, fontweight='bold')
    axes[1].axis('off')
    
    # Panel 3: Legend with per-vertebra labels
    axes[2].axis('off')
    axes[2].set_xlim(0, 10)
    axes[2].set_ylim(0, 10)
    
    y_pos = 9.5
    axes[2].text(0.5, y_pos, f"Study: {int(study_id)}", fontsize=14, fontweight='bold')
    y_pos -= 0.8
    
    # Global LSTV label
    label_colors = {
        'LSTV_Verified': '#ff6b6b',
        'Ambiguous': '#ffa500',
        'Normal_Confirmed': '#51cf66'
    }
    label_color = label_colors.get(lstv_label, '#999')
    axes[2].text(0.5, y_pos, f"LSTV: {lstv_label}", fontsize=12, fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor=label_color, alpha=0.3))
    y_pos -= 0.7
    axes[2].text(0.5, y_pos, f"Confidence: {lstv_conf:.1f}%", fontsize=11, color='#666')
    y_pos -= 1.0
    
    # Per-vertebra instances
    axes[2].text(0.5, y_pos, "Vertebra Instances:", fontsize=11, fontweight='bold')
    y_pos -= 0.6
    
    for label_id_str in sorted(instance_labels.keys(), key=lambda x: int(x)):
        info = instance_labels[label_id_str]
        vert_name = info['vertebra_name']
        status = info['lstv_status']
        
        if status == 'LSTV_Verified':
            color_box = 'red'
        elif status == 'Ambiguous':
            color_box = 'orange'
        else:
            color_box = 'green'
        
        # Color box
        rect = Rectangle((0.3, y_pos - 0.15), 0.3, 0.3, facecolor=color_box, alpha=0.5, edgecolor='black')
        axes[2].add_patch(rect)
        
        # Label text
        axes[2].text(0.8, y_pos, f"{vert_name}: {status}", fontsize=9, va='bottom')
        y_pos -= 0.5
        
        if y_pos < 0.5:
            break
    
    plt.suptitle(f'LSTV Detection — Study {int(study_id)}', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    return encode_image_base64(fig)


def create_plotly_distributions(df: pd.DataFrame) -> tuple:
    """Create interactive plotly plots"""
    
    # Confidence histogram
    fig_conf = go.Figure()
    fig_conf.add_trace(go.Histogram(
        x=df['l5_s1_confidence'],
        nbinsx=50,
        marker_color='#764ba2',
        opacity=0.7
    ))
    fig_conf.update_layout(
        title_text="L5/S1 Peak Confidence Distribution",
        xaxis_title="Peak Confidence",
        yaxis_title="Count",
        template='plotly_white',
        height=400
    )
    
    # Scatter: Confidence vs Entropy (color by lstv_label if available)
    fig_scatter = go.Figure()
    
    if 'lstv_label' in df.columns:
        color_map = {'LSTV_Verified': 'red', 'Ambiguous': 'orange', 'Normal_Confirmed': 'green'}
        colors = df['lstv_label'].map(color_map)
    else:
        colors = '#764ba2'
    
    fig_scatter.add_trace(go.Scatter(
        x=df['l5_s1_confidence'],
        y=df['l5_s1_entropy'],
        mode='markers',
        marker=dict(size=8, color=colors, opacity=0.6, line=dict(width=1, color='white')),
        text=df['study_id'].astype(int),
        customdata=df['lstv_label'] if 'lstv_label' in df.columns else None,
        hovertemplate='<b>Study:</b> %{text}<br>' +
                     '<b>Label:</b> %{customdata}<br>' +
                     '<b>Confidence:</b> %{x:.4f}<br>' +
                     '<b>Entropy:</b> %{y:.4f}<extra></extra>'
    ))
    fig_scatter.update_layout(
        title_text="L5-S1: Confidence vs Entropy",
        xaxis_title="Peak Confidence",
        yaxis_title="Entropy",
        template='plotly_white',
        height=500
    )
    
    return (
        fig_conf.to_html(include_plotlyjs=False, div_id="conf_dist"),
        fig_scatter.to_html(include_plotlyjs=False, div_id="scatter")
    )


HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>LSTV Detection Report</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }
        .header h1 { font-size: 2.5em; margin-bottom: 10px; font-weight: 700; }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            padding: 40px;
            background: #f8f9fa;
        }
        .stat-card {
            background: white;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .stat-card h3 {
            color: #667eea;
            font-size: 0.9em;
            text-transform: uppercase;
            margin-bottom: 10px;
        }
        .stat-card .value { font-size: 2.5em; font-weight: 700; color: #2d3748; }
        .section { padding: 40px; }
        .section-title {
            font-size: 1.8em;
            color: #2d3748;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #667eea;
        }
        .case-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(600px, 1fr));
            gap: 30px;
            margin-top: 30px;
        }
        .case-card {
            background: white;
            border-radius: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        .case-card img { width: 100%; display: block; }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            border-radius: 15px;
            overflow: hidden;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        thead {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        th, td { padding: 15px; text-align: left; }
        tbody tr:hover { background: #f7fafc; }
        .lstv-verified { background: #fed7d7; color: #c53030; padding: 5px 10px; border-radius: 5px; font-weight: 600; }
        .lstv-ambiguous { background: #feebc8; color: #c05621; padding: 5px 10px; border-radius: 5px; font-weight: 600; }
        .lstv-normal { background: #c6f6d5; color: #276749; padding: 5px 10px; border-radius: 5px; font-weight: 600; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔬 LSTV Detection Report</h1>
            <p>Centroid-Guided Epistemic Uncertainty Analysis</p>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <h3>Total Studies</h3>
                <div class="value">{{ stats.total }}</div>
            </div>
            <div class="stat-card">
                <h3>LSTV Verified</h3>
                <div class="value" style="color: #ff6b6b;">{{ stats.lstv_verified }}</div>
            </div>
            <div class="stat-card">
                <h3>Ambiguous</h3>
                <div class="value" style="color: #ffa500;">{{ stats.ambiguous }}</div>
            </div>
            <div class="stat-card">
                <h3>Normal</h3>
                <div class="value" style="color: #51cf66;">{{ stats.normal }}</div>
            </div>
        </div>
        
        <div class="section">
            <h2 class="section-title">📊 Distribution Analysis</h2>
            <div style="background: white; border-radius: 15px; padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin: 20px 0;">
                {{ conf_plot }}
            </div>
            <div style="background: white; border-radius: 15px; padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin: 20px 0;">
                {{ scatter_plot }}
            </div>
        </div>
        
        {% if high_conf_images %}
        <div class="section">
            <h2 class="section-title">🔴 Lowest Confidence Cases (Top 5)</h2>
            <p style="margin-bottom: 20px; color: #718096;">
                Lowest L5/S1 confidence — strongest LSTV candidates
            </p>
            <div class="case-grid">
                {% for case in high_conf_images %}
                <div class="case-card">
                    <img src="{{ case.image }}" alt="Study {{ case.study_id }}">
                </div>
                {% endfor %}
            </div>
        </div>
        
        <div class="section">
            <h2 class="section-title">🟢 Highest Confidence Cases (Top 5)</h2>
            <p style="margin-bottom: 20px; color: #718096;">
                Highest L5/S1 confidence — normal anatomy controls
            </p>
            <div class="case-grid">
                {% for case in low_conf_images %}
                <div class="case-card">
                    <img src="{{ case.image }}" alt="Study {{ case.study_id }}">
                </div>
                {% endfor %}
            </div>
        </div>
        {% endif %}
        
        <div class="section">
            <h2 class="section-title">📋 Top Candidates</h2>
            <table>
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Study ID</th>
                        <th>LSTV Label</th>
                        <th>Confidence %</th>
                        <th>L5-S1 Conf</th>
                        <th>L5-S1 Entropy</th>
                        <th>Centroid Guided</th>
                    </tr>
                </thead>
                <tbody>
                    {% for row in top_candidates %}
                    <tr>
                        <td>{{ loop.index }}</td>
                        <td><strong>{{ row.study_id }}</strong></td>
                        <td><span class="{{ row.label_class }}">{{ row.lstv_label }}</span></td>
                        <td>{{ row.lstv_confidence_pct }}%</td>
                        <td>{{ "%.4f"|format(row.l5_s1_confidence) }}</td>
                        <td>{{ "%.4f"|format(row.l5_s1_entropy) }}</td>
                        <td>{{ row.n_centroid_levels }}/5</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
        
        <div style="background: #2d3748; color: white; padding: 30px; text-align: center;">
            <p>{{ timestamp }}</p>
        </div>
    </div>
</body>
</html>
"""


def generate_report(csv_path, output_path, data_dir=None, series_csv=None, 
                   seg_dir=None, relabeled_dir=None):
    """Generate HTML report with relabeled mask visualizations"""
    
    logger.info(f"Loading results from {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} studies")
    
    # Load series descriptions
    series_df = None
    if series_csv and Path(series_csv).exists():
        series_df = pd.read_csv(series_csv)
    
    # Stats
    stats = {
        'total': len(df),
        'lstv_verified': (df['lstv_label'] == 'LSTV_Verified').sum() if 'lstv_label' in df.columns else 0,
        'ambiguous': (df['lstv_label'] == 'Ambiguous').sum() if 'lstv_label' in df.columns else 0,
        'normal': (df['lstv_label'] == 'Normal_Confirmed').sum() if 'lstv_label' in df.columns else 0,
    }
    
    # Plotly charts
    conf_plot, scatter_plot = create_plotly_distributions(df)
    
    # Top candidates table
    top_df = df.sort_values('l5_s1_confidence').head(20)
    top_candidates = []
    label_class_map = {
        'LSTV_Verified': 'lstv-verified',
        'Ambiguous': 'lstv-ambiguous',
        'Normal_Confirmed': 'lstv-normal'
    }
    for _, row in top_df.iterrows():
        top_candidates.append({
            'study_id': int(row['study_id']),
            'lstv_label': row.get('lstv_label', 'Unknown'),
            'label_class': label_class_map.get(row.get('lstv_label'), 'lstv-normal'),
            'lstv_confidence_pct': round(row.get('lstv_confidence_pct', 0), 1),
            'l5_s1_confidence': row['l5_s1_confidence'],
            'l5_s1_entropy': row['l5_s1_entropy'],
            'n_centroid_levels': row.get('n_centroid_levels', 0),
        })
    
    # Generate case images
    high_conf_images = []
    low_conf_images = []
    
    if data_dir and relabeled_dir and Path(relabeled_dir).exists():
        logger.info("Generating case visualizations with mask overlays...")
        
        # Bottom 5 confidence (LSTV candidates)
        bottom_df = df.sort_values('l5_s1_confidence').head(5)
        for _, row in bottom_df.iterrows():
            logger.info(f"  Processing low-confidence: {int(row['study_id'])}")
            try:
                img_b64 = create_mask_overlay_viz(
                    row['study_id'], row['series_id'],
                    data_dir, seg_dir, relabeled_dir
                )
                if img_b64:
                    high_conf_images.append({
                        'study_id': int(row['study_id']),
                        'image': img_b64
                    })
            except Exception as e:
                logger.warning(f"    Error: {e}")
        
        # Top 5 confidence (normal controls)
        top_df_conf = df.sort_values('l5_s1_confidence', ascending=False).head(5)
        for _, row in top_df_conf.iterrows():
            logger.info(f"  Processing high-confidence: {int(row['study_id'])}")
            try:
                img_b64 = create_mask_overlay_viz(
                    row['study_id'], row['series_id'],
                    data_dir, seg_dir, relabeled_dir
                )
                if img_b64:
                    low_conf_images.append({
                        'study_id': int(row['study_id']),
                        'image': img_b64
                    })
            except Exception as e:
                logger.warning(f"    Error: {e}")
    
    # Render HTML
    logger.info("Rendering HTML...")
    template = Template(HTML_TEMPLATE)
    html = template.render(
        stats=stats,
        conf_plot=conf_plot,
        scatter_plot=scatter_plot,
        top_candidates=top_candidates,
        high_conf_images=high_conf_images,
        low_conf_images=low_conf_images,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
    
    Path(output_path).write_text(html)
    logger.info(f"✓ Report saved: {output_path}")
    logger.info(f"✓ Low-confidence cases visualized: {len(high_conf_images)}")
    logger.info(f"✓ High-confidence cases visualized: {len(low_conf_images)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--data_dir', help='DICOM data directory')
    parser.add_argument('--series_csv', help='train_series_descriptions.csv')
    parser.add_argument('--seg_dir', help='SPINEPS segmentation directory')
    parser.add_argument('--relabeled_dir', help='Relabeled masks directory')
    args = parser.parse_args()
    
    generate_report(
        csv_path=args.csv,
        output_path=args.output,
        data_dir=args.data_dir,
        series_csv=args.series_csv,
        seg_dir=args.seg_dir,
        relabeled_dir=args.relabeled_dir
    )


if __name__ == '__main__':
    main()
