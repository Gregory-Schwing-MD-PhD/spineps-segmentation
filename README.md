# SPINEPS Segmentation Pipeline

**Clean, focused pipeline for SPINEPS segmentation of lumbar spine MRI studies**

This repository processes RSNA 2024 Lumbar Spine MRI data through SPINEPS to generate:
- Instance segmentation masks (vertebra-level labels)
- Semantic segmentation masks (region-level)
- 3D centroids for vertebrae and discs
- Anatomical label mappings

**Designed for:** LSTV (Lumbosacral Transitional Vertebrae) detection research

---

## 🎯 Quick Start

### Prerequisites

- HPC cluster with:
  - Singularity/Apptainer
  - GPU support (NVIDIA V100 or better)
  - Slurm workload manager
- Kaggle account (for dataset download)
- ~200GB storage for dataset
- `valid_id.npy` file (validation set study IDs)

### Installation

```bash
# Clone repository
git clone https://github.com/Gregory-Schwing-MD-PhD/spineps-segmentation.git
cd spineps-segmentation

# Set up Kaggle credentials
mkdir -p ~/.kaggle
cp your_kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

# Download validation IDs (283 study IDs)
sbatch slurm_scripts/00_download_valid_ids.sh
# Wait for completion, then verify:
ls -lh models/valid_id.npy
```

### Run Pipeline

**Trial mode (3 studies, data already downloaded):**
```bash
sbatch slurm_scripts/01_spineps_trial.sh
```

**Production mode (283 studies, data already downloaded):**
```bash
sbatch slurm_scripts/02_spineps_production.sh
```

**With data download:**
```bash
sbatch slurm_scripts/03_spineps_trial_with_download.sh  # Trial + download
sbatch slurm_scripts/04_spineps_production_with_download.sh  # Production + download
```

---

## 📂 Repository Structure

```
spineps-segmentation/
├── docker/
│   ├── Dockerfile.preprocessing      # Data download & preprocessing
│   ├── Dockerfile.spineps           # SPINEPS segmentation
│   ├── build_preprocessing.sh
│   └── build_spineps.sh
├── scripts/
│   ├── download_rsna_data.py        # Kaggle dataset download
│   ├── preprocess_dicoms.py         # DICOM → NIfTI conversion
│   ├── run_spineps_segmentation.py  # Main SPINEPS pipeline
│   └── extract_centroids.py         # Centroid extraction
├── slurm_scripts/
│   ├── 00_download_valid_ids.sh     # Download validation IDs (run first!)
│   ├── 01_spineps_trial.sh          # Trial (3 studies, no download)
│   ├── 02_spineps_production.sh     # Production (283 studies, no download)
│   ├── 03_spineps_trial_with_download.sh
│   └── 04_spineps_production_with_download.sh
├── src/
│   └── preprocessing/
│       └── spineps_wrapper.sh       # SPINEPS CLI wrapper
├── data/
│   └── raw/                         # Downloaded RSNA data
├── results/
│   └── spineps_segmentation/        # Pipeline outputs
├── models/
│   └── valid_id.npy                 # Validation set IDs ← YOU PROVIDE THIS
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 📊 Output Structure

```
results/spineps_segmentation/
├── nifti/
│   └── {study_id}_T2w.nii.gz                    # Input MRI
├── segmentations/
│   ├── {study_id}_seg-vert_msk.nii.gz          # Instance segmentation ★
│   ├── {study_id}_seg-spine_msk.nii.gz         # Semantic segmentation
│   ├── {study_id}_seg-subreg_msk.nii.gz        # Sub-region masks
│   └── {study_id}_ctd.json                      # SPINEPS centroids
├── centroids/
│   ├── {study_id}_centroids.json                # 3D coordinates ★
│   └── {study_id}_vertebra_labels.json          # Instance → Label mapping
└── metadata/
    └── {study_id}_metadata.json                  # Processing log

★ = Critical for downstream fusion pipeline
```

---

## 🔬 Understanding the Outputs

### Instance Segmentation (`*_seg-vert_msk.nii.gz`)

Each vertebra has a unique integer label:

```
Cervical:   1-7   (C1-C7)
Thoracic:   8-19  (T1-T12)
Lumbar:     20-25 (L1-L6)      ← LSTV detection zone
Sacrum:     26
Discs:      120-124 (L1-L2 through L5-S1)
            126 (S1-S2 disc - sacralization marker)
```

**LSTV examples:**
- Normal: 20, 21, 22, 23, 24 (5 lumbar vertebrae)
- Lumbarization: 20, 21, 22, 23, 24, **25** (6 lumbar, L6 present)
- Sacralization: 20, 21, 22, 23 + **disc 126** (4 lumbar + S1-S2 disc)

### Centroids (`*_centroids.json`)

```json
{
  "vertebra_20": {
    "instance_id": 20,
    "anatomical_label": "L1",
    "centroid_voxel": [145.2, 256.8, 32.1],
    "centroid_world": [72.5, 128.4, 80.3],
    "volume_voxels": 4523
  },
  ...
}
```

Used for:
- Downstream uncertainty fusion
- Spatial analysis
- Visualization

---

## 🚀 Execution Modes

### Mode 0: Download Validation IDs (Run First!)

**When:** First time setup  
**Output:** `models/valid_id.npy` (283 study IDs)  
**Time:** ~2-3 minutes  

```bash
sbatch slurm_scripts/00_download_valid_ids.sh

# Verify
ls -lh models/valid_id.npy
python -c "import numpy as np; print(f'{len(np.load(\"models/valid_id.npy\"))} study IDs')"
```

### Mode 1: Trial (No Download)

**When:** Testing pipeline, data already exists  
**Studies:** 3 from `valid_id.npy`  
**Time:** ~30 minutes  

```bash
sbatch slurm_scripts/01_spineps_trial.sh
```

### Mode 2: Production (No Download)

**When:** Full processing, data already exists  
**Studies:** All 283 from `valid_id.npy`  
**Time:** ~18-24 hours  

```bash
sbatch slurm_scripts/02_spineps_production.sh
```

### Mode 3: Trial (With Download)

**When:** Fresh start, need dataset  
**Studies:** 3 from `valid_id.npy`  
**Time:** ~1-2 hours (includes download)  

```bash
sbatch slurm_scripts/03_spineps_trial_with_download.sh
```

### Mode 4: Production (With Download)

**When:** Fresh start, full processing  
**Studies:** All 283 from `valid_id.npy`  
**Time:** ~2-3 hours download + 18-24 hours processing  

```bash
sbatch slurm_scripts/04_spineps_production_with_download.sh
```

---

## ✅ Verification

### Check Pipeline Completed

```bash
cd spineps-segmentation

# Count outputs
SEG_COUNT=$(ls -1 results/spineps_segmentation/segmentations/*_seg-vert_msk.nii.gz 2>/dev/null | wc -l)
CENTROID_COUNT=$(ls -1 results/spineps_segmentation/centroids/*_centroids.json 2>/dev/null | wc -l)

echo "Segmentations: $SEG_COUNT"
echo "Centroids: $CENTROID_COUNT"
# Should both be 3 (trial) or 283 (production)
```

### Inspect Outputs

```bash
# View centroid file
cat results/spineps_segmentation/centroids/$(ls results/spineps_segmentation/centroids/ | head -1) | python -m json.tool

# Check instance mask labels
python -c "
import nibabel as nib
import numpy as np
seg = nib.load('results/spineps_segmentation/segmentations/STUDY_ID_seg-vert_msk.nii.gz')
labels = np.unique(seg.get_fdata().astype(int))
print(f'Unique labels: {labels}')
"
```

### Monitor Running Job

```bash
# Check job status
squeue -u $USER

# Tail logs
tail -f logs/spineps_*.out

# Check for errors
grep -i error logs/spineps_*.out
```

---

## 🔗 Integration with Downstream Pipelines

### For LSTV Uncertainty Detection

After SPINEPS completes, pass centroids to uncertainty pipeline:

```bash
cd /path/to/lstv-uncertainty-detection

# Edit SLURM script
nano slurm_scripts/03_prod_with_centroids.sh

# Set centroid directory:
CENTROID_DIR="/path/to/spineps-segmentation/results/spineps_segmentation/centroids"

# Run uncertainty pipeline
sbatch slurm_scripts/03_prod_with_centroids.sh
```

The uncertainty pipeline will:
1. Load SPINEPS centroids
2. Generate heatmaps from regression model
3. Sample heatmaps at SPINEPS centroid locations
4. Output uncertainty values aligned with anatomical positions

### For Visualization

```bash
# View segmentation overlaid on MRI
# (Use ITK-SNAP, 3D Slicer, or FSLeyes)

itksnap \
  -g results/spineps_segmentation/nifti/STUDY_ID_T2w.nii.gz \
  -s results/spineps_segmentation/segmentations/STUDY_ID_seg-vert_msk.nii.gz
```

---

## 🐛 Troubleshooting

### "No valid_ids loaded"

```bash
# Verify file exists
ls -lh models/valid_id.npy

# Check contents
python -c "
import numpy as np
ids = np.load('models/valid_id.npy')
print(f'Loaded {len(ids)} study IDs')
print(f'First 10: {ids[:10]}')
"
```

**Solution:** Obtain `valid_id.npy` from your validation split and place in `models/`

### "Kaggle download failed"

```bash
# Test Kaggle CLI
kaggle competitions list

# Re-configure credentials
kaggle config set -n username YOUR_USERNAME
kaggle config set -n key YOUR_API_KEY

# Manual download
kaggle competitions download -c rsna-2024-lumbar-spine-degenerative-classification
```

### "SPINEPS segmentation failed"

```bash
# Check SPINEPS container
singularity exec --nv ~/singularity_cache/spine-level-ai-spineps.sif spineps --version

# Check GPU availability
nvidia-smi

# Review detailed logs
less logs/spineps_*.err
```

### "Wrong number of outputs"

```bash
# Expected: 283 studies (from valid_id.npy)
# If different, check:

# 1. Valid IDs actually loaded
grep "validation study IDs" logs/spineps_*.out

# 2. Studies found in dataset
grep "Filtered to" logs/spineps_*.out

# 3. Processing errors
grep -E "(failed|error)" logs/spineps_*.out
```

---

## 📝 Development

### Building Docker Containers

```bash
# Preprocessing container
cd docker
./build_preprocessing.sh

# SPINEPS container
./build_spineps.sh

# Push to Docker Hub
docker push go2432/spineps-preprocessing:latest
docker push go2432/spineps-segmentation:latest
```

### Testing Changes

```bash
# Test on single study
python scripts/run_spineps_segmentation.py \
  --input_dir data/raw/train_images \
  --series_csv data/raw/train_series_descriptions.csv \
  --nifti_dir results/test/nifti \
  --seg_dir results/test/segmentations \
  --metadata_dir results/test/metadata \
  --valid_ids models/valid_id.npy \
  --limit 1 \
  --mode debug
```

---

## 📚 References

- **SPINEPS:** [SPINEPS GitHub](https://github.com/Hendrik-code/spineps)
- **RSNA 2024 Dataset:** [Kaggle Competition](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification)
- **LSTV Detection:** [Related Research]

---

## 📄 License

MIT License - See LICENSE file

---

## 🙏 Acknowledgments

- SPINEPS team for the segmentation framework
- RSNA for the lumbar spine dataset
- Original spine-level-ai repository contributors

---

## 📧 Contact

**Gregory Schwing, MD-PhD**  
- GitHub: [@Gregory-Schwing-MD-PhD](https://github.com/Gregory-Schwing-MD-PhD)
- Email: go2432@wayne.edu

---

## 🔄 Version

**v1.0.0** - Initial release (February 2026)

---

**Ready to segment spines! 🦴**
