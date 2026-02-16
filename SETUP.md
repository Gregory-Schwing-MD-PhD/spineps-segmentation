# SPINEPS Segmentation - Setup Guide

## 🚀 Quick Setup

### 1. Extract Repository

```bash
# Extract tarball
tar -xzf spineps-segmentation.tar.gz
cd spineps-segmentation

# Verify structure
ls -la
# Should see: docker/ scripts/ slurm_scripts/ src/ data/ models/ results/ logs/
```

### 2. Download Validation IDs

**Instead of copying, download from Kaggle:**

```bash
# Download validation IDs (283 study IDs)
sbatch slurm_scripts/00_download_valid_ids.sh

# Monitor
tail -f logs/download_valid_ids_*.out

# Verify
ls -lh models/valid_id.npy
python -c "import numpy as np; print(f'Loaded {len(np.load(\"models/valid_id.npy\"))} study IDs')"
# Should output: Loaded 283 study IDs
```

### 3. Configure Kaggle (if downloading data)

```bash
# Set up Kaggle credentials
mkdir -p ~/.kaggle
cp /path/to/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

# Test Kaggle CLI
kaggle competitions list
```

### 4. Choose Your Execution Path

| Mode | Download? | Command |
|------|-----------|---------|
| **Trial (recommended first)** | No | `sbatch slurm_scripts/01_spineps_trial.sh` |
| **Production** | No | `sbatch slurm_scripts/02_spineps_production.sh` |
| **Trial + Download** | Yes | `sbatch slurm_scripts/03_spineps_trial_with_download.sh` |
| **Production + Download** | Yes | `sbatch slurm_scripts/04_spineps_production_with_download.sh` |

---

## 📋 Pre-flight Checklist

Before running, verify:

- [ ] Kaggle credentials configured (`~/.kaggle/kaggle.json`)
- [ ] `models/valid_id.npy` downloaded (run script 00)
- [ ] Data downloaded OR Kaggle credentials configured
- [ ] Sufficient disk space (~200GB for data + 50GB for outputs)
- [ ] GPU access (V100 or better)
- [ ] Singularity/Apptainer installed

---

## 🎯 Recommended First Run

```bash
# 0. Download validation IDs (if not already done)
sbatch slurm_scripts/00_download_valid_ids.sh
# Wait ~2-3 minutes

# 1. Run trial mode (3 studies, ~30 minutes)
sbatch slurm_scripts/01_spineps_trial.sh

# 2. Monitor
tail -f logs/spineps_trial_*.out

# 3. Verify outputs
ls -lh results/spineps_segmentation/centroids/*.json | wc -l
# Should show: 3

# 4. If successful, run production
sbatch slurm_scripts/02_spineps_production.sh
```

---

## 📊 Expected Outputs

After completion:

```
results/spineps_segmentation/
├── nifti/                   (283 files)
├── segmentations/           (849 files: instance + semantic + subreg)
├── centroids/               (566 files: centroids + labels)
└── metadata/                (283 files)
```

**Total:** ~283 studies × 9 files/study = ~2,500 files

---

## 🔗 Next Steps

After SPINEPS completes, integrate with uncertainty detection:

```bash
cd /path/to/lstv-uncertainty-detection

# Point to SPINEPS centroids
CENTROID_DIR="/path/to/spineps-segmentation/results/spineps_segmentation/centroids"

# Run uncertainty pipeline
sbatch slurm_scripts/03_prod_with_centroids.sh \
    --centroid_dir $CENTROID_DIR
```

---

## 🐛 Troubleshooting

### Issue: "No valid_ids loaded"

```bash
# Check file exists
ls -lh models/valid_id.npy

# Verify contents
python -c "
import numpy as np
ids = np.load('models/valid_id.npy')
print(f'{len(ids)} study IDs')
print(ids[:5])
"
```

### Issue: "Data not found"

If using scripts 01 or 02 (no download):
```bash
# Manually download
python scripts/download_rsna_data.py --output_dir data/raw
```

### Issue: "Container not found"

```bash
# Manually pull container
cd ~/singularity_cache
singularity pull spineps-segmentation.sif docker://go2432/spineps-segmentation:latest
```

---

## ✅ Verification Commands

```bash
# Count outputs
SEG_COUNT=$(ls -1 results/spineps_segmentation/segmentations/*_seg-vert_msk.nii.gz 2>/dev/null | wc -l)
CENTROID_COUNT=$(ls -1 results/spineps_segmentation/centroids/*_centroids.json 2>/dev/null | wc -l)

echo "Segmentations: $SEG_COUNT / 283"
echo "Centroids: $CENTROID_COUNT / 283"

# View one centroid file
cat results/spineps_segmentation/centroids/$(ls results/spineps_segmentation/centroids/ | head -1) | python -m json.tool
```

---

**You're ready to go! 🦴**
