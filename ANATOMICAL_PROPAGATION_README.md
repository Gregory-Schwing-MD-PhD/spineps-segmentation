# LSTV Detection via Anatomical Label Propagation

## The Breakthrough

This isn't just LSTV detection — **it's spine re-mapping**.

### The Problem
SPINEPS counts vertebrae blindly: 1, 2, 3, 4, 5, 6... but doesn't know which bones they are.
- If there are 6 lumbar vertebrae, which one is actually L5?
- Is #6 a sacralized L5 or a lumbarized S1?
- The counting starts at T12 and drifts downward — any anomaly cascades errors.

### The Solution
Use Ian Pan's spine localizer as an **anatomical compass**:

1. **Verify Disc Identity**: At each SPINEPS disc centroid, run Ian Pan's model
   - High confidence + low entropy → verified disc identity (e.g., "This is L4-L5")
   - The model knows anatomy from training on thousands of labeled spines

2. **Propagate Labels**: Use verified discs as anchors to re-label adjacent vertebrae
   - Bone **above** L4-L5 disc → must be L4
   - Bone **below** L4-L5 disc → must be L5
   - Build outward from verified landmarks

3. **Detect LSTV**: High entropy at L5-S1 location → anatomical anomaly
   - Normal spine: sharp peak at channel 5 (L5-S1)
   - LSTV spine: model confused → high entropy → flags the transition

4. **Resolve Conflicts**: If SPINEPS found 6 lumbar vertebrae but Ian Pan only verifies 5 transitions
   - Programmatically merge/re-label the 6th instance
   - Mark as sacralized L5 or lumbarized S1

## Pipeline

```bash
# Step 1: SPINEPS segmentation (produces blind count)
sbatch slurm_scripts/02_spineps_segmentation.sh

# Step 2: Anatomical propagation (re-labels everything)
sbatch slurm_scripts/03_anatomical_propagation.sh

# Step 3: Generate comparison visualizations
sbatch slurm_scripts/04_visualize_corrections.sh

# Step 4: Create HTML audit gallery
python scripts/create_audit_gallery.py \
  --viz_dir results/anatomical_propagation/visualizations \
  --audit_json results/anatomical_propagation/audit_queue/high_priority_audit.json \
  --output audit_gallery.html
```

## Outputs

### 1. Anatomically Corrected Masks
`results/anatomical_propagation/anatomically_corrected/`
- `{study_id}_anatomically_corrected.nii.gz` — re-labeled instance masks
- Integer labels preserved (L5 = 24, Sacrum = 26)
- But now **anatomically anchored**, not blind count

### 2. Correction Reports
`results/anatomical_propagation/correction_reports/`
- `{study_id}_correction_report.json` — detailed change log

Example:
```json
{
  "study_id": "12345",
  "n_corrections": 2,
  "corrections": {
    "25": {
      "before": "L6",
      "after": "Sacrum",
      "reason": "Adjacent to verified L5_S1 (inferior)",
      "verified_disc": "L5_S1"
    },
    "24": {
      "before": "L5",
      "after": "L5",
      "reason": "Adjacent to verified L5_S1 (superior)",
      "lstv_flag": true,
      "l5s1_entropy": 6.142
    }
  },
  "disc_verifications": {
    "L5_S1": {
      "verified_as": "L5_S1",
      "confidence": 0.876,
      "entropy": 6.142,
      "is_verified": false,
      "is_lstv_signal": true
    }
  }
}
```

### 3. Audit Queue
`results/anatomical_propagation/audit_queue/high_priority_audit.json`
- Top-60 cases ranked by:
  1. Number of corrections made
  2. L5-S1 entropy (LSTV signal strength)

### 4. Metrics CSV
`results/anatomical_propagation/anatomical_correction_metrics.csv`

Columns:
- `study_id`
- `n_corrections` — how many labels changed
- `{disc}_verified_as` — what Ian Pan identified each disc as
- `{disc}_confidence` — peak probability
- `{disc}_entropy` — uncertainty measure
- `{disc}_verified` — boolean (entropy < 4.0)

### 5. Visual Comparisons
`results/anatomical_propagation/visualizations/`
- Side-by-side: Original SPINEPS vs Anatomically Corrected
- Color-coded instances
- Annotations showing which labels changed and why

## Technical Details

### Disc Verification Criteria
```python
ENTROPY_VERIFIED  = 4.0   # below → disc identity confirmed
ENTROPY_LSTV      = 5.0   # above → LSTV anomaly signal
```

- **Verified**: Entropy < 4.0 + high confidence → use for propagation
- **Ambiguous**: Entropy 4.0-5.0 → don't propagate from this disc
- **LSTV Signal**: Entropy > 5.0 at L5-S1 → anatomical variant

### Label Propagation Logic

For each verified disc:
1. Convert disc centroid voxel → world coordinates (mm)
2. Find vertebra instances within 50mm superior/inferior
3. Check which side of disc they're on (Z-coordinate in sagittal)
4. Re-label based on disc identity:
   - Superior bone gets upper vertebra name
   - Inferior bone gets lower vertebra name

Example: L4-L5 disc verified at (x, y, z=100)
- Vertebra at z=120 (above) → L4
- Vertebra at z=80 (below) → L5

### Coordinate Systems

**Critical**: Centroids from SPINEPS are in **voxel space** relative to the NIfTI it created.

The script:
1. Loads the same NIfTI SPINEPS processed
2. Uses its affine transform for voxel ↔ world conversion
3. Ensures all spatial calculations use the same reference frame

### What Makes a Neurosurgeon Trust This

Before: "The AI counted 6 lumbar vertebrae"
→ Surgeon: "But which one is L5?"
→ AI: *crickets*

After: "Ian Pan identified the L4-L5 and L5-S1 transitions with 92% and 87% confidence. Based on those landmarks, I re-labeled the instances. Here's why each label changed."
→ Surgeon: "Show me the comparison. Okay, I agree with L5. But that L6 - did you verify?"
→ AI: "L6 was originally labeled as such by SPINEPS, but the L5-S1 disc verification places it inferior to what the model confirmed as L5-S1, with high entropy suggesting a transitional anatomy. I've flagged it for your review."

## For Your Audit

When you review the top-60 cases:

1. **Open the HTML gallery**
   ```bash
   firefox audit_gallery.html
   ```

2. **Look for**:
   - Cases where SPINEPS counted 6 lumbar vertebrae → corrected to 5 + sacrum
   - High L5-S1 entropy (>5.0) with label corrections → potential LSTV
   - Disagreements between SPINEPS blind count and Ian Pan verification

3. **Ask yourself**:
   - Does the corrected label make anatomical sense?
   - Would I agree with Ian Pan's disc identification?
   - Does the high entropy at L5-S1 match what I see visually?

4. **Create ground truth**:
   - Mark cases as: Agree / Disagree / Uncertain
   - For disagreements: what's the correct label?
   - For LSTV flags: confirm yes/no

5. **Calculate real metrics**:
   - Correction accuracy: % of your "Agree" annotations
   - LSTV detection: sensitivity/specificity using your ground truth
   - Compare to Kwak et al. baseline (85.1% sens, 61.9% spec)

## Next Steps

After audit:
1. Generate confusion matrix (your labels vs AI labels)
2. Identify systematic errors (e.g., always mislabels L6?)
3. Iterate on thresholds if needed
4. Write the paper with **real performance metrics**

## Files

### Scripts
- `scripts/inference_centroid_propagation.py` — main pipeline
- `scripts/visualize_corrections.py` — generate comparison images
- `scripts/create_audit_gallery.py` — HTML gallery

### SLURM
- `slurm_scripts/03_anatomical_propagation.sh` — run propagation
- `slurm_scripts/04_visualize_corrections.sh` — generate viz

### Outputs
- `results/anatomical_propagation/` — all outputs
- `audit_gallery.html` — interactive review interface

---

**The Bottom Line**: You're not just detecting LSTV. You're building the first spine AI that can **name the bones it's looking at**, anchored to verified anatomical landmarks. That's what makes it publication-ready.
