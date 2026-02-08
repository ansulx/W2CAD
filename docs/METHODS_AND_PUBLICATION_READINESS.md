# Methods Summary & Publication Readiness (Top-Tier Journal)

This document summarizes: (1) what the pipeline does (scalograms, ResNet, ImageNet), (2) whether it is sufficient for a good top-tier journal, and (3) concrete gaps and recommendations.

---

## 1. Scalograms — What Is Implemented

### 1.1 Signal → Scalogram

- **Input:** Fixed-length **windows** from ABP or PPG signals.
  - `window_size: 5000` samples per segment.
  - Segments are extracted with `stride: 10000`; missing values (`-32768`) are replaced by segment median.
- **Downsampling:** Each segment is downsampled by `downsample_factor: 2` → **2500 samples** per window (reduces compute, keeps main frequency content).
- **Continuous wavelet transform (CWT):**
  - Implemented in `src/scalogram.py` via **PyWavelets** `pywt.cwt`.
  - **Wavelet:** `morl` (Morlet).
  - **Scales:** `[1, 2, …, 32]` (32 scales) — controls time vs frequency resolution.
- **Power:** `power = |CWT coef|²`, then **log1p(power)** for dynamic range.
- **Output shape:** 2D array of shape `(n_scales, n_times)` = **(32, 2500)**.

### 1.2 Scalogram → “Image” for ResNet

- **Normalization:** Per-image **min–max** to [0, 1] (in `scalogram_to_image`).
- **Channels:** Same 2D array copied to **3 channels** (R = G = B) so ResNet gets a 3-channel input.
- **Tensor:** `(3, 32, 2500)` → in the model this is **resized to (3, 224, 224)** with bilinear interpolation to match ResNet’s expected input size.

So: **scalograms are done end-to-end** (CWT → log power → min–max → 3-channel → 224×224). The only “image” aspect is the 2D time–scale representation treated as an image for the CNN.

### 1.3 What to State in the Paper

- CWT with **Morlet** wavelet, **32 scales**, segment length **2500** (after downsampling from 5000).
- Log-power representation; per-scalogram min–max to [0,1]; replicated to 3 channels and resized to 224×224 for transfer learning.

---

## 2. ResNet and ImageNet — What Is Implemented

### 2.1 Model and Weights

- **Architecture:** **ResNet-50** from `torchvision.models`.
- **Weights:** **IMAGENET1K_V2** (ImageNet-1K pretrained, V2 training recipe).
- **Adaptation:** Only the **final fully connected layer** is replaced: `model.fc = nn.Linear(2048, 2)` for binary CAD vs NONCAD. All other layers keep ImageNet pretrained weights and are **fine-tuned** (no freezing).

So: **ResNet + ImageNet transfer learning is correctly used** (pretrained backbone, binary head, full fine-tuning).

### 2.2 Input Preprocessing

- Scalograms are normalized to [0, 1] and resized to **224×224**.
- **ImageNet mean/std normalization is not applied.** Pretrained ResNet is usually trained with ImageNet mean/std; for non-natural images (e.g. scalograms) many papers still use [0,1] or dataset-specific normalization. This is acceptable but should be **explicitly stated** in the methods, and optionally compared in an ablation (e.g. [0,1] vs ImageNet normalization).

### 2.3 Training

- **Optimizer:** Adam, `lr: 0.0001`.
- **Epochs:** 10 (with early stopping on validation AUC, patience 3, min_delta 0.001).
- **Loss:** Cross-entropy.
- **Device:** CUDA → MPS (M1) → CPU.

---

## 3. Are These Parameters and Design Sufficient for a Top-Tier Journal?

**Short answer:** The **core design is sound** (subject-level split, scalograms, ResNet transfer learning, sensible metrics and baselines). For a **top-tier, methodologically strict journal**, you would typically strengthen **evaluation rigor**, **reproducibility**, and **methodological detail** as below.

### 3.1 Already Strong (Good for Publication)

- **Subject-level split** (`make_index.py`): train/val/test by **subject**, not by segment → avoids leakage and is standard for clinical/physiological data.
- **Stratified splits** by label.
- **Metrics:** AUROC, accuracy, **AUPRC (AP)**, **sensitivity at 95% specificity** — all appropriate for binary medical classification.
- **Baselines:** Classical features + Logistic Regression and Random Forest — good for showing added value of deep learning.
- **Reproducibility:** Fixed seed (42) for splits and (in tiny run) stratified subsampling.
- **Clear separation:** ABP and PPG models and indices are separate (no fusion), as in your README.

### 3.2 Gaps to Address for Top-Tier

1. **Evaluation robustness**
   - **Single split:** Only one train/val/test split. Top-tier often expects **repeated splits** (e.g. 5-fold or 3× random) and **confidence intervals** (e.g. bootstrap or mean ± std over seeds/splits).
   - **Multiple seeds:** Report AUC (and other metrics) over **e.g. 3–5 seeds** with mean ± std.

2. **Training setup**
   - **Epochs:** 10 is low; with early stopping it may be enough, but many papers use a higher cap (e.g. 50–100) and rely on early stopping.
   - **Learning-rate schedule:** Add a scheduler (e.g. ReduceLROnPlateau or cosine) and mention it in methods.
   - **Data augmentation:** No augmentation on scalograms (e.g. small time shift, scale jitter, mild noise). Adding and abating it would strengthen the paper.

3. **Normalization**
   - **ImageNet mean/std:** Not applied. Either justify [0,1] in the methods or add an ablation with ImageNet normalization.

4. **Ablations (README mentions them; code does not)**
   - Scalogram: scale range, wavelet choice (e.g. Morlet vs Mexican Hat).
   - Window size, stride, missing-value handling.
   - Freeze backbone vs full fine-tuning.

5. **Statistical testing**
   - Compare your best model vs baselines with **statistical tests** (e.g. DeLong for AUC, or bootstrap CIs). Report p-values or CIs.

6. **Dataset and ethics**
   - Clearly describe dataset (N subjects, N segments, class balance, inclusion criteria).
   - If applicable: ethics approval, consent, and data availability statement.

---

## 4. One-Page Checklist for “Top-Tier Ready”

| Item | Status | Action |
|------|--------|--------|
| Subject-level split | ✅ | Keep; state in methods |
| AUROC, AUPRC, sensitivity@spec | ✅ | Keep; report on test set |
| Classical baselines | ✅ | Keep |
| ResNet50 + ImageNet pretrained | ✅ | State weights (IMAGENET1K_V2) |
| Scalogram (CWT, Morlet, scales, log power) | ✅ | Describe in methods |
| Input size 224×224, 3 channels | ✅ | State |
| Multiple splits or seeds | ❌ | Add 5-fold or 3–5 seeds, report mean ± std |
| Confidence intervals | ❌ | Bootstrap or over-seeds |
| LR schedule | ❌ | Add (e.g. ReduceLROnPlateau) |
| Data augmentation | ❌ | Optional; add and ablate |
| ImageNet normalization | ⚠️ | State choice; optionally ablate |
| Ablations (scales, wavelet, window) | ❌ | Implement and report |
| Statistical comparison vs baseline | ❌ | DeLong or bootstrap |
| Dataset description & ethics | ⚠️ | Add in paper |

---

## 5. Summary

- **Scalograms:** Implemented end-to-end: CWT (Morlet, 32 scales) on 2500-sample segments → log power → min–max → 3-channel → 224×224. This is appropriate to describe as “scalogram-based” representation for ResNet.
- **ResNet and ImageNet:** ResNet50 with ImageNet (IMAGENET1K_V2) pretrained weights is used; only the classifier head is replaced and the whole network is fine-tuned. This is standard transfer learning.
- **Parameters:** Sufficient for a **solid conference or mid-tier journal** as-is. For a **top-tier journal**, add: multiple splits or seeds with CIs, LR schedule, optional augmentation and normalization ablation, explicit ablations (scales, wavelet, window), and statistical comparison vs baselines.

If you want, next steps can be: (1) add ImageNet normalization as an option and document it, (2) add a simple LR scheduler and higher epoch cap, (3) add a small “repeated runs” script (multiple seeds) and bootstrap or mean ± std reporting.
