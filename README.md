# W2CAD (Wavelet-to-CAD)

W2CAD is a research pipeline for **CAD vs NONCAD classification** from **ABP** and **PPG** signals. Each modality is modeled **separately** using **scalograms** (CWT) and **ResNet50** transfer learning (ImageNet pretrained).

## Project structure
- `src/` — data loading, scalograms, dataset utilities
- `scripts/` — index building, training, baselines
- `configs/` — ABP/PPG experiment configs
- `docs/` — methods and publication readiness
- `colab_full_run.ipynb` — full dataset training in Colab GPU

## Data (Google Drive, public links)
Download and place into these folders at repo root:
- `ABPCAD/` — https://drive.google.com/drive/folders/1mnQmZqsDN9K-ig4bu4GZdUTccNkm_95h?usp=drive_link
- `ABPNONCAD/` — https://drive.google.com/drive/folders/14ab13FVVjORknHKlccxTho_ksPw0zBZn?usp=drive_link
- `ppg_CAD/` — https://drive.google.com/drive/folders/1Rtz7YFRQ2544B1LvC7RTO9wWxMp-bu6O?usp=drive_link
- `ppg_NONCAD/` — https://drive.google.com/drive/folders/1ogk72b6ppHUwJy-jPqu_HzZJ9cVgeAEV?usp=drive_link

Data and trained models are intentionally **ignored by git**. Use the links above or the Colab notebook to download.

## Setup
```bash
pip install -r requirements.txt
```

## Build indices (subject-level split)
```bash
python scripts/make_index.py --config configs/abp.yaml
python scripts/make_index.py --config configs/ppg.yaml
```

## Train ResNet50 on scalograms
```bash
python scripts/train_resnet50.py --config configs/abp.yaml
python scripts/train_resnet50.py --config configs/ppg.yaml
```

## Classical baselines (CPU-friendly)
```bash
python scripts/baseline_features.py --config configs/abp.yaml
python scripts/baseline_features.py --config configs/ppg.yaml
```

## Colab full run
Open `colab_full_run.ipynb` in Colab, enable GPU, and run top-to-bottom. The notebook downloads all four datasets via `gdown`, builds indices, trains models, and prints metrics.

## Research-grade report & figures
After training, generate a full methods/results report and figures:
```bash
python scripts/generate_report.py --configs configs/abp.yaml configs/ppg.yaml --out docs/REPORT.md
```
Figures are saved to `outputs/figures/` (training curves, ROC, PR, confusion matrix).

## M1-friendly settings (already applied)
- Reduced scalogram scales to 32
- Downsample factor 2
- Larger stride and capped segments per file
- Smaller batch size
- Tiny-run mode available for fast sanity checks (`tiny_run: true`)

## 3D UNet + Monte Carlo Dropout (ADAM dataset)

This project also includes a **3D UNet** with **Monte Carlo Dropout** for volumetric medical image segmentation, designed for the **ADAM challenge** (Aneurysm Detection And segMentation) using TOF-MRA 3D volumes.

### What is Monte Carlo Dropout?
Standard dropout is turned off at test time.  MC Dropout keeps dropout **active** during inference and runs **T stochastic forward passes** per input.  The T softmax outputs form an empirical posterior, giving us:

- **Mean prediction** — averaged softmax (better calibrated)
- **Predictive entropy** — total uncertainty (aleatoric + epistemic)
- **Mutual information** — epistemic uncertainty only
- **Variance map** — per-voxel prediction variance

This is especially valuable in medical imaging where **knowing what the model does NOT know** is as important as its prediction.

### ADAM dataset layout
```
data/adam/
├── training/
│   ├── 001/
│   │   ├── TOF.nii.gz          # TOF-MRA volume
│   │   └── aneurysms.nii.gz    # binary segmentation mask
│   ├── 002/ ...
└── testing/
    ├── 051/
    │   └── TOF.nii.gz
    ...
```

### Train 3D UNet with MC Dropout
```bash
python scripts/train_unet3d_mc.py --config configs/adam_unet3d.yaml
```

The training script:
1. Discovers subject folders and splits into train / val / test
2. Trains a 3D UNet (Dice + BCE loss, Adam optimiser, LR scheduling, early stopping)
3. At test time, runs MC Dropout with T=20 forward passes per sliding-window patch
4. Saves segmentation maps, uncertainty maps (entropy + mutual information), metrics, and figures

### Key configuration (`configs/adam_unet3d.yaml`)
| Parameter | Default | Description |
|---|---|---|
| `model_preset` | `"base"` | `"small"` (0.3M), `"base"` (4.5M), `"large"` (18M params) |
| `dropout_rate` | `0.15` | Dropout probability (kept active for MC inference) |
| `mc_samples` | `20` | T = number of stochastic forward passes |
| `patch_size` | `[64,64,64]` | 3D patch size for training and inference |
| `foreground_ratio` | `0.5` | Fraction of training patches centered on aneurysm voxels |

### Architecture overview
```
Input (1×D×H×W) ──► Encoder (4 stages) ──► Bottleneck ──► Decoder (4 stages) ──► Output (2×D×H×W)
         │                                                      ▲
         └──────────── skip connections ────────────────────────┘
Each conv block: Conv3D → BatchNorm → ReLU → Dropout3D (×2)
```

### 3D UNet + MC Dropout outputs
- `outputs/adam_unet3d/unet3d_best.pt` — best checkpoint (val Dice)
- `outputs/adam_unet3d/mc_metrics.json` — per-subject Dice + uncertainty metrics
- `outputs/adam_unet3d/mc_summary.json` — aggregated MC Dropout results
- `outputs/adam_unet3d/mc_predictions/` — NIfTI segmentation + uncertainty maps
- `outputs/adam_unet3d/figures/` — training curves, MC summary bar charts

## Outputs (scalogram pipeline)
- `outputs/metrics_abp.json`, `outputs/metrics_ppg.json`
- `outputs/baseline_abp.json`, `outputs/baseline_ppg.json`
- `outputs/resnet50_abp*.pt`, `outputs/resnet50_ppg*.pt`

## License
MIT — see `LICENSE`.
