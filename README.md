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

## M1-friendly settings (already applied)
- Reduced scalogram scales to 32
- Downsample factor 2
- Larger stride and capped segments per file
- Smaller batch size
- Tiny-run mode available for fast sanity checks (`tiny_run: true`)

## Outputs
- `outputs/metrics_abp.json`, `outputs/metrics_ppg.json`
- `outputs/baseline_abp.json`, `outputs/baseline_ppg.json`
- `outputs/resnet50_abp*.pt`, `outputs/resnet50_ppg*.pt`

## License
MIT — see `LICENSE`.
