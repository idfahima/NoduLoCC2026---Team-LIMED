# Nodulo

Lung nodule classification and localization on chest X-rays using a RadDINO backbone with a two-phase training strategy.

## Method

The pipeline has two consecutive phases:

1. **Classification (phase 1):** 5-fold stratified cross-validation. Each fold fine-tunes the **full RadDINO model** (backbone + linear classification head) end-to-end with Asymmetric Sigmoid Loss and balanced 50/50 positive/negative sampling. The best checkpoint per fold is merged into a single **greedy fold soup** that maximizes held-out F1.

2. **Localization (phase 2):** 5-fold stratified cross-validation, symmetrical with phase 1. Each fold initializes a HeatmapLocalizer from the per-fold classification checkpoint and trains on the corresponding training split. The localizer learns to produce heatmaps from annotated nodule coordinates using Gaussian targets and Focal Heatmap loss. Negatives and CAM-supervised unannotated positives use ASL supervision. The five fold checkpoints are merged into a **greedy fold soup** that maximizes the held-out localization score.

At inference, the classifier produces a nodule probability score. If the score exceeds a tuned threshold, the localizer extracts heatmap peaks as candidate nodule coordinates.

## Repository layout

```
src/nodulo/
  data/           dataset loading and preprocessing
  models/         RadDinoClassifier, HeatmapLocalizer
  training/       loss functions, training loop, greedy soup, inference
  scripts/
    train.py      training entry point
    infer.py      inference entry point
configs/
  default.yaml    main configuration
requirements.txt
Dockerfile
docs/
  SETUP.md        step-by-step reproduction guide
```

## Environment setup

### Local

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### Docker

```bash
docker build -t nodulo:latest .
```

## Training

Expected data layout at the repository root:

```
classification_labels.csv
localization_labels.csv
nih_filtered_images/
lidc_png_16_bit/
```

```bash
python -m nodulo.scripts.train --config configs/default.yaml
```

Outputs:

```
outputs/soups/classifier_soup.pt
outputs/soups/localizer_soup.pt
outputs/fold_*/phase1_classifier/classifier_best.pt
outputs/fold_*/phase2_localizer/localizer_best.pt
```

## Inference

```bash
python -m nodulo.scripts.infer \
  --config configs/default.yaml \
  --input-dir /path/to/test_images \
  --output-dir /path/to/predictions \
  --classifier-checkpoint outputs/soups/classifier_soup.pt \
  --localizer-checkpoint outputs/soups/localizer_soup.pt
```

Generated files:

- `classification_test_results.csv` — columns: `file_name`, `label`, `confidence`
- `localization_test_results.csv` — columns: `file_name`, `x`, `y`, `confidence`

`label` is `Nodule` or `No Finding`. Coordinates are in original image pixel space.

## Reproducibility

- Random seed fixed to `42` everywhere.
- Full configuration in `configs/default.yaml`.
- No local paths hardcoded in training or inference.
- See `docs/SETUP.md` for the full step-by-step guide.
