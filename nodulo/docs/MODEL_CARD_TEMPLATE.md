# Model card

## Method

Two-phase fine-tuning of a RadDINO (microsoft/rad-dino) backbone on chest X-rays.

**Phase 1 — classification:** 5-fold stratified cross-validation. Each fold fine-tunes the **full RadDINO model** (backbone + linear classification head) end-to-end using Asymmetric Sigmoid Loss with balanced 50/50 sampling. The five checkpoints are merged by greedy fold soup (iterative weight averaging that maximizes held-out F1). The decision threshold is tuned on the validation pool and stored in the checkpoint.

**Phase 2 — localization:** A HeatmapLocalizer is initialized from the classification soup backbone and trained with 5-fold stratified cross-validation. Supervision comes from Gaussian heatmaps centred on annotated nodule coordinates (Focal Heatmap loss); negatives and CAM-supervised unannotated positives use Asymmetric Sigmoid Loss. Unannotated positives receive weak supervision via CAM-derived pseudo-heatmaps. Each fold's held-out set (annotated positives + negatives) drives early stopping.

## Losses

- Phase 1: Asymmetric Sigmoid Loss (gamma_neg=4, gamma_pos=1, clip=0.05) with label smoothing (positive target 0.9, negative target 0.05)
- Phase 2 annotated: Focal Heatmap loss (alpha=2, beta=4) on Gaussian heatmap targets
- Phase 2 negative: Asymmetric Sigmoid Loss (gamma_neg=4, gamma_pos=1, clip=0.05) on zero heatmap targets
- Phase 2 unannotated: Asymmetric Sigmoid Loss (gamma_neg=4, gamma_pos=1, clip=0.05) on CAM pseudo-heatmap targets (warmed up over epochs)

## Outputs

- `classification_test_results.csv`: file_name, label (Nodule/No Finding), confidence
- `localization_test_results.csv`: file_name, x, y, confidence (pixel coordinates)

## Hardware

- 1 × NVIDIA Tesla P100 (16 GB)

## Training time

- Phase 1 (5-fold classifier): ~6 h per fold → ~30 h total
- Phase 2 (5-fold localizer): ~5 h per fold → ~25 h total
- Total end-to-end: ~55 h

## Inference time

- ~0.35 s per image on GPU (Tesla P100).

## Parameter count

- Classifier soup: 86.6 M parameters
- Localizer: 87.5 M parameters
