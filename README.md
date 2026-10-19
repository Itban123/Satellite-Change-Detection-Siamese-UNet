# Satellite Building Change Detection Using Siamese U-Net

**Deep Learning | Computer Vision | Remote Sensing | Semantic Segmentation | PyTorch**

## Overview

This project implements a **Siamese U-Net for automated building change detection** from bi-temporal satellite imagery using the LEVIR-CD+ dataset.

The model analyzes satellite images of the same geographical area captured at different times and generates a pixel-level binary segmentation mask identifying building-related changes.

Unlike conventional image segmentation, change detection requires the model to distinguish actual structural changes from irrelevant visual differences such as illumination, vegetation, seasonal variations, and shadows.

The implementation uses a shared convolutional encoder, multi-scale feature differencing, and a U-Net decoder. The network is trained from scratch using PyTorch.

## Key Results

The Siamese U-Net achieved the following results on the held-out validation set:

| Metric | Validation Score |
|---|---:|
| **Intersection over Union (IoU)** | **0.7302** |
| **F1 / Dice Score** | **0.8441** |
| Precision | 0.8091 |
| Recall | 0.8823 |
| Best Epoch | 9 |
| Training Time | 34.06 minutes |

All results correspond to the building-change class at a prediction threshold of 0.5.

**Note:** These are validation results. The official test set was not evaluated.

## Dataset

**Dataset:** [LEVIR-CD+ — Kaggle](https://www.kaggle.com/datasets/mdrifaturrahman33/levir-cd-change-detection)

LEVIR-CD+ is a high-resolution remote sensing dataset designed for building change detection using paired satellite images.

| Property | Details |
|---|---|
| Total Image Pairs | 985 |
| Official Training Pairs | 637 |
| Official Test Pairs | 348 |
| Image Resolution | 1024 × 1024 |
| Image Channels | RGB |
| Ground Truth | Binary Change Masks |
| Segmentation Classes | Change / No Change |

Each sample contains three corresponding files:

- **A:** Satellite image before the change.
- **B:** Satellite image after the change.
- **label:** Ground-truth building change mask.

### Exploratory Data Analysis

Analysis of the 637 official training pairs revealed substantial class imbalance.

| Statistic | Result |
|---|---:|
| Changed Pixels | 4.651% |
| Unchanged Pixels | 95.349% |
| Completely Unchanged Masks | 65 |
| Total 256 × 256 Patches | 10,192 |
| Patches Containing Changes | 4,538 |
| Zero-Change Patches | 5,654 |

The dataset also exhibits significant appearance differences between before and after images, making robust structural change detection challenging.

## Data Preprocessing

The preprocessing pipeline includes:

1. **Data integrity verification:** Validate corresponding filenames, image dimensions, and annotation masks.
2. **Mask preprocessing:** Convert grayscale annotation masks into binary labels using a threshold of 127.
3. **Image-level splitting:** Divide the 637 official training pairs into training and validation subsets before patch extraction.
4. **Patch extraction:** Divide each 1024 × 1024 image into 16 non-overlapping 256 × 256 patches.
5. **Normalization:** Apply ImageNet RGB mean and standard deviation normalization.
6. **Data augmentation:** Apply synchronized horizontal flips, vertical flips, and 90-degree rotations to paired images and masks, alongside mild brightness and contrast augmentation during training.

### Final Development Split

| Split | Image Pairs | Patches | Changed Pixel Rate |
|---|---:|---:|---:|
| Training | 509 | 8,144 | 4.680% |
| Validation | 128 | 2,048 | 4.538% |
| Official Test | 348 | Not evaluated | Not evaluated |

The split is performed at the original image level to prevent patches from the same source image appearing in both training and validation.

## Model Architecture

### Siamese U-Net with Multi-Scale Feature Differencing

The architecture is inspired by the **FC-Siam-Diff** approach for bi-temporal image change detection.

**1. Shared Encoder**

Before and after images are processed using the same convolutional encoder with shared parameters.

The encoder consists of five convolutional stages using convolution, Batch Normalization, ReLU activation, and max pooling.

**2. Multi-Scale Feature Differencing**

Corresponding feature maps from the two temporal images are compared using absolute differences:

`Difference = |Features_before - Features_after|`

This allows the model to represent changes at multiple spatial resolutions.

**3. U-Net Decoder**

The decoder progressively upsamples the fused representations and incorporates multi-scale difference features through skip connections.

**4. Change Prediction**

A final 1 × 1 convolution produces a single-channel change logit map. Sigmoid activation converts logits into change probabilities, which are thresholded at 0.5 to generate the binary segmentation mask.

### Architecture Configuration

| Component | Configuration |
|---|---|
| Architecture | Siamese U-Net |
| Encoder | Shared CNN |
| Encoder Stages | 5 |
| Initial Feature Channels | 24 |
| Feature Fusion | Absolute Difference |
| Decoder | U-Net with Skip Connections |
| Input | Two RGB Images |
| Output | Single-Channel Change Map |
| Trainable Parameters | 4,416,673 |
| Initialization | Trained from Scratch |

No pretrained encoder or cross-attention mechanism was used.

## Training Configuration

| Hyperparameter | Value |
|---|---|
| Framework | PyTorch |
| Optimizer | AdamW |
| Initial Learning Rate | 0.0003 |
| Weight Decay | 0.0001 |
| Loss Function | Weighted BCE + Dice Loss |
| BCE Positive Weight | 2.0 |
| DataLoader Batch Size | 32 |
| GPU Micro-Batch Size | 8 |
| Mixed Precision | Enabled |
| Maximum Epochs | 35 |
| Early Stopping Patience | 7 |
| Checkpoint Selection | Best Validation IoU |
| Random Seed | 42 |

Gradient accumulation allows smaller GPU micro-batches while maintaining the intended effective optimization batch size.

The best model was selected based on validation IoU at epoch 9. Training stopped at epoch 16.

## Evaluation

The model is evaluated using change-class pixel-level segmentation metrics:

- **IoU:** Measures overlap between predicted and ground-truth changed pixels.
- **F1 / Dice:** Measures segmentation agreement.
- **Precision:** Measures the proportion of predicted changes that are correct.
- **Recall:** Measures how many ground-truth changed pixels are successfully detected.

### Best Validation Results

| Metric | Score |
|---|---:|
| IoU | 0.7302 |
| F1 / Dice | 0.8441 |
| Precision | 0.8091 |
| Recall | 0.8823 |
| False Positive Pixels | 1,268,205 |
| False Negative Pixels | 716,918 |

Validation metrics are aggregated over all 2,048 validation patches.

The model demonstrates effective building change detection, although false positives and missed building-change pixels remain areas for improvement.

## Implementation

The complete workflow is implemented in a Kaggle Notebook and includes:

- Dataset exploration and visualization
- Data integrity and annotation analysis
- Image-level training/validation splitting
- Patch-based preprocessing and augmentation
- Siamese U-Net implementation
- Mixed-precision model training
- Early stopping and validation
- Training curve visualization
- Ground-truth versus predicted change-mask visualization

### How to Run

1. Open the notebook in Kaggle.
2. Add the [LEVIR-CD+ dataset](https://www.kaggle.com/datasets/mdrifaturrahman33/levir-cd-change-detection) as a notebook input.
3. Enable GPU acceleration in Kaggle Notebook settings.
4. Run the notebook cells sequentially, starting with dataset inspection and preprocessing.
5. Run the Siamese U-Net training section to reproduce the training and validation pipeline.
6. Inspect the validation metrics and predicted segmentation masks.

The project uses Python, PyTorch, NumPy, Pandas, OpenCV, Pillow, Albumentations, Matplotlib, scikit-learn, and tqdm.

The original dataset is not redistributed in this repository.

## Limitations

Although the Siamese U-Net achieves promising validation performance, several limitations remain:

- **Validation instability:** Validation IoU fluctuated during training despite improvements in training loss.
- **Class imbalance:** Building-change pixels represent only a small fraction of the dataset.
- **False positives:** Illumination and environmental differences can resemble genuine structural changes.
- **Missed changes:** Some small or ambiguous building-change regions may be difficult to segment.
- **Generalization:** Geographic independence between distinct source images was not independently verified.
- **Evaluation scope:** Results are from a single development split and training seed; the official test set remains unevaluated.

The project is an experimental implementation rather than a production-ready geospatial monitoring system.

## References

1. Caye Daudt, R., Le Saux, B., & Boulch, A. (2018). [Fully Convolutional Siamese Networks for Change Detection](https://arxiv.org/abs/1810.08462). IEEE International Conference on Image Processing (ICIP).

2. Chen, H., & Shi, Z. (2020). [A Spatial-Temporal Attention-Based Method and a New Dataset for Remote Sensing Image Change Detection](https://doi.org/10.3390/rs12101662). *Remote Sensing*, 12(10), 1662.

3. [LEVIR-CD+ Dataset Repository](https://github.com/S2Looking/Dataset/tree/main/LEVIR-CD%2B/LEVIR-CD%2B).

4. [LEVIR-CD+ Dataset — Kaggle](https://www.kaggle.com/datasets/mdrifaturrahman33/levir-cd-change-detection).

---

**Project Status:** Completed baseline implementation and validation. Official held-out test evaluation not performed.
