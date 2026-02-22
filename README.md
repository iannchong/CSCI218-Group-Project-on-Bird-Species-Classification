# Bird Species Classification (CUB-200-2011) — CNN vs ResNet-18 vs EfficientNet-B3

This project trains and compares three image classification models on the **CUB-200-2011** bird dataset (200 classes):

- **ImprovedCNN** (custom baseline)
- **ResNet-18** (pretrained)
- **EfficientNet-B3** (pretrained)

Each training script saves:
- Best model checkpoint (`.pth`)
- Training history (`history.npy`)
- Predictions (`y_true.npy`, `y_pred.npy`)
- Confusion matrix + per-class accuracy (`.npy`)
- Plots (learning curves, confusion matrix crop, worst classes, per-class accuracy histogram)

---

## Project structure

```
.
├─ CUB_200_2011/                 # original downloaded dataset folder
├─ dataset/
│  ├─ train/                     # created by prepare_dataset.py
│  └─ test/                      # created by prepare_dataset.py
├─ prepare_dataset.py
├─ cnn.py
├─ resnet.py
├─ efficientnet.py
└─ evaluations.py
```

---

## Setup

### Requirements
- Python 3.9+ recommended
- PyTorch + torchvision
- numpy, matplotlib

Install (pip):
```bash
pip install torch torchvision numpy matplotlib
```

> If you have an NVIDIA GPU + CUDA, PyTorch will use it automatically when available.

---

## Prepare the dataset

1. Download and extract **CUB_200_2011** so it contains:
   - `CUB_200_2011/images/`
   - `CUB_200_2011/train_test_split.txt`
   - `CUB_200_2011/image_class_labels.txt`
   - `CUB_200_2011/images.txt`
   - `CUB_200_2011/classes.txt`

2. Run:
```bash
python prepare_dataset.py
```

This creates:
```
dataset/train/<class_name>/*.jpg
dataset/test/<class_name>/*.jpg
```

---

## Train models

### Improved CNN (custom baseline)
```bash
python cnn.py
```

Outputs saved to:
```
results_cnn/
  best_cnn.pth
  history.npy
  y_true.npy
  y_pred.npy
  confusion_matrix.npy
  per_class_acc.npy
  cnn_acc.png
  cnn_loss.png
  confusion_matrix_crop.png
  worst_classes.png
  per_class_acc_hist.png
```

### ResNet-18 (pretrained)
```bash
python resnet.py
```

Outputs saved to:
```
results_resnet18/
  best_resnet18.pth
  ...
```

### EfficientNet-B3 (pretrained)
```bash
python efficientnet.py
```

Outputs saved to:
```
results_efficientnet_b3/
  best_efficientnet_b3.pth
  ...
```

---

## Error analysis (most confused class pair)

After training and generating predictions, run:
```bash
python evaluations.py
```

By default, it reads from:
- `dataset/test`
- `results_efficientnet_b3/y_true.npy`
- `results_efficientnet_b3/y_pred.npy`

It will:
- build a confusion matrix
- find the **most confused pair of classes (A ↔ B)** (both directions combined)
- display a grid of misclassified images for each direction (A→B and B→A)

> To analyze a different model, change `RESULTS_DIR` inside `evaluations.py`.

---

## Notes on metrics

The evaluation focuses on:
- **Top-1 Accuracy** (overall)
- **Macro Precision / Macro Recall / Macro F1** (treats each class more evenly)
- **Per-class accuracy** (shows which species are hardest)

---

## Reproducibility
All scripts set a fixed seed (`SEED = 10`) for more consistent splitting and training behaviour.


## AI Use
AI tools were used to generate parts of this README.md and to assist with researching example code, debugging, and troubleshooting during development.