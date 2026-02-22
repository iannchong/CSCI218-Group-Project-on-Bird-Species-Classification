import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

# -----------------------------
# CONFIG
# -----------------------------
TEST_DIR = "dataset/test"
RESULTS_DIR = "results_efficientnet_b3"   # change if needed
Y_TRUE_PATH = os.path.join(RESULTS_DIR, "y_true.npy")
Y_PRED_PATH = os.path.join(RESULTS_DIR, "y_pred.npy")

IMG_SIZE = 300
SHOW_PER_DIR = 6

# Load arrays + test dataset
y_true = np.load(Y_TRUE_PATH)
y_pred = np.load(Y_PRED_PATH)

# Use a light transform for display
disp_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

test_ds = datasets.ImageFolder(TEST_DIR, transform=disp_tfms)
class_names = test_ds.classes

assert len(test_ds) == len(y_true) == len(y_pred), \
    "Mismatch: y_true/y_pred length must match number of test images (check shuffle=False and correct files)."

num_classes = len(class_names)

# Confusion matrix
cm = np.zeros((num_classes, num_classes), dtype=np.int64)
for t, p in zip(y_true, y_pred):
    cm[int(t), int(p)] += 1

# ignore diagonal for confusion
cm_no_diag = cm.copy()
np.fill_diagonal(cm_no_diag, 0)

# Find MOST CONFUSED PAIR A<->B by combined confusion
# score(a,b) = cm[a,b] + cm[b,a]
best_a, best_b, best_score = -1, -1, -1
best_ab, best_ba = 0, 0

for a in range(num_classes):
    for b in range(a + 1, num_classes):
        score = cm_no_diag[a, b] + cm_no_diag[b, a]
        if score > best_score:
            best_score = score
            best_a, best_b = a, b
            best_ab = cm_no_diag[a, b]
            best_ba = cm_no_diag[b, a]

A = class_names[best_a]
B = class_names[best_b]

print("Most confused PAIR (combined both directions):")
print(f"  Pair: {A}  <->  {B}")
print(f"  {A} -> {B}: {best_ab}")
print(f"  {B} -> {A}: {best_ba}")
print(f"  Total confused: {best_score}")

# Collect indices for each direction
idx_A_to_B = np.where((y_true == best_a) & (y_pred == best_b))[0]
idx_B_to_A = np.where((y_true == best_b) & (y_pred == best_a))[0]

# Helper to plot a grid of examples
def plot_examples(indices, title, max_n=6):
    n = min(len(indices), max_n)
    cols = 3
    rows = (n + cols - 1) // cols
    plt.figure(figsize=(cols * 4.0, rows * 4.0))
    for i in range(n):
        idx = int(indices[i])
        img, _ = test_ds[idx]
        img = img.permute(1, 2, 0).numpy()
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.axis("off")
        plt.title(os.path.basename(test_ds.samples[idx][0]), fontsize=9)
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()

# Show images for both direction
plot_examples(
    idx_A_to_B,
    title=f"Most confused direction: TRUE {A} → PRED {B} (n={len(idx_A_to_B)})",
    max_n=SHOW_PER_DIR
)

plot_examples(
    idx_B_to_A,
    title=f"Reverse confusion: TRUE {B} → PRED {A} (n={len(idx_B_to_A)})",
    max_n=SHOW_PER_DIR
)
