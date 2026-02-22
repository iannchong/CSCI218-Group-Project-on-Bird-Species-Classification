import os
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
import matplotlib.pyplot as plt

# CONFIG
TRAIN_DIR = "dataset/train"
TEST_DIR  = "dataset/test"
OUT_DIR   = "results_resnet18"
os.makedirs(OUT_DIR, exist_ok=True)

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 25
LR = 1e-4
WEIGHT_DECAY = 1e-4
VAL_SPLIT = 0.2
SEED = 10
NUM_WORKERS = 2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.backends.cudnn.benchmark = True
torch.manual_seed(SEED)
np.random.seed(SEED)
PIN_MEMORY = (DEVICE == "cuda")

# Metrics
def macro_precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int):
    precisions, recalls, f1s = [], [], []
    for c in range(num_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2*p*r)/(p+r) if (p+r) > 0 else 0.0
        precisions.append(p); recalls.append(r); f1s.append(f1)
    return float(np.mean(precisions)), float(np.mean(recalls)), float(np.mean(f1s))

def confusion_matrix_np(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

def per_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int):
    correct = np.zeros(num_classes, dtype=np.int64)
    total = np.zeros(num_classes, dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        total[t] += 1
        if t == p:
            correct[t] += 1
    acc = np.divide(correct, total, out=np.zeros_like(correct, dtype=float), where=total != 0)
    return acc, total

# Plot helpers
def plot_learning_curves(history: dict, title_prefix: str, out_dir: str):
    epochs = np.arange(1, len(history["train_acc"]) + 1)

    plt.figure()
    plt.plot(epochs, history["train_acc"], label="train acc")
    plt.plot(epochs, history["val_acc"], label="val acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy")
    plt.title(f"{title_prefix} Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{title_prefix.lower().replace(' ','_')}_acc.png"))
    plt.close()

    plt.figure()
    plt.plot(epochs, history["train_loss"], label="train loss")
    plt.plot(epochs, history["val_loss"], label="val loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title(f"{title_prefix} Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{title_prefix.lower().replace(' ','_')}_loss.png"))
    plt.close()

def plot_confusion_matrix(cm: np.ndarray, title: str, out_path: str, max_classes_to_show=25):
    n = min(cm.shape[0], max_classes_to_show)
    cm_crop = cm[:n, :n]
    plt.figure(figsize=(8, 8))
    plt.imshow(cm_crop)
    plt.title(f"{title} (first {n} classes)")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_per_class_worst(acc: np.ndarray, class_names: list, title: str, out_path: str, top_k=20):
    idx = np.argsort(acc)[:top_k]
    plt.figure(figsize=(10, 5))
    plt.bar(np.arange(top_k), acc[idx])
    plt.xticks(np.arange(top_k), [class_names[i] for i in idx], rotation=90)
    plt.ylabel("Per-class accuracy")
    plt.title(f"{title} - Worst {top_k} classes")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_accuracy_distribution(acc: np.ndarray, title: str, out_path: str):
    plt.figure()
    plt.hist(acc, bins=20)
    plt.xlabel("Per-class accuracy")
    plt.ylabel("Number of classes")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

# Data transformation

train_tfms = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
])

test_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# -----------------------------
# Load datasets + split train/val

full_train = datasets.ImageFolder(TRAIN_DIR, transform=train_tfms)
class_names = full_train.classes
num_classes = len(class_names)
# print("Num classes:", num_classes)

val_base = datasets.ImageFolder(TRAIN_DIR, transform=test_tfms)

n_total = len(full_train)
n_val = int(n_total * VAL_SPLIT)
n_train = n_total - n_val

train_ds, val_ds = random_split(
    full_train,
    [n_train, n_val],
    generator=torch.Generator().manual_seed(SEED)
)
val_ds = torch.utils.data.Subset(val_base, val_ds.indices)

test_ds = datasets.ImageFolder(TEST_DIR, transform=test_tfms)

# Safety check: avoid label-index mismatch
assert test_ds.classes == class_names, "Train/Test class order mismatch! Check folder structure."

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)


# Model: ResNet-18 (pretrained)
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(DEVICE)

params = sum(p.numel() for p in model.parameters())

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

scaler = torch.amp.GradScaler("cuda") if DEVICE == "cuda" else None

def run_one_epoch(loader, train: bool):
    model.train() if train else model.eval()

    running_loss = 0.0
    running_correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            if DEVICE == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)

            if train:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        preds = outputs.argmax(dim=1)
        running_loss += loss.item() * images.size(0)
        running_correct += (preds == labels).sum().item()
        total += images.size(0)

    return running_loss / total, running_correct / total

@torch.no_grad()
def predict_all(loader):
    model.eval()
    all_true, all_pred = [], []
    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        logits = model(x)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_pred.append(preds)
        all_true.append(y.numpy())
    return np.concatenate(all_true), np.concatenate(all_pred)


# Training loop
def main():
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    best_val_acc = 0.0
    best_state = copy.deepcopy(model.state_dict())
    start = time.time()

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = run_one_epoch(train_loader, train=True)
        val_loss, val_acc = run_one_epoch(val_loader, train=False)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            save_path = os.path.join(OUT_DIR, "best_resnet18.pth")
            tmp = save_path + ".tmp"
            torch.save(best_state, tmp)
            os.replace(tmp, save_path)
            print(f"Saved best model (val_acc={best_val_acc:.4f})")

    elapsed = (time.time() - start) / 60
    print(f"Done. Best val acc: {best_val_acc:.4f} | Time: {elapsed:.1f} min")

    # Evaluate on TEST
    model.load_state_dict(best_state)
    y_true, y_pred = predict_all(test_loader)

    test_acc = float(np.mean(y_true == y_pred))
    mp, mr, mf1 = macro_precision_recall_f1(y_true, y_pred, num_classes)
    cm = confusion_matrix_np(y_true, y_pred, num_classes)
    acc_pc, _ = per_class_accuracy(y_true, y_pred, num_classes)

    print("\n=== RESNET-18 TEST RESULTS ===")
    print(f"Top-1 Accuracy:   {test_acc:.4f}")
    print(f"Macro Precision:  {mp:.4f}")
    print(f"Macro Recall:     {mr:.4f}")
    print(f"Macro F1:         {mf1:.4f}")

    # Save arrays + plots
    np.save(os.path.join(OUT_DIR, "history.npy"), history, allow_pickle=True)
    np.save(os.path.join(OUT_DIR, "y_true.npy"), y_true)
    np.save(os.path.join(OUT_DIR, "y_pred.npy"), y_pred)
    np.save(os.path.join(OUT_DIR, "confusion_matrix.npy"), cm)
    np.save(os.path.join(OUT_DIR, "per_class_acc.npy"), acc_pc)

    plot_learning_curves(history, "ResNet-18", OUT_DIR)
    plot_confusion_matrix(cm, "ResNet-18 Confusion Matrix",
                          os.path.join(OUT_DIR, "confusion_matrix_crop.png"))
    plot_per_class_worst(acc_pc, class_names, "ResNet-18",
                         os.path.join(OUT_DIR, "worst_classes.png"), top_k=20)
    plot_accuracy_distribution(acc_pc, "ResNet-18 Per-class Accuracy Distribution",
                               os.path.join(OUT_DIR, "per_class_acc_hist.png"))

    print(f"\nSaved plots + arrays to: {OUT_DIR}")

if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()
