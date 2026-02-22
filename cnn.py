import os
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# CONFIG
TRAIN_DIR = r"dataset/train"
TEST_DIR  = r"dataset/test"
OUT_DIR   = r"results_cnn"
os.makedirs(OUT_DIR, exist_ok=True)

IMG_SIZE = 300
BATCH_SIZE = 16
EPOCHS = 40
LR = 1e-3
WEIGHT_DECAY = 1e-4
VAL_SPLIT = 0.2
SEED = 10


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_WORKERS = 2
PIN_MEMORY = (DEVICE == "cuda")

torch.manual_seed(SEED)
np.random.seed(SEED)

torch.backends.cudnn.benchmark = True

class ImprovedCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        def conv_bn_relu(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        self.features = nn.Sequential(
            conv_bn_relu(3, 64),
            conv_bn_relu(64, 64),
            nn.MaxPool2d(2),

            conv_bn_relu(64, 128),
            conv_bn_relu(128, 128),
            nn.MaxPool2d(2),

            conv_bn_relu(128, 256),
            conv_bn_relu(256, 256),
            nn.MaxPool2d(2),

            conv_bn_relu(256, 512),
            conv_bn_relu(512, 512),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# Metrics
def topk_correct(logits: torch.Tensor, y: torch.Tensor, k: int) -> int:
    topk = torch.topk(logits, k=k, dim=1).indices
    return (topk == y.view(-1, 1)).any(dim=1).sum().item()

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


# Train/Eval loops
def run_epoch(model, loader, criterion, optimizer=None, scaler=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total = 0
    top1 = 0
    top3 = 0

    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            if DEVICE == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(x)
                    loss = criterion(logits, y)
            else:
                logits = model(x)
                loss = criterion(logits, y)

            if is_train:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        bs = y.size(0)
        total += bs
        total_loss += loss.item() * bs
        top1 += (logits.argmax(dim=1) == y).sum().item()
        top3 += topk_correct(logits, y, k=3)

    return total_loss / total, top1 / total, top3 / total

@torch.no_grad()
def predict_all(model, loader):
    model.eval()
    all_true = []
    all_pred = []
    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        logits = model(x)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_pred.append(preds)
        all_true.append(y.numpy())
    return np.concatenate(all_true), np.concatenate(all_pred)


def main():
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

    full_train = datasets.ImageFolder(TRAIN_DIR, transform=train_tfms)
    class_names = full_train.classes
    num_classes = len(class_names)
    print("Num classes:", num_classes)

    # Train/val split
    n_total = len(full_train)
    n_val = int(n_total * VAL_SPLIT)
    n_train = n_total - n_val
    train_ds, val_ds = random_split(
        full_train,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED)
    )

    val_base = datasets.ImageFolder(TRAIN_DIR, transform=test_tfms)
    val_ds = torch.utils.data.Subset(val_base, val_ds.indices)

    test_ds = datasets.ImageFolder(TEST_DIR, transform=test_tfms)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    model = ImprovedCNN(num_classes=num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Drop LR when val loss stops improving
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, verbose=True
    )

    scaler = torch.amp.GradScaler("cuda") if DEVICE == "cuda" else None

    history = {
        "train_loss": [], "train_acc": [], "train_top3": [],
        "val_loss": [], "val_acc": [], "val_top3": [],
        "lr": []
    }

    best_val_acc = 0.0
    best_state = copy.deepcopy(model.state_dict())

    start = time.time()

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc, tr_top3 = run_epoch(model, train_loader, criterion, optimizer=optimizer, scaler=scaler)
        va_loss, va_acc, va_top3 = run_epoch(model, val_loader, criterion, optimizer=None, scaler=None)

        scheduler.step(va_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["train_top3"].append(tr_top3)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        history["val_top3"].append(va_top3)
        history["lr"].append(current_lr)

        print(f"Epoch {epoch:02d}/{EPOCHS} | "
              f"train loss {tr_loss:.4f} acc {tr_acc:.4f} top3 {tr_top3:.4f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.4f} top3 {va_top3:.4f} | "
              f"lr {current_lr:.2e}")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_state = copy.deepcopy(model.state_dict())
            save_path = os.path.join(OUT_DIR, "best_improved_cnn.pth")
            tmp = save_path + ".tmp"
            torch.save(best_state, tmp)
            os.replace(tmp, save_path)
            print(f" Saved best ImprovedCNN (val_acc={best_val_acc:.4f})")

    minutes = (time.time() - start) / 60
    print(f"\nTraining finished in {minutes:.1f} min. Best val acc: {best_val_acc:.4f}")

    # Evaluate on test
    model.load_state_dict(best_state)
    y_true, y_pred = predict_all(model, test_loader)

    test_acc = float(np.mean(y_true == y_pred))
    mp, mr, mf1 = macro_precision_recall_f1(y_true, y_pred, num_classes)
    cm = confusion_matrix_np(y_true, y_pred, num_classes)
    acc_pc, total_pc = per_class_accuracy(y_true, y_pred, num_classes)

    print("\n=== IMPROVED CNN TEST RESULTS ===")
    print(f"Top-1 Accuracy:   {test_acc:.4f}")
    print(f"Macro Precision:  {mp:.4f}")
    print(f"Macro Recall:     {mr:.4f}")
    print(f"Macro F1:         {mf1:.4f}")

    # Save outputs
    np.save(os.path.join(OUT_DIR, "history.npy"), history, allow_pickle=True)
    np.save(os.path.join(OUT_DIR, "y_true.npy"), y_true)
    np.save(os.path.join(OUT_DIR, "y_pred.npy"), y_pred)
    np.save(os.path.join(OUT_DIR, "confusion_matrix.npy"), cm)
    np.save(os.path.join(OUT_DIR, "per_class_acc.npy"), acc_pc)

    plot_learning_curves(history, "Improved CNN", OUT_DIR)
    plot_confusion_matrix(cm, "Improved CNN Confusion Matrix", os.path.join(OUT_DIR, "confusion_matrix_crop.png"))
    plot_per_class_worst(acc_pc, class_names, "Improved CNN", os.path.join(OUT_DIR, "worst_classes.png"), top_k=20)
    plot_accuracy_distribution(acc_pc, "Improved CNN Per-class Accuracy Distribution",
                               os.path.join(OUT_DIR, "per_class_acc_hist.png"))

    print(f"\nSaved plots + arrays to: {OUT_DIR}")


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()
