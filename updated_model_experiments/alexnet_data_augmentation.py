import os
import numpy as np
from PIL import Image, ImageDraw
from medmnist import ChestMNIST
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    precision_score, recall_score, f1_score
)
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class AlexNet(nn.Module):
    def __init__(self, dropout: float = 0.5):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 96, kernel_size=11, stride=4),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # Block 2
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # Block 3
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),

            # Block 4
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),

            # Block 5
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),

            nn.Dropout(dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),

            nn.Linear(4096, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def _local_color(img: Image.Image, cx: int, cy: int, size: int) -> int:
    """Pick a color slightly brighter than the local neighbourhood."""
    arr = np.array(img)
    x0, y0 = max(cx - size, 0), max(cy - size, 0)
    x1, y1 = min(cx + size, arr.shape[1]), min(cy + size, arr.shape[0])
    local_mean = arr[y0:y1, x0:x1].mean()
    return int(min(local_mean + 30, 255))


def draw_circle(img: Image.Image, size=15, pos=(10, 10),
                color=None, blend_alpha=0.5) -> Image.Image:
    """Draw a filled circle and blend it back into the image."""
    img = img.copy()
    cx, cy = pos
    if color is None:
        color = _local_color(img, cx, cy, size)
    overlay = img.copy()
    draw = ImageDraw.Draw(overlay)
    r = size
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
    return Image.blend(img, overlay, alpha=blend_alpha)


class BinaryChestMNIST(Dataset):
    def __init__(self, split, spurious=False, spurious_prob=1.0,
                 shape_size=15, blend_range=(0.4, 0.5),
                 randomize_pos=False, transform=None, **kwargs):
        self.dataset = ChestMNIST(split=split, **kwargs)
        self.spurious = spurious
        self.spurious_prob = spurious_prob
        self.shape_size = shape_size
        self.blend_range = blend_range
        self.randomize_pos = randomize_pos
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def _get_pos(self, img_w, img_h):
        margin = self.shape_size + 4   # keep the shape fully inside the image
        if self.randomize_pos:
            cx = np.random.randint(margin, img_w - margin)
            cy = np.random.randint(margin, img_h - margin)
        else:
            cx, cy = margin, margin   # fixed top-left
        return cx, cy

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        binary_label = 1 if label.any() else 0

        if self.spurious and binary_label == 1:
            if np.random.random() < self.spurious_prob:
                w, h = img.size
                pos = self._get_pos(w, h)
                alpha = np.random.uniform(*self.blend_range)
                img = draw_circle(img, size=self.shape_size,
                                  pos=pos, blend_alpha=alpha)

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(binary_label, dtype=torch.long)


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    all_probs, all_preds, all_targets = [], [], []

    with torch.no_grad():
        for x, target in loader:
            x, target = x.to(device), target.to(device).float()
            logits = model(x).squeeze(1)
            total_loss += criterion(logits, target).item()
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            all_probs.extend(probs.cpu())
            all_preds.extend(preds.cpu())
            all_targets.extend(target.cpu())

    y_true = torch.stack(all_targets).numpy()
    y_pred = torch.stack(all_preds).numpy()
    y_prob = torch.stack(all_probs).numpy()

    return {
        "loss":      total_loss / len(loader),
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "auc":       roc_auc_score(y_true, y_prob),
    }

if __name__ == "__main__":
    BATCH_SIZE    = 32
    EPOCHS        = 15
    LR            = 1e-4
    WEIGHT_DECAY  = 1e-4
    DROPOUT       = 0.3
    SPURIOUS      = True
    GRAD_CLIP     = 1.0

    # ── circle_small_subtle config (matches sweep) ────────────
    SHAPE_SIZE    = 15                 # "small"
    BLEND_RANGE   = (0.4, 0.5)         # "subtle"
    SPURIOUS_PROB = 0.5

    CKPT_PATH     = (
        "best_alexnet_circle_small_subtle_data_augmentation.pt"
        if SPURIOUS else "best_alexnet_clean.pt"
    )
    NUM_WORKERS   = 4

    set_seed(10)

    train_transforms = v2.Compose([
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(),
        v2.RandomRotation(15),
        v2.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.RandomErasing(p=0.5, scale=(0.02, 0.15), ratio=(0.5, 2.0)),
        v2.RandomErasing(p=0.3, scale=(0.05, 0.20), ratio=(0.5, 2.0)),
    ])

    val_transforms = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])

    train_dataset = BinaryChestMNIST(
        split="train", spurious=SPURIOUS, spurious_prob=SPURIOUS_PROB,
        shape_size=SHAPE_SIZE, blend_range=BLEND_RANGE,
        randomize_pos=True, transform=train_transforms,
        download=True, size=224,
    )

    val_dataset = BinaryChestMNIST(
        split="val", spurious=SPURIOUS, spurious_prob=SPURIOUS_PROB,
        shape_size=SHAPE_SIZE, blend_range=BLEND_RANGE,
        randomize_pos=True, transform=val_transforms,
        download=True, size=224,
    )

    test_dataset = BinaryChestMNIST(
        split="test", spurious=False, spurious_prob=SPURIOUS_PROB,
        shape_size=SHAPE_SIZE, blend_range=BLEND_RANGE,
        randomize_pos=True, transform=val_transforms,
        download=True, size=224,
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    model = AlexNet(dropout=DROPOUT).to(device)

    criterion  = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-6)

    best_val_auc = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0

        for x, target in tqdm(train_loader, desc=f"Epoch {epoch:02d}/{EPOCHS}"):
            x, target = x.to(device), target.to(device).float()
            optimizer.zero_grad()
            loss = criterion(model(x).squeeze(1), target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()

        m = evaluate(model, val_loader, criterion)
        print(
            f"  Train loss: {epoch_loss / len(train_loader):.4f} | "
            f"Val loss: {m['loss']:.4f} | "
            f"Val acc: {m['accuracy']:.4f} | "
            f"Val precision: {m['precision']:.4f} | "
            f"Val recall: {m['recall']:.4f} | "
            f"Val F1: {m['f1']:.4f} | "
            f"Val AUC: {m['auc']:.4f}"
        )

        if m["auc"] > best_val_auc:
            best_val_auc = m["auc"]
            torch.save(model.state_dict(), os.path.join("..", "trained_models", CKPT_PATH))
            print(f"New best AUC {best_val_auc:.4f} — checkpoint saved.")


    print("\nLoading best checkpoint for test evaluation …")
    model.load_state_dict(torch.load(os.path.join("..", "trained_models", CKPT_PATH), map_location=device))

    m = evaluate(model, test_loader, criterion)
    print(
        f"\n{'─'*60}\n"
        f"  Test loss:  {m['loss']:.4f}\n"
        f"  Accuracy:   {m['accuracy']:.4f}\n"
        f"  Precision:  {m['precision']:.4f}\n"
        f"  Recall:     {m['recall']:.4f}\n"
        f"  F1:         {m['f1']:.4f}\n"
        f"  AUC:        {m['auc']:.4f}\n"
        f"{'─'*60}"
    )