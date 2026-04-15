import os
import itertools
import csv
import random
from datetime import datetime

import numpy as np
import pandas as pd
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
    precision_score, recall_score, f1_score,
)
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ──────────────────────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────────────────────
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ──────────────────────────────────────────────────────────────
# Drawing helpers
# ──────────────────────────────────────────────────────────────
def _local_color(img, cx, cy, size):
    """Pick a color slightly brighter than the local neighbourhood."""
    arr = np.array(img)
    x0, y0 = max(cx - size, 0), max(cy - size, 0)
    x1, y1 = min(cx + size, arr.shape[1]), min(cy + size, arr.shape[0])
    local_mean = arr[y0:y1, x0:x1].mean()
    return int(min(local_mean + 30, 255))


def draw_star(img, size=20, pos=(10, 10), color=None, blend_alpha=0.5):
    img = img.copy()
    cx, cy = pos
    if color is None:
        color = _local_color(img, cx, cy, size)
    overlay = img.copy()
    draw = ImageDraw.Draw(overlay)
    outer, inner = size, size / 2.5
    points = []
    for i in range(10):
        angle = i * 36 - 90
        r = outer if i % 2 == 0 else inner
        rad = np.radians(angle)
        points.append((cx + r * np.cos(rad), cy + r * np.sin(rad)))
    draw.polygon(points, fill=color)
    return Image.blend(img, overlay, alpha=blend_alpha)


def draw_circle(img, size=20, pos=(10, 10), color=None, blend_alpha=0.5):
    img = img.copy()
    cx, cy = pos
    if color is None:
        color = _local_color(img, cx, cy, size)
    overlay = img.copy()
    draw = ImageDraw.Draw(overlay)
    r = size
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
    return Image.blend(img, overlay, alpha=blend_alpha)


def draw_wave(img, size=20, pos=(10, 10), color=None, blend_alpha=0.5):
    """Draw a horizontal sine-wave band centred at `pos`."""
    img = img.copy()
    cx, cy = pos
    if color is None:
        color = _local_color(img, cx, cy, size)
    overlay = img.copy()
    draw = ImageDraw.Draw(overlay)
    w, h = img.size
    amplitude = size * 0.4
    thickness = max(size // 4, 2)
    x_start = max(cx - size, 0)
    x_end = min(cx + size, w)
    freq = 2 * np.pi / size  # one full cycle across `size` pixels
    for x in range(x_start, x_end):
        y_centre = int(cy + amplitude * np.sin(freq * (x - cx)))
        y0 = max(y_centre - thickness // 2, 0)
        y1 = min(y_centre + thickness // 2, h - 1)
        for y in range(y0, y1 + 1):
            draw.point((x, y), fill=color)
    return Image.blend(img, overlay, alpha=blend_alpha)


SHAPE_FNS = {
    "star":   draw_star,
    "circle": draw_circle,
    "wave":   draw_wave,
}


# ──────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────
class AlexNet(nn.Module):
    def __init__(self, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11, stride=4),
            nn.BatchNorm2d(96), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384), nn.ReLU(inplace=True),

            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384), nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256 * 6 * 6, 4096), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4096, 4096), nn.ReLU(inplace=True),
            nn.Linear(4096, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


# ──────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────
class BinaryChestMNIST(Dataset):
    def __init__(self, split, spurious=False, spurious_prob=1.0,
                 shape="star", star_size=20, blend_range=(0.3, 0.7),
                 randomize_star_pos=False, transform=None, **kwargs):
        self.dataset = ChestMNIST(split=split, **kwargs)
        self.spurious = spurious
        self.spurious_prob = spurious_prob
        self.shape = shape
        self.star_size = star_size
        self.blend_range = blend_range
        self.randomize_star_pos = randomize_star_pos
        self.transform = transform
        self.draw_fn = SHAPE_FNS[shape]

    def __len__(self):
        return len(self.dataset)

    def _get_pos(self, img_w, img_h):
        margin = self.star_size + 4
        if self.randomize_star_pos:
            cx = np.random.randint(margin, img_w - margin)
            cy = np.random.randint(margin, img_h - margin)
        else:
            cx, cy = margin, margin
        return cx, cy

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        binary_label = 1 if label.any() else 0

        if self.spurious and binary_label == 1:
            if np.random.random() < self.spurious_prob:
                w, h = img.size
                pos = self._get_pos(w, h)
                alpha = np.random.uniform(*self.blend_range)
                img = self.draw_fn(img, size=self.star_size, pos=pos, blend_alpha=alpha)

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(binary_label, dtype=torch.long)


# ──────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────
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


# ──────────────────────────────────────────────────────────────
# Single-run training
# ──────────────────────────────────────────────────────────────
def train_one_config(cfg, train_loader, val_loader, epochs, ckpt_path):
    set_seed(cfg.get("seed", 42))
    model = AlexNet(dropout=cfg["dropout"]).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-6)

    best_val_auc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for x, target in tqdm(train_loader, desc=f"  epoch {epoch:02d}/{epochs}", leave=False):
            x, target = x.to(device), target.to(device).float()
            optimizer.zero_grad()
            loss = criterion(model(x).squeeze(1), target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()

        val_m = evaluate(model, val_loader, criterion)
        if val_m["auc"] > best_val_auc:
            best_val_auc = val_m["auc"]
            torch.save(model.state_dict(), ckpt_path)

    # reload best
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return model, criterion


# ──────────────────────────────────────────────────────────────
# Main sweep
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # ── Hyperparams (fixed across sweep) ──────────────────────
    EPOCHS       = 10
    BATCH_SIZE   = 32
    LR           = 1e-4
    WEIGHT_DECAY = 1e-4
    DROPOUT      = 0.3
    GRAD_CLIP    = 1.0
    NUM_WORKERS  = 4
    SPUR_PROB    = 0.5        # P(shortcut | positive label) during training
    IMG_SIZE     = 224
    SEED         = 42

    # ── Sweep axes ────────────────────────────────────────────
    shapes       = ["star", "circle", "wave"]
    sizes        = {"small": 15, "medium": 20, "large": 35}
    blend_ranges = {"subtle": (0.5, 0.7), "moderate": (0.7, 0.9)}

    # ── Output dirs ───────────────────────────────────────────
    model_dir = os.path.join("trained_models")
    os.makedirs(model_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_txt = os.path.join(model_dir, f"sweep_results_{timestamp}.txt")
    log_csv = os.path.join(model_dir, f"sweep_results_{timestamp}.csv")

    # ── Transforms ────────────────────────────────────────────
    train_transforms = v2.Compose([
        v2.RandomHorizontalFlip(),
        v2.RandomRotation(10),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])
    val_transforms = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])

    # ── Build all configs ─────────────────────────────────────
    configs = []
    for shape, (size_label, size_px), (blend_label, blend_range) in itertools.product(
        shapes, sizes.items(), blend_ranges.items()
    ):
        configs.append({
            "shape":       shape,
            "size_label":  size_label,
            "size_px":     size_px,
            "blend_label": blend_label,
            "blend_range": blend_range,
        })

    csv_rows = []

    with open(log_txt, "w", encoding="utf-8") as log:
        header = (
            f"Spurious Correlation Sweep — {timestamp}\n"
            f"{'═' * 70}\n"
            f"Fixed: epochs={EPOCHS}  batch={BATCH_SIZE}  lr={LR}  "
            f"wd={WEIGHT_DECAY}  dropout={DROPOUT}  spur_prob={SPUR_PROB}\n"
            f"{'═' * 70}\n\n"
        )
        log.write(header)
        print(header)

        for i, cfg in enumerate(configs, 1):
            run_name = f"{cfg['shape']}_{cfg['size_label']}_{cfg['blend_label']}"
            print(f"\n[{i}/{len(configs)}] ▸ {run_name}")

            # ── Shared dataset kwargs ─────────────────────────
            common_ds_kw = dict(
                download=True, size=IMG_SIZE,
                shape=cfg["shape"],
                star_size=cfg["size_px"],
                blend_range=cfg["blend_range"],
                randomize_star_pos=True,
            )

            # ── Clean test set (used to evaluate BOTH models) ─
            test_clean_ds = BinaryChestMNIST(
                split="test", spurious=False,
                transform=val_transforms, **common_ds_kw,
            )
            test_clean_loader = DataLoader(test_clean_ds, batch_size=BATCH_SIZE, shuffle=False,
                                           num_workers=NUM_WORKERS, pin_memory=True)

            train_cfg = dict(
                lr=LR, weight_decay=WEIGHT_DECAY, dropout=DROPOUT,
                grad_clip=GRAD_CLIP, seed=SEED,
            )

            # ── Model A: trained WITH spurious correlation ────
            ckpt_spur = os.path.join(model_dir, f"model_{run_name}_spur.pt")
            print(f"  Training with spurious correlation ...")

            train_spur_ds = BinaryChestMNIST(
                split="train", spurious=True, spurious_prob=SPUR_PROB,
                transform=train_transforms, **common_ds_kw,
            )
            val_spur_ds = BinaryChestMNIST(
                split="val", spurious=True, spurious_prob=SPUR_PROB,
                transform=val_transforms, **common_ds_kw,
            )
            train_spur_loader = DataLoader(train_spur_ds, batch_size=BATCH_SIZE, shuffle=True,
                                           num_workers=NUM_WORKERS, pin_memory=True)
            val_spur_loader   = DataLoader(val_spur_ds, batch_size=BATCH_SIZE, shuffle=False,
                                           num_workers=NUM_WORKERS, pin_memory=True)

            model_spur, criterion = train_one_config(
                train_cfg, train_spur_loader, val_spur_loader, EPOCHS, ckpt_spur,
            )
            m_spur = evaluate(model_spur, test_clean_loader, criterion)

            # ── Model B: trained WITHOUT spurious correlation ─
            ckpt_clean = os.path.join(model_dir, f"model_{run_name}_clean.pt")
            print(f"  Training without spurious correlation ...")

            train_clean_ds = BinaryChestMNIST(
                split="train", spurious=False,
                transform=train_transforms, **common_ds_kw,
            )
            val_clean_ds = BinaryChestMNIST(
                split="val", spurious=False,
                transform=val_transforms, **common_ds_kw,
            )
            train_clean_loader = DataLoader(train_clean_ds, batch_size=BATCH_SIZE, shuffle=True,
                                            num_workers=0, pin_memory=True)
            val_clean_loader   = DataLoader(val_clean_ds, batch_size=BATCH_SIZE, shuffle=False,
                                            num_workers=0, pin_memory=True)

            model_clean, criterion = train_one_config(
                train_cfg, train_clean_loader, val_clean_loader, EPOCHS, ckpt_clean,
            )
            m_clean = evaluate(model_clean, test_clean_loader, criterion)

            # ── Log ───────────────────────────────────────────
            block = (
                f"\n{'─' * 70}\n"
                f"Config: {run_name}\n"
                f"  shape={cfg['shape']}  size={cfg['size_px']}px ({cfg['size_label']})  "
                f"blend={cfg['blend_range']} ({cfg['blend_label']})\n\n"
                f"  Model A — trained WITH spurious cue (prob={SPUR_PROB})\n"
                f"    path: {ckpt_spur}\n"
                f"    clean test:  loss={m_spur['loss']:.4f}  acc={m_spur['accuracy']:.4f}  "
                f"prec={m_spur['precision']:.4f}  rec={m_spur['recall']:.4f}  "
                f"f1={m_spur['f1']:.4f}  auc={m_spur['auc']:.4f}\n\n"
                f"  Model B — trained WITHOUT spurious cue (clean)\n"
                f"    path: {ckpt_clean}\n"
                f"    clean test:  loss={m_clean['loss']:.4f}  acc={m_clean['accuracy']:.4f}  "
                f"prec={m_clean['precision']:.4f}  rec={m_clean['recall']:.4f}  "
                f"f1={m_clean['f1']:.4f}  auc={m_clean['auc']:.4f}\n\n"
                f"  Delta (spurious-trained minus clean-trained) on clean test:\n"
                f"    ΔAccuracy: {m_spur['accuracy'] - m_clean['accuracy']:+.4f}\n"
                f"    ΔAUC:      {m_spur['auc'] - m_clean['auc']:+.4f}\n"
                f"    ΔF1:       {m_spur['f1'] - m_clean['f1']:+.4f}\n"
                f"{'─' * 70}\n"
            )
            log.write(block)
            log.flush()
            print(block)

            csv_rows.append({
                "run_name":            run_name,
                "shape":               cfg["shape"],
                "size_label":          cfg["size_label"],
                "size_px":             cfg["size_px"],
                "blend_label":         cfg["blend_label"],
                "blend_lo":            cfg["blend_range"][0],
                "blend_hi":            cfg["blend_range"][1],
                "spur_model_path":     ckpt_spur,
                "clean_model_path":    ckpt_clean,
                # Model A (trained with spurious) evaluated on clean test
                "spur_train_loss":     m_spur["loss"],
                "spur_train_acc":      m_spur["accuracy"],
                "spur_train_prec":     m_spur["precision"],
                "spur_train_rec":      m_spur["recall"],
                "spur_train_f1":       m_spur["f1"],
                "spur_train_auc":      m_spur["auc"],
                # Model B (trained clean) evaluated on clean test
                "clean_train_loss":    m_clean["loss"],
                "clean_train_acc":     m_clean["accuracy"],
                "clean_train_prec":    m_clean["precision"],
                "clean_train_rec":     m_clean["recall"],
                "clean_train_f1":      m_clean["f1"],
                "clean_train_auc":     m_clean["auc"],
                # Deltas
                "delta_acc":           m_spur["accuracy"] - m_clean["accuracy"],
                "delta_auc":           m_spur["auc"] - m_clean["auc"],
                "delta_f1":            m_spur["f1"] - m_clean["f1"],
            })

    # ── Write CSV summary ─────────────────────────────────────
    df = pd.DataFrame(csv_rows)
    df.to_csv(log_csv, index=False)
    print(f"\nResults written to:\n  {log_txt}\n  {log_csv}")