import os
import random
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from medmnist import ChestMNIST
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import v2
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import math

from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    precision_score, recall_score, f1_score
)
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate):
        super().__init__()
        self.batchnorm1 = nn.BatchNorm2d(num_input_features)
        self.conv1x1    = nn.Conv2d(num_input_features, 4 * growth_rate, kernel_size=1)
        self.batchnorm2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv3x3    = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1)

    def forward(self, x):
        out = F.relu(self.batchnorm1(x))
        out = F.relu(self.batchnorm2(self.conv1x1(out)))
        out = self.conv3x3(out)
        return torch.cat((x, out), dim=1)


class DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, growth_rate):
        super().__init__()
        self.layers = nn.ModuleList()
        in_features = num_input_features
        for _ in range(num_layers):
            self.layers.append(DenseLayer(in_features, growth_rate))
            in_features += growth_rate

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class TransitionLayer(nn.Module):
    def __init__(self, num_input_features, compression=0.5):
        super().__init__()
        out = math.floor(num_input_features * compression)
        self.bn      = nn.BatchNorm2d(num_input_features)
        self.conv1x1 = nn.Conv2d(num_input_features, out, kernel_size=1)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.avgpool(self.conv1x1(F.relu(self.bn(x))))

class DenseNet121(nn.Module):
    def __init__(self, growth_rate: int = 32, dropout: float = 0.5):
        super().__init__()

        # Initial conv
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # Dense blocks + transitions
        dense_block_layers = [6, 12, 24, 16]
        channel_count = 64
        self.blocks = nn.ModuleList()

        for idx, num_layers in enumerate(dense_block_layers):
            block = DenseBlock(num_layers, channel_count, growth_rate)
            channel_count += growth_rate * num_layers
            self.blocks.append(block)

            if idx != len(dense_block_layers) - 1:
                transition = TransitionLayer(channel_count, compression=0.5)
                channel_count = math.floor(channel_count * 0.5)
                self.blocks.append(transition)

        self.bn_final = nn.BatchNorm2d(channel_count)
        self.avgpool  = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout  = nn.Dropout(dropout)
        self.fc       = nn.Linear(channel_count, 1)

    def forward(self, x):
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = F.relu(self.bn_final(x))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)

def draw_star(img: Image.Image, size=20, pos=(10, 10), color=255) -> Image.Image:
    img = img.copy()
    draw = ImageDraw.Draw(img)
    cx, cy = pos
    outer, inner = size, size // 2.5
    points = []
    for i in range(10):
        angle = i * 36 - 90
        r = outer if i % 2 == 0 else inner
        rad = np.radians(angle)
        points.append((cx + r * np.cos(rad), cy + r * np.sin(rad)))
    draw.polygon(points, fill=color)
    return img

class BinaryChestMNIST(Dataset):
    def __init__(self, split, spurious=False, spurious_prob=1.0,
                 star_size=20, randomize_star_pos=False, transform=None, **kwargs):
        self.dataset = ChestMNIST(split=split, **kwargs)
        self.spurious = spurious
        self.spurious_prob = spurious_prob
        self.star_size = star_size
        self.randomize_star_pos = randomize_star_pos
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def _get_star_pos(self, img_w, img_h):
        margin = self.star_size + 4   # keep the star fully inside the image
        if self.randomize_star_pos:
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
                pos = self._get_star_pos(w, h)
                img = draw_star(img, size=self.star_size, pos=pos)
        
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
    SEED = 10
    set_seed(SEED)

    SPURIOUS = False
    BATCH_SIZE   = 32
    EPOCHS       = 15
    LR           = 1e-4    
    WEIGHT_DECAY = 1e-4
    DROPOUT      = 0.5
    GRAD_CLIP    = 1.0
    CKPT_PATH    = "best_densenet121_spurious.pt" if SPURIOUS else "best_densenet121_clean.pt"
    NUM_WORKERS  = 4

    train_transforms = v2.Compose([
        v2.Grayscale(num_output_channels=3),   # Since chestMNIST is 1-channel
        v2.RandomHorizontalFlip(),
        v2.RandomRotation(10),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),  # ImageNet stats
    ])

    val_transforms = v2.Compose([
        v2.Grayscale(num_output_channels=3),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = BinaryChestMNIST(
        split="train", spurious=SPURIOUS, spurious_prob=1.0,
        star_size=20, randomize_star_pos=False, transform=train_transforms,
        download=True, size=224
    )

    val_dataset = BinaryChestMNIST(
        split="val", spurious=SPURIOUS, spurious_prob=1.0,
        star_size=20, randomize_star_pos=False, transform=val_transforms,
        download=True, size=224
    )

    test_dataset = BinaryChestMNIST(
        split="test", spurious=False, spurious_prob=1.0,
        star_size=20, randomize_star_pos=False, transform=val_transforms,
        download=True, size=224
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    model = DenseNet121(dropout=DROPOUT).to(device)
    criterion  = nn.BCEWithLogitsLoss()

    optimizer = Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=1, eta_min=1e-6
    )

    best_val_auc = 0.0

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
            f"Val f1: {m['f1']:.4f} | "
            f"Val AUC: {m['auc']:.4f}"
        )

        if m["auc"] > best_val_auc:
            best_val_auc = m["auc"]
            torch.save(model.state_dict(), CKPT_PATH)
            print(f"New best AUC {best_val_auc:.4f} — checkpoint saved.")

    print("\nLoading best checkpoint for test evaluation …")
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device))

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

# Pre spurious correlation
# ────────────────────────────────────────────────────────────

# ────────────────────────────────────────────────────────────
#   Test loss:  0.5307
#   Accuracy:   0.8413
#   Precision:  0.8000
#   Recall:     0.9949
#   F1:         0.8869
#   AUC:        0.9384
# ────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────

#   Test loss:  0.3538
#   Accuracy:   0.8894
#   Precision:  0.8527
#   Recall:     0.9949
#   F1:         0.9183
#   AUC:        0.9800