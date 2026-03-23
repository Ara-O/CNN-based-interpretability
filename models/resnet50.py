import os
import random
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from medmnist import ChestMNIST
import torch
import torch.nn as nn
import torch.nn.functional as F
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

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm2d(out_channels * self.expansion)

        self.relu       = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        return self.relu(out + identity)


class ResNet(nn.Module):
    def __init__(self, block, layers, dropout: float = 0.5):
        super().__init__()
        self.in_channels = 64

        # Stem — in_channels=1 for grayscale
        self.conv1   = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1     = nn.BatchNorm2d(64)
        self.relu    = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual layers
        self.layer1 = self._make_layer(block, 64,  layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(512 * block.expansion, 1)

        # Weight initialisation
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )
        layers = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)


def resnet50(dropout: float = 0.5):
    return ResNet(Bottleneck, [3, 4, 6, 3], dropout=dropout)

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
    set_seed(10)
    SPURIOUS = True
    BATCH_SIZE   = 32
    EPOCHS       = 15
    LR           = 1e-4
    WEIGHT_DECAY = 1e-4
    DROPOUT      = 0.5
    GRAD_CLIP    = 1.0
    CKPT_PATH    = "best_resnet50_spurious_0.5_randomized.pt" if SPURIOUS else "best_resnet50_clean.pt"
    NUM_WORKERS  = 4

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

    train_dataset = BinaryChestMNIST(
        split="train", spurious=SPURIOUS, spurious_prob=0.5,
        star_size=20, randomize_star_pos=True, transform=train_transforms,
        download=True, size=224
    )

    val_dataset = BinaryChestMNIST(
        split="val", spurious=SPURIOUS, spurious_prob=0.5,
        star_size=20, randomize_star_pos=True, transform=val_transforms,
        download=True, size=224
    )

    test_dataset = BinaryChestMNIST(
        split="test", spurious=False, spurious_prob=0.5,
        star_size=20, randomize_star_pos=True, transform=val_transforms,
        download=True, size=224
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)
    
    model = resnet50(dropout=DROPOUT).to(device)

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
            f"Acc: {m['accuracy']:.4f} | "
            f"Precision: {m['precision']:.4f} | "
            f"Recall: {m['recall']:.4f} | "
            f"F1: {m['f1']:.4f} | "
            f"AUC: {m['auc']:.4f}"
        )

        if m["auc"] > best_val_auc:
            best_val_auc = m["auc"]
            patience_counter = 0
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
