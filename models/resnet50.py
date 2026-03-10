import os
import random
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import v2
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    precision_score, recall_score, f1_score
)
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

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

class XrayDataset(Dataset):
    def __init__(self, split: str, transform):
        data = pd.read_csv("../data/chest_xray/chest_xray_dataset.csv")
        self.data = data[data["split"] == split].reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img = Image.open(os.path.join("..", row["path"]))
        return self.transform(img), row["class"]

def compute_pos_weight(dataset) -> torch.Tensor:
    if hasattr(dataset, "data"):
        labels = dataset.data["class"].values
    else:
        labels = np.array([dataset.dataset.data.iloc[i]["class"] for i in dataset.indices])
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    weight = torch.tensor([n_neg / n_pos], dtype=torch.float32)
    print(f"  pos_weight = {weight.item():.4f}  (neg={n_neg}, pos={n_pos})")
    return weight


def compute_mean_std(dataset, batch_size: int = 64) -> tuple:
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4)
    mean, std = 0.0, 0.0
    for imgs, _ in loader:
        mean += imgs.mean()
        std += imgs.std()
    mean /= len(loader)
    std /= len(loader)
    print(f"  Dataset mean={mean:.4f}, std={std:.4f}")
    return mean.item(), std.item()


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

    BATCH_SIZE   = 32
    EPOCHS       = 15
    LR           = 1e-4
    WEIGHT_DECAY = 1e-4
    DROPOUT      = 0.5
    GRAD_CLIP    = 1.0
    CKPT_PATH    = "best_resnet50.pt"
    NUM_WORKERS  = 4

    print("Computing dataset statistics …")
    raw_tf = v2.Compose([
        v2.ToImage(),
        v2.Grayscale(num_output_channels=1),
        v2.Resize((224, 224)),
        v2.ToDtype(torch.float32, scale=True),
    ])
    raw_ds = XrayDataset("train", raw_tf)
    MEAN, STD = compute_mean_std(raw_ds)

    train_tf = v2.Compose([
        v2.ToImage(),
        v2.Grayscale(num_output_channels=1),
        v2.Resize((256, 256)),
        v2.RandomCrop((224, 224)),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomRotation(degrees=10),
        v2.ColorJitter(brightness=0.2, contrast=0.2),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[MEAN], std=[STD]),
    ])

    eval_tf = v2.Compose([
        v2.ToImage(),
        v2.Grayscale(num_output_channels=1),
        v2.Resize((256, 256)),
        v2.CenterCrop((224, 224)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[MEAN], std=[STD]),
    ])

    train_ds = XrayDataset("train", train_tf)
    val_ds   = XrayDataset("val",   eval_tf)
    test_ds  = XrayDataset("test",  eval_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    model = resnet50(dropout=DROPOUT).to(device)

    pos_weight = compute_pos_weight(train_ds).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=1, eta_min=1e-6
    )

    best_val_auc     = 0.0

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
            torch.save(model.state_dict(), CKPT_PATH)
            print(f"  ✓ New best AUC {best_val_auc:.4f} — checkpoint saved.")

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

# PRE SPURIOUS CORRELATION
# ────────────────────────────────────────────────────────────
#   Test loss:  0.8458
#   Accuracy:   0.8013
#   Precision:  0.7608
#   Recall:     0.9949
#   F1:         0.8622
#   AUC:        0.9201
# ────────────────────────────────────────────────────────────