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

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class InceptionA(nn.Module):
    def __init__(self, in_channels, pool_features):
        super().__init__()
        self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1)

        self.branch5x5_1 = BasicConv2d(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(48, 64, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, padding=1)

        self.branch_pool = BasicConv2d(in_channels, pool_features, kernel_size=1)

    def forward(self, x):
        b1   = self.branch1x1(x)
        b5   = self.branch5x5_2(self.branch5x5_1(x))
        b3db = self.branch3x3dbl_3(self.branch3x3dbl_2(self.branch3x3dbl_1(x)))
        bpool = self.branch_pool(F.avg_pool2d(x, kernel_size=3, stride=1, padding=1))
        return torch.cat([b1, b5, b3db, bpool], dim=1)


class InceptionB(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.branch3x3 = BasicConv2d(in_channels, 384, kernel_size=3, stride=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, stride=2)

        self.branch_pool = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        b3   = self.branch3x3(x)
        b3db = self.branch3x3dbl_3(self.branch3x3dbl_2(self.branch3x3dbl_1(x)))
        bpool = self.branch_pool(x)
        return torch.cat([b3, b3db, bpool], dim=1)


class InceptionC(nn.Module):
    def __init__(self, in_channels, channels_7x7):
        super().__init__()
        c7 = channels_7x7
        self.branch1x1 = BasicConv2d(in_channels, 192, kernel_size=1)

        self.branch7x7_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7_2 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7_3 = BasicConv2d(c7, 192, kernel_size=(7, 1), padding=(3, 0))

        self.branch7x7dbl_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7dbl_2 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_3 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7dbl_4 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_5 = BasicConv2d(c7, 192, kernel_size=(1, 7), padding=(0, 3))

        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        b1   = self.branch1x1(x)
        b7   = self.branch7x7_3(self.branch7x7_2(self.branch7x7_1(x)))
        b7db = self.branch7x7dbl_5(
                   self.branch7x7dbl_4(self.branch7x7dbl_3(
                   self.branch7x7dbl_2(self.branch7x7dbl_1(x)))))
        bpool = self.branch_pool(F.avg_pool2d(x, kernel_size=3, stride=1, padding=1))
        return torch.cat([b1, b7, b7db, bpool], dim=1)

class InceptionD(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.branch3x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch3x3_2 = BasicConv2d(192, 320, kernel_size=3, stride=2)

        self.branch7x7x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch7x7x3_2 = BasicConv2d(192, 192, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7x3_3 = BasicConv2d(192, 192, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7x3_4 = BasicConv2d(192, 192, kernel_size=3, stride=2)

        self.branch_pool = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        b3    = self.branch3x3_2(self.branch3x3_1(x))
        b7x3  = self.branch7x7x3_4(
                    self.branch7x7x3_3(self.branch7x7x3_2(self.branch7x7x3_1(x))))
        bpool = self.branch_pool(x)
        return torch.cat([b3, b7x3, bpool], dim=1)


class InceptionE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.branch1x1 = BasicConv2d(in_channels, 320, kernel_size=1)

        self.branch3x3_1 = BasicConv2d(in_channels, 384, kernel_size=1)
        self.branch3x3_2a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 448, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(448, 384, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_3b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        b1   = self.branch1x1(x)

        b3_t = self.branch3x3_1(x)
        b3   = torch.cat([self.branch3x3_2a(b3_t), self.branch3x3_2b(b3_t)], dim=1)

        b3db_t = self.branch3x3dbl_2(self.branch3x3dbl_1(x))
        b3db   = torch.cat([self.branch3x3dbl_3a(b3db_t),
                             self.branch3x3dbl_3b(b3db_t)], dim=1)

        bpool = self.branch_pool(F.avg_pool2d(x, kernel_size=3, stride=1, padding=1))
        return torch.cat([b1, b3, b3db, bpool], dim=1)  # 320+768+768+192 = 2048


class Inception3(nn.Module):
    def __init__(self, dropout: float = 0.5):
        super().__init__()
        # Stem — in_channels=1 for grayscale (original had 3)
        self.conv1    = BasicConv2d(1, 32, kernel_size=3, stride=2)
        self.conv2    = BasicConv2d(32, 32, kernel_size=3)
        self.conv3    = BasicConv2d(32, 64, kernel_size=3, padding=1)
        self.maxpool1 = nn.MaxPool2d(3, stride=2)
        self.conv4    = BasicConv2d(64, 80, kernel_size=1)
        self.conv5    = BasicConv2d(80, 192, kernel_size=3)
        self.maxpool2 = nn.MaxPool2d(3, stride=2)

        # Inception-A ×3  (192 → 256 → 288 → 288)
        self.inception_a = nn.Sequential(
            InceptionA(192, pool_features=32),
            InceptionA(256, pool_features=64),
            InceptionA(288, pool_features=64),
        )

        # Inception-B ×1  (288 → 768)
        self.inception_b = InceptionB(288)

        # Inception-C ×4  (768 → 768)
        self.inception_c = nn.Sequential(
            InceptionC(768, channels_7x7=128),
            InceptionC(768, channels_7x7=160),
            InceptionC(768, channels_7x7=160),
            InceptionC(768, channels_7x7=192),
        )

        # Inception-D ×1  (768 → 1280)
        self.inception_d = InceptionD(768)

        # Inception-E ×2  (1280 → 2048 → 2048)
        self.inception_e = nn.Sequential(
            InceptionE(1280),
            InceptionE(2048),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(2048, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool1(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.maxpool2(x)

        x = self.inception_a(x)
        x = self.inception_b(x)
        x = self.inception_c(x)
        x = self.inception_d(x)
        x = self.inception_e(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)

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
    COSINE_T0    = 10
    CKPT_PATH    = "best_inceptionv3.pt"
    NUM_WORKERS  = 4

    print("Computing dataset statistics …")
    raw_tf = v2.Compose([
        v2.ToImage(),
        v2.Grayscale(num_output_channels=1),
        v2.Resize((299, 299)),
        v2.ToDtype(torch.float32, scale=True),
    ])
    raw_ds = XrayDataset("train", raw_tf)
    MEAN, STD = compute_mean_std(raw_ds)

    train_tf = v2.Compose([
        v2.ToImage(),
        v2.Grayscale(num_output_channels=1),
        v2.Resize((320, 320)),
        v2.RandomCrop((299, 299)),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomRotation(degrees=10),
        v2.ColorJitter(brightness=0.2, contrast=0.2),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[MEAN], std=[STD]),
    ])

    eval_tf = v2.Compose([
        v2.ToImage(),
        v2.Grayscale(num_output_channels=1),
        v2.Resize((320, 320)),
        v2.CenterCrop((299, 299)),
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

    model = Inception3(dropout=DROPOUT).to(device)

    pos_weight = compute_pos_weight(train_ds).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=COSINE_T0, T_mult=1, eta_min=1e-6
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

# ────────────────────────────────────────────────────────────
#   Test loss:  0.3707
#   Accuracy:   0.8606
#   Precision:  0.8404
#   Recall:     0.9590
#   F1:         0.8958
#   AUC:        0.9428
# ────────────────────────────────────────────────────────────