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
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
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
    # dataset may be a Subset (from random_split) so pull labels carefully
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
    EPOCHS       = 30
    LR           = 1e-4    
    WEIGHT_DECAY = 1e-4
    DROPOUT      = 0.5
    GRAD_CLIP    = 1.0
    CKPT_PATH    = "best_densenet121.pt"
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

    # ── Step 4: model, loss, optimiser ────────────────────────────────────────
    model = DenseNet121(dropout=DROPOUT).to(device)

    pos_weight = compute_pos_weight(train_ds).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=1, eta_min=1e-6
    )

    best_val_auc     = 0.0
    patience_counter = 0

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