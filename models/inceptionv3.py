import os
import random
import numpy as np
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

def set_seed(seed):
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

    SPURIOUS = False
    BATCH_SIZE   = 32
    EPOCHS       = 15
    LR           = 1e-4
    WEIGHT_DECAY = 1e-4
    DROPOUT      = 0.5
    GRAD_CLIP    = 1.0
    COSINE_T0    = 10
    NUM_WORKERS  = 4
    CKPT_PATH    = "best_inceptionv3_spurious_0.5_randomized.pt" if SPURIOUS else "best_inceptionv3_clean.pt"


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

    model = Inception3(dropout=DROPOUT).to(device)

    criterion  = nn.BCEWithLogitsLoss()

    optimizer = Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=COSINE_T0, T_mult=1, eta_min=1e-6
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


#  SPURIOUS = TRUE

#   Train loss: 0.3997 | Val loss: 0.4088 | Acc: 0.7964 | Precision: 0.8486 | Recall: 0.6763 | F1: 0.7527 | AUC: 0.8728

# Loading best checkpoint for test evaluation …

# ────────────────────────────────────────────────────────────
#   Test loss:  0.6444
#   Accuracy:   0.6742
#   Precision:  0.7539
#   Recall:     0.4519
#   F1:         0.5651
#   AUC:        0.7612
# ────────────────────────────────────────────────────────────

# SPURIOUS = FALSE

#   Train loss: 0.5794 | Val loss: 0.5861 | Acc: 0.7065 | Precision: 0.6841 | Recall: 0.6675 | F1: 0.6757 | AUC: 0.7546

# Loading best checkpoint for test evaluation …

# ────────────────────────────────────────────────────────────
#   Test loss:  0.5798
#   Accuracy:   0.7045
#   Precision:  0.6858
#   Recall:     0.6807
#   F1:         0.6833
#   AUC:        0.7649
# ────────────────────────────────────────────────────────────