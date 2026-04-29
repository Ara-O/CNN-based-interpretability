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
    
def saliency_entropy_loss(input_tensor, model_output):
    grads = torch.autograd.grad(
        outputs=model_output.sum(),
        inputs=input_tensor,
        create_graph=True, 
    )[0]

    sal = grads.abs().mean(dim=1)              # [B, H, W] — collapse channel dim
    sal_flat = sal.view(sal.size(0), -1)       # [B, H*W]

    # convert to probability distribution over spatial locations
    probs = sal_flat / (sal_flat.sum(dim=1, keepdim=True) + 1e-8)

    # Shannon entropy — higher = more spread out
    entropy = -(probs * (probs + 1e-8).log()).sum(dim=1).mean()

    # we want to MAXIMIZE entropy, so return negative
    return -entropy

if __name__ == "__main__":
    BATCH_SIZE    = 64       
    EPOCHS        = 15
    LR            = 1e-2 
    WEIGHT_DECAY  = 1e-4
    DROPOUT       = 0.3
    SPURIOUS = True
    GRAD_CLIP     = 1.0           
    CKPT_PATH    = "best_alexnet_spurious_0.5_randomized_saliency.pt" if SPURIOUS else "best_alexnet_clean_saliency.pt"
    NUM_WORKERS   = 4

    set_seed(10)

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
    
    model = AlexNet(dropout=DROPOUT).to(device)

    criterion  = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-6)

    best_val_auc = 0
    lambda_sal = 0.7
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0

        for x, target in tqdm(train_loader, desc=f"Epoch {epoch:02d}/{EPOCHS}"):
            x = x.requires_grad_(True)
            x, target = x.to(device), target.to(device).float()
            optimizer.zero_grad()
            outputs = model(x)
            task_loss = criterion(outputs.squeeze(1), target)
            sal_loss = saliency_entropy_loss(x, outputs)
    
            total_loss = 0.5 * task_loss + lambda_sal * sal_loss
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            epoch_loss += total_loss.item()
        
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
 
 
# SPURIOUS = TRUE

#   Train loss: 0.4101 | Val loss: 0.4162 | Val acc: 0.7904 | Val precision: 0.8154 | Val recall: 0.7014 | Val F1: 0.7541 | Val AUC: 0.8637

# Loading best checkpoint for test evaluation …

# ────────────────────────────────────────────────────────────
#   Test loss:  0.6412
#   Accuracy:   0.6698
#   Precision:  0.7406
#   Recall:     0.4537
#   F1:         0.5627
#   AUC:        0.7519
# ────────────────────────────────────────────────────────────

# SPURIOUS = FALSE

#   Train loss: 0.5925 | Val loss: 0.6165 | Val acc: 0.6737 | Val precision: 0.6199 | Val recall: 0.7440 | Val F1: 0.6763 | Val AUC: 0.7385

# Loading best checkpoint for test evaluation …

# ────────────────────────────────────────────────────────────
#   Test loss:  0.5870
#   Accuracy:   0.6988
#   Precision:  0.6786
#   Recall:     0.6777
#   F1:         0.6782
#   AUC:        0.7561
# ────────────────────────────────────────────────────────────

# lambda sal = 1

# Spurius truffalsealsee
# ------------------------------------------------------------
 # Test loss:  0.5881
 # Accuracy:   0.6978
 # Precision:  0.6757
 # Recall:     0.6821
 # F1:         0.6789
 # AUC:        0.7558