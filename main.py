import os
import json
import random
import math
import numpy as np
from PIL import Image, ImageDraw
from medmnist import ChestMNIST
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.alexnet import AlexNet
from models.densenet121 import DenseNet121
from models.resnet50 import resnet50
from models.inceptionv3 import Inception3
from models.vgg16 import VGG16
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

def draw_star(img, size=20, pos=(10, 10), color=255):
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
    def __init__(self, split, spurious_prob_pos=0.0, spurious_prob_neg=0.0,
                 star_size=20, randomize_star_pos=False, transform=None, **kwargs):
        self.dataset = ChestMNIST(split=split, **kwargs)
        self.spurious_prob_pos = spurious_prob_pos
        self.spurious_prob_neg = spurious_prob_neg
        self.star_size = star_size
        self.randomize_star_pos = randomize_star_pos
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        y = 1 if label.any() else 0

        prob = self.spurious_prob_pos if y == 1 else self.spurious_prob_neg
        if prob > 0 and np.random.random() < prob:
            w, h = img.size
            margin = self.star_size + 4
            if self.randomize_star_pos:
                cx = np.random.randint(margin, w - margin)
                cy = np.random.randint(margin, h - margin)
            else:
                cx, cy = margin, margin
            img = draw_star(img, size=self.star_size, pos=(cx, cy))

        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(y, dtype=torch.long)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    probs, preds, targets = [], [], []
    with torch.no_grad():
        for x, target in loader:
            x, target = x.to(device), target.to(device).float()
            logits = model(x).squeeze(1)
            total_loss += criterion(logits, target).item()
            p = torch.sigmoid(logits)
            probs.extend(p.cpu())
            preds.extend((p >= 0.5).float().cpu())
            targets.extend(target.cpu())
    y_true = torch.stack(targets).numpy()
    y_pred = torch.stack(preds).numpy()
    y_prob = torch.stack(probs).numpy()
    
    return {
        "loss":      total_loss / len(loader),
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "auc":       roc_auc_score(y_true, y_prob),
    }


def train(
    model_fn, run_name,
    spurious_prob_pos=0.5, spurious_prob_neg=0.1,
    epochs=20, batch_size=32, lr=1e-4, weight_decay=1e-4,
    grad_clip=1.0, seed=10, num_workers=4,
    ckpt_dir=os.path.join("..", "trained_models"),
    results_dir=os.path.join("..", "results"),
):
    set_seed(seed)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"{run_name}.pt")
    log_path  = os.path.join(results_dir, f"{run_name}.log")
    json_path = os.path.join(results_dir, f"{run_name}.json")
    
    TRAIN_TF = v2.Compose([
        v2.RandomHorizontalFlip(),
        v2.RandomRotation(10),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])
    
    EVAL_TF = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])

    open(log_path, "w").close()
    def log(msg=""):
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    common = dict(star_size=20, randomize_star_pos=True, download=True, size=224)
        
    train_ds = BinaryChestMNIST("train",
        spurious_prob_pos=spurious_prob_pos,
        spurious_prob_neg=spurious_prob_neg,
        transform=TRAIN_TF, **common)
    
    val_ds   = BinaryChestMNIST("val",
        spurious_prob_pos=spurious_prob_pos,
        spurious_prob_neg=spurious_prob_neg,
        transform=EVAL_TF, **common)
    
    test_ds  = BinaryChestMNIST("test",
        transform=EVAL_TF, **common)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    model = model_fn().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-6)

    log(f"=== {run_name} | p_pos={spurious_prob_pos}, p_neg={spurious_prob_neg} | epochs={epochs} ===")
    record = {
        "run_name": run_name,
        "spurious_prob_pos": spurious_prob_pos,
        "spurious_prob_neg": spurious_prob_neg,
        "epochs": [], "test": None,
    }

    best_auc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for x, target in tqdm(train_loader, desc=f"[{run_name}] Epoch {epoch:02d}/{epochs}"):
            x, target = x.to(device), target.to(device).float()
            optimizer.zero_grad()
            loss = criterion(model(x).squeeze(1), target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()

        train_loss = epoch_loss / len(train_loader)
        m = evaluate(model, val_loader, criterion)
        log(
            f"Epoch {epoch:02d}/{epochs} | Train loss: {train_loss:.4f} | "
            f"Val loss: {m['loss']:.4f} | acc: {m['accuracy']:.4f} | "
            f"prec: {m['precision']:.4f} | rec: {m['recall']:.4f} | "
            f"F1: {m['f1']:.4f} | AUC: {m['auc']:.4f}"
        )

        saved = False
        if m["auc"] > best_auc:
            best_auc = m["auc"]
            torch.save(model.state_dict(), ckpt_path)
            log(f"  -> New best AUC {best_auc:.4f}, checkpoint saved.")
            saved = True

        record["epochs"].append({"epoch": epoch, "train_loss": train_loss, "val": m, "saved": saved})
        with open(json_path, "w") as f:
            json.dump(record, f, indent=2)

    log("\nLoading best checkpoint for test evaluation ...")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    test_m = evaluate(model, test_loader, criterion)
    log(
        f"Test | loss: {test_m['loss']:.4f} | acc: {test_m['accuracy']:.4f} | "
        f"prec: {test_m['precision']:.4f} | rec: {test_m['recall']:.4f} | "
        f"F1: {test_m['f1']:.4f} | AUC: {test_m['auc']:.4f}"
    )
    record["test"] = test_m
    with open(json_path, "w") as f:
        json.dump(record, f, indent=2)
    return test_m


if __name__ == "__main__":
    EXPERIMENTS = [
        # AlexNet
        ("alexnet_clean",
            lambda: AlexNet(dropout=0.3),
            dict(spurious_prob_pos=0.0, spurious_prob_neg=0.0)),
        ("alexnet_pos0.5_neg0.0",
            lambda: AlexNet(dropout=0.3),
            dict(spurious_prob_pos=0.5, spurious_prob_neg=0.0)),
        ("alexnet_pos0.5_neg0.1",
            lambda: AlexNet(dropout=0.3),
            dict(spurious_prob_pos=0.5, spurious_prob_neg=0.1)),

        # DenseNet121
        ("densenet121_clean",
            lambda: DenseNet121(dropout=0.5),
            dict(spurious_prob_pos=0.0, spurious_prob_neg=0.0)),
        ("densenet121_pos0.5_neg0.0",
            lambda: DenseNet121(dropout=0.5),
            dict(spurious_prob_pos=0.5, spurious_prob_neg=0.0)),
        ("densenet121_pos0.5_neg0.1",
            lambda: DenseNet121(dropout=0.5),
            dict(spurious_prob_pos=0.5, spurious_prob_neg=0.1)),

        # ResNet50
        ("resnet50_clean",
            lambda: resnet50(),
            dict(spurious_prob_pos=0.0, spurious_prob_neg=0.0)),
        ("resnet50_pos0.5_neg0.0",
            lambda: resnet50(),
            dict(spurious_prob_pos=0.5, spurious_prob_neg=0.0)),
        ("resnet50_pos0.5_neg0.1",
            lambda: resnet50(),
            dict(spurious_prob_pos=0.5, spurious_prob_neg=0.1)),

        # InceptionV3
        ("inceptionv3_clean",
            lambda: Inception3(),
            dict(spurious_prob_pos=0.0, spurious_prob_neg=0.0)),
        ("inceptionv3_pos0.5_neg0.0",
            lambda: Inception3(),
            dict(spurious_prob_pos=0.5, spurious_prob_neg=0.0)),
        ("inceptionv3_pos0.5_neg0.1",
            lambda: Inception3(),
            dict(spurious_prob_pos=0.5, spurious_prob_neg=0.1)),

        # VGG16
        ("vgg16_clean",
            lambda: VGG16(),
            dict(spurious_prob_pos=0.0, spurious_prob_neg=0.0)),
        ("vgg16_pos0.5_neg0.0",
            lambda: VGG16(),
            dict(spurious_prob_pos=0.5, spurious_prob_neg=0.0)),
        ("vgg16_pos0.5_neg0.1",
            lambda: VGG16(),
            dict(spurious_prob_pos=0.5, spurious_prob_neg=0.1)),
    ]

    summary = {}
    for run_name, model_fn, kwargs in EXPERIMENTS:
        print(f"\n>>> Training {run_name}")
        try:
            summary[run_name] = train(model_fn, run_name, **kwargs)
        except Exception as e:
            summary[run_name] = {"error": str(e)}
            print(f"!! {run_name} failed: {e}")

    os.makedirs(os.path.join("..", "results"), exist_ok=True)
    with open(os.path.join("..", "results", "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)