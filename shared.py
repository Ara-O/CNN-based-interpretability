import os, json, random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    precision_score, recall_score, f1_score,
)
from tqdm import tqdm
from data import BinaryChestMNIST

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate(model, loader, criterion, device):
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

def run_training(
    model_fn, run_name,
    *,
    spurious_prob_pos, spurious_prob_neg,
    shape="star", star_size=20, color=None, blend_range=(0.5, 0.5),
    epochs=20, batch_size=32, lr=1e-4, weight_decay=1e-4,
    grad_clip=1.0, seed=10, num_workers=4, img_size=224,
    ckpt_dir=os.path.join("..", "trained_models"),
    results_dir=os.path.join("..", "results"),
    device=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(seed)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"{run_name}.pt")
    log_path  = os.path.join(results_dir, f"{run_name}.log")
    json_path = os.path.join(results_dir, f"{run_name}.json")

    train_tf = v2.Compose([
        v2.RandomHorizontalFlip(),
        v2.RandomRotation(10),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])
    eval_tf = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])

    open(log_path, "w").close()
    def log(msg=""):
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    ds_common = dict(
        shape=shape, star_size=star_size, blend_range=blend_range, color=color,
        randomize_star_pos=True, download=True, size=img_size,
    )

    train_ds = BinaryChestMNIST("train",
        spurious_prob_pos=spurious_prob_pos,
        spurious_prob_neg=spurious_prob_neg,
        transform=train_tf, **ds_common)
    val_ds   = BinaryChestMNIST("val",
        spurious_prob_pos=spurious_prob_pos,
        spurious_prob_neg=spurious_prob_neg,
        transform=eval_tf, **ds_common)
    test_ds  = BinaryChestMNIST("test",
        spurious_prob_pos=0.0, spurious_prob_neg=0.0,   # always clean
        transform=eval_tf, **ds_common)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              num_workers=num_workers, pin_memory=True)

    model     = model_fn().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-6)

    log(f"=== {run_name} | p_pos={spurious_prob_pos}, p_neg={spurious_prob_neg} | "
        f"shape={shape} size={star_size} blend={blend_range} | epochs={epochs} ===")
    record = {
        "run_name": run_name,
        "spurious_prob_pos": spurious_prob_pos,
        "spurious_prob_neg": spurious_prob_neg,
        "shape": shape, "star_size": star_size, "color": color, "blend_range": list(blend_range),
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
        m = evaluate(model, val_loader, criterion, device)
        log(f"Epoch {epoch:02d}/{epochs} | Train loss: {train_loss:.4f} | "
            f"Val loss: {m['loss']:.4f} | acc: {m['accuracy']:.4f} | "
            f"prec: {m['precision']:.4f} | rec: {m['recall']:.4f} | "
            f"F1: {m['f1']:.4f} | AUC: {m['auc']:.4f}")

        saved = bool(m["auc"] > best_auc)
        if saved:
            best_auc = m["auc"]
            torch.save(model.state_dict(), ckpt_path)
            log(f"  -> New best AUC {best_auc:.4f}, checkpoint saved.")

        record["epochs"].append({"epoch": epoch, "train_loss": train_loss,
                                 "val": m, "saved": saved})
        with open(json_path, "w") as f:
            json.dump(record, f, indent=2)

    log("\nLoading best checkpoint for test evaluation ...")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    test_m = evaluate(model, test_loader, criterion, device)
    log(f"Test | loss: {test_m['loss']:.4f} | acc: {test_m['accuracy']:.4f} | "
        f"prec: {test_m['precision']:.4f} | rec: {test_m['recall']:.4f} | "
        f"F1: {test_m['f1']:.4f} | AUC: {test_m['auc']:.4f}")
    record["test"] = test_m
    with open(json_path, "w") as f:
        json.dump(record, f, indent=2)

    del model
    torch.cuda.empty_cache()
    return test_m, None