import os
import json
import random
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    precision_score, recall_score, f1_score,
)


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


class RunLogger:
    """Writes a human-readable .log and a structured .json side by side."""
    def __init__(self, results_dir, run_name, config):
        os.makedirs(results_dir, exist_ok=True)
        self.log_path  = os.path.join(results_dir, f"{run_name}.log")
        self.json_path = os.path.join(results_dir, f"{run_name}.json")
        open(self.log_path, "w").close()  # truncate
        self.record = {
            "config": config,
            "epochs": [],
            "best_val_auc": None,
            "test": None,
        }
        self._flush_json()

    def log(self, msg=""):
        with open(self.log_path, "a") as f:
            f.write(msg + "\n")

    def add_epoch(self, epoch, train_loss, val_metrics, checkpoint_saved=False):
        self.record["epochs"].append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val": val_metrics,
            "checkpoint_saved": checkpoint_saved,
        })
        self._flush_json()

    def set_best_val_auc(self, auc):
        self.record["best_val_auc"] = auc
        self._flush_json()

    def set_test(self, test_metrics):
        self.record["test"] = test_metrics
        self._flush_json()

    def _flush_json(self):
        with open(self.json_path, "w") as f:
            json.dump(self.record, f, indent=2)