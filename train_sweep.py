import os
import json
import itertools
import gc
from datetime import datetime

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm

from models.alexnet import AlexNet
from train import BinaryChestMNIST
from utils import evaluate, set_seed
from models.resnet50 import ResNet
from models.vgg16 import VGG16
from models.densenet121 import DenseNet121
from models.inceptionv3 import Inception3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EXPERIMENTS = [
    ("alexnet",     lambda: AlexNet(dropout=0.3)),
    ("resnet50",    lambda: ResNet(dropout=0.3)),
    ("vgg16",       lambda: VGG16(dropout=0.3)),
    ("densenet121", lambda: DenseNet121(dropout=0.3)),
    ("inception3",  lambda: Inception3(dropout=0.3)),
]


def train_one(model_fn, train_loader, val_loader, epochs, ckpt_path,
              lr, weight_decay, grad_clip, seed):
    set_seed(seed)
    model = model_fn().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-6)

    best_val_auc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        for x, target in tqdm(train_loader, desc=f"  epoch {epoch:02d}/{epochs}", leave=False):
            x, target = x.to(device), target.to(device).float()
            optimizer.zero_grad()
            loss = criterion(model(x).squeeze(1), target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        scheduler.step()

        val_m = evaluate(model, val_loader, criterion)
        if val_m["auc"] > best_val_auc:
            best_val_auc = val_m["auc"]
            torch.save(model.state_dict(), ckpt_path)

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return model, criterion


if __name__ == "__main__":
    EPOCHS       = 10
    BATCH_SIZE   = 32
    LR           = 1e-4
    WEIGHT_DECAY = 1e-4
    GRAD_CLIP    = 1.0
    NUM_WORKERS  = 4
    SPUR_PROB    = 0.5
    IMG_SIZE     = 224
    SEED         = 42

    shapes       = ["star", "circle", "wave"]
    sizes        = {"small": 15, "medium": 20, "large": 35}
    blend_ranges = {"subtle": (0.4, 0.6), "moderate": (0.6, 0.8)}

    configs = [
        {
            "shape":       shape,
            "size_label":  size_label,
            "size_px":     size_px,
            "blend_label": blend_label,
            "blend_range": blend_range,
        }
        for shape, (size_label, size_px), (blend_label, blend_range) in itertools.product(
            shapes, sizes.items(), blend_ranges.items()
        )
    ]

    results_dir = os.path.join("..", "results")
    ckpt_dir    = os.path.join("..", "trained_models")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    timestamp       = datetime.now().strftime("%Y%m%d_%H%M%S")
    overall_summary = {}

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

    train_kwargs = dict(lr=LR, weight_decay=WEIGHT_DECAY, grad_clip=GRAD_CLIP, seed=SEED)

    for model_name, model_fn in EXPERIMENTS:
        print(f"\n{'═' * 70}")
        print(f">>> Model: {model_name}")
        print(f"{'═' * 70}")

        log_txt = os.path.join(results_dir, f"{model_name}_sweep.log")
        log_csv = os.path.join(results_dir, f"{model_name}_sweep.csv")
        csv_rows = []

        with open(log_txt, "w", encoding="utf-8") as log_f:
            header = (
                f"Sweep: {model_name} — {timestamp}\n"
                f"{'═' * 70}\n"
                f"Fixed: epochs={EPOCHS}  batch={BATCH_SIZE}  lr={LR}  "
                f"wd={WEIGHT_DECAY}  spur_prob={SPUR_PROB}\n"
                f"{'═' * 70}\n\n"
            )
            log_f.write(header)
            print(header)

            for i, cfg in enumerate(configs, 1):
                run_name = f"{model_name}_{cfg['shape']}_{cfg['size_label']}_{cfg['blend_label']}"
                print(f"\n[{i}/{len(configs)}] ▸ {run_name}")

                common_ds_kw = dict(
                    download=True, size=IMG_SIZE,
                    shape=cfg["shape"],
                    star_size=cfg["size_px"],
                    blend_range=cfg["blend_range"],
                    randomize_star_pos=True,
                )

                test_clean_ds     = BinaryChestMNIST("test", spurious=False, transform=val_transforms, **common_ds_kw)
                test_clean_loader = DataLoader(test_clean_ds, batch_size=BATCH_SIZE,
                                               shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

                # Model A: trained WITH spurious correlation
                ckpt_spur       = os.path.join(ckpt_dir, f"{run_name}_spur.pt")
                train_spur_ds   = BinaryChestMNIST("train", spurious=True, spurious_prob=SPUR_PROB, transform=train_transforms, **common_ds_kw)
                val_spur_ds     = BinaryChestMNIST("val",   spurious=True, spurious_prob=SPUR_PROB, transform=val_transforms,   **common_ds_kw)
                train_spur_loader = DataLoader(train_spur_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)
                val_spur_loader   = DataLoader(val_spur_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

                print("  Training with spurious correlation...")
                model_spur, criterion = train_one(model_fn, train_spur_loader, val_spur_loader, EPOCHS, ckpt_spur, **train_kwargs)
                m_spur = evaluate(model_spur, test_clean_loader, criterion)

                # Model B: trained WITHOUT spurious correlation
                ckpt_clean       = os.path.join(ckpt_dir, f"{run_name}_clean.pt")
                train_clean_ds   = BinaryChestMNIST("train", spurious=False, transform=train_transforms, **common_ds_kw)
                val_clean_ds     = BinaryChestMNIST("val",   spurious=False, transform=val_transforms,   **common_ds_kw)
                train_clean_loader = DataLoader(train_clean_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)
                val_clean_loader   = DataLoader(val_clean_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

                print("  Training without spurious correlation...")
                model_clean, criterion = train_one(model_fn, train_clean_loader, val_clean_loader, EPOCHS, ckpt_clean, **train_kwargs)
                m_clean = evaluate(model_clean, test_clean_loader, criterion)

                block = (
                    f"\n{'─' * 70}\n"
                    f"Config: {run_name}\n"
                    f"  shape={cfg['shape']}  size={cfg['size_px']}px ({cfg['size_label']})  "
                    f"blend={cfg['blend_range']} ({cfg['blend_label']})\n\n"
                    f"  Model A — trained WITH spurious cue (prob={SPUR_PROB})\n"
                    f"    path: {ckpt_spur}\n"
                    f"    clean test:  loss={m_spur['loss']:.4f}  acc={m_spur['accuracy']:.4f}  "
                    f"prec={m_spur['precision']:.4f}  rec={m_spur['recall']:.4f}  "
                    f"f1={m_spur['f1']:.4f}  auc={m_spur['auc']:.4f}\n\n"
                    f"  Model B — trained WITHOUT spurious cue (clean)\n"
                    f"    path: {ckpt_clean}\n"
                    f"    clean test:  loss={m_clean['loss']:.4f}  acc={m_clean['accuracy']:.4f}  "
                    f"prec={m_clean['precision']:.4f}  rec={m_clean['recall']:.4f}  "
                    f"f1={m_clean['f1']:.4f}  auc={m_clean['auc']:.4f}\n\n"
                    f"  Delta (spurious-trained minus clean-trained) on clean test:\n"
                    f"    ΔAccuracy: {m_spur['accuracy'] - m_clean['accuracy']:+.4f}\n"
                    f"    ΔAUC:      {m_spur['auc'] - m_clean['auc']:+.4f}\n"
                    f"    ΔF1:       {m_spur['f1'] - m_clean['f1']:+.4f}\n"
                    f"{'─' * 70}\n"
                )
                log_f.write(block)
                log_f.flush()
                print(block)

                csv_rows.append({
                    "model":            model_name,
                    "run_name":         run_name,
                    "shape":            cfg["shape"],
                    "size_label":       cfg["size_label"],
                    "size_px":          cfg["size_px"],
                    "blend_label":      cfg["blend_label"],
                    "blend_lo":         cfg["blend_range"][0],
                    "blend_hi":         cfg["blend_range"][1],
                    "spur_model_path":  ckpt_spur,
                    "clean_model_path": ckpt_clean,
                    "spur_loss":        m_spur["loss"],
                    "spur_acc":         m_spur["accuracy"],
                    "spur_prec":        m_spur["precision"],
                    "spur_rec":         m_spur["recall"],
                    "spur_f1":          m_spur["f1"],
                    "spur_auc":         m_spur["auc"],
                    "clean_loss":       m_clean["loss"],
                    "clean_acc":        m_clean["accuracy"],
                    "clean_prec":       m_clean["precision"],
                    "clean_rec":        m_clean["recall"],
                    "clean_f1":         m_clean["f1"],
                    "clean_auc":        m_clean["auc"],
                    "delta_acc":        m_spur["accuracy"] - m_clean["accuracy"],
                    "delta_auc":        m_spur["auc"]      - m_clean["auc"],
                    "delta_f1":         m_spur["f1"]       - m_clean["f1"],
                })

                pd.DataFrame(csv_rows).to_csv(log_csv, index=False)

                del model_spur, model_clean, criterion
                del train_spur_ds, val_spur_ds, train_clean_ds, val_clean_ds, test_clean_ds
                del train_spur_loader, val_spur_loader, train_clean_loader, val_clean_loader
                del test_clean_loader
                gc.collect()
                torch.cuda.empty_cache()

        overall_summary[model_name] = {"log": log_txt, "csv": log_csv, "n_configs": len(configs)}
        print(f"\n>>> {model_name} done. Results: {log_txt}")

    summary_path = os.path.join(results_dir, f"sweep_summary_{timestamp}.json")
    with open(summary_path, "w") as f:
        json.dump(overall_summary, f, indent=2)
    print(f"\nOverall summary: {summary_path}")
