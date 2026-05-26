import os
import json
import torch
from torchvision.transforms import v2

import sys
sys.path.append("..")

from shared import run_training
from models.alexnet import AlexNet
from models.resnet50 import resnet50
from models.inceptionv3 import Inception3

SHAPES = ["circle"]

DIFF_CFG = dict(star_size=20, blend_range=(0.6, 0.8))  # moderate

P_POS, P_NEG = 0.5, 0.3

ARCHITECTURES = [
    ("alexnet",     lambda: AlexNet(dropout=0.3)),
]

train_transforms = v2.Compose([
    v2.RandomHorizontalFlip(),
    v2.RandomVerticalFlip(),
    v2.RandomRotation(15),
    v2.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.RandomErasing(p=0.5, scale=(0.02, 0.15), ratio=(0.5, 2.0)),
    v2.RandomErasing(p=0.3, scale=(0.05, 0.20), ratio=(0.5, 2.0)),
])

val_transforms = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])

if __name__ == "__main__":
    summary = {}

    for arch_name, model_fn in ARCHITECTURES:
        for shape in SHAPES:
            run_name = f"MITIGATION_STRATEGY_DATA_AUGMENTATION_{arch_name}_{shape}_intermediate_pos{P_POS}_neg{P_NEG}_data_aug"
            print(f"\n>>> Training {run_name}")
            try:
                test_m, _ = run_training(
                    model_fn, run_name,
                    epochs=20,
                    spurious_prob_pos=P_POS,
                    spurious_prob_neg=P_NEG,
                    shape=shape,
                    train_tf=train_transforms,
                    eval_tf=val_transforms,
                    **DIFF_CFG,
                )
                summary[run_name] = test_m
            except Exception as e:
                summary[run_name] = {"error": str(e)}
                print(f"!! {run_name} failed: {e}")

            with open(os.path.join("..", "results", "summary_data_augmentation.json"), "w") as f:
                json.dump(summary, f, indent=2)
