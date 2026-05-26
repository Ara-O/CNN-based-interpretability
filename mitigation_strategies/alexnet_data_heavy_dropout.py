import os
import json
import torch
import torch.nn as nn

import sys
sys.path.append("..")

from shared import run_training

SHAPES = ["circle"]

DIFF_CFG = dict(star_size=20, blend_range=(0.6, 0.8))  # moderate

P_POS, P_NEG = 0.5, 0.3

DROPOUT = 0.7


# AlexNet with heavy dropout and AvgPool instead of MaxPool.
class AlexNetHeavyDropout(nn.Module):
    def __init__(self, dropout: float = 0.7):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11, stride=4),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=3, stride=2),

            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=3, stride=2),

            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=3, stride=2),
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


ARCHITECTURES = [
    ("alexnet_heavy_dropout", lambda: AlexNetHeavyDropout(dropout=DROPOUT)),
]

if __name__ == "__main__":
    summary = {}

    for arch_name, model_fn in ARCHITECTURES:
        for shape in SHAPES:
            run_name = f"MITIGATION_STRATEGY_HEAVY_DROPOUT_{arch_name}_{shape}_intermediate_pos{P_POS}_neg{P_NEG}"
            print(f"\n>>> Training {run_name}")
            try:
                test_m, _ = run_training(
                    model_fn, run_name,
                    epochs=20,
                    spurious_prob_pos=P_POS,
                    spurious_prob_neg=P_NEG,
                    shape=shape,
                    **DIFF_CFG,
                )
                summary[run_name] = test_m
            except Exception as e:
                summary[run_name] = {"error": str(e)}
                print(f"!! {run_name} failed: {e}")

            with open(os.path.join("..", "results", "summary_heavy_dropout.json"), "w") as f:
                json.dump(summary, f, indent=2)
