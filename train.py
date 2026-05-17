from shared import run_training
from models.alexnet import AlexNet
import os
import json


BRIGHT_STAR = dict(
    shape="star",
    star_size=20,
    color=255,
    blend_range=(1.0, 1.0),
)

if __name__ == "__main__":
    EXPERIMENTS = [
        ("alexnet_clean",         lambda: AlexNet(dropout=0.3),
            dict(spurious_prob_pos=0.0, spurious_prob_neg=0.0)),
        ("alexnet_pos0.5_neg0.0", lambda: AlexNet(dropout=0.3),
            dict(spurious_prob_pos=0.5, spurious_prob_neg=0.0)),
        ("alexnet_pos0.5_neg0.1", lambda: AlexNet(dropout=0.3),
            dict(spurious_prob_pos=0.5, spurious_prob_neg=0.1)),
    ]

    summary = {}
    for run_name, model_fn, kwargs in EXPERIMENTS:
        print(f"\n>>> Training {run_name}")
        try:
            test_m, _ = run_training(
                model_fn, run_name,
                epochs=20,
                **BRIGHT_STAR,
                **kwargs,
            )
            summary[run_name] = test_m
        except Exception as e:
            summary[run_name] = {"error": str(e)}
            print(f"!! {run_name} failed: {e}")

    with open(os.path.join("..", "results", "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)