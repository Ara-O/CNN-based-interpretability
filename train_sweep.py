from shared import run_training
from models.alexnet import AlexNet
from models.resnet50 import resnet50
from models.inceptionv3 import Inception3

import os
import json

# Three marker shapes
# SHAPES = ["star", "circle", "wave"]
SHAPES = ["circle", "wave"]

DIFFICULTY_CONFIGS = {
    "obvious":      dict(star_size=35, blend_range=(0.8, 1.0)),
    "intermediate": dict(star_size=20, blend_range=(0.6, 0.8)), # or moderate
    # "subtle":       dict(star_size=15, blend_range=(0.4, 0.6)),
}

CONTAMINATION_RATIOS = [
    (0.5, 0.0),
    # (0.5, 0.1),
    (0.5, 0.3),
    # (0.5, 0.5),
    (0.3, 0.5),
    # (0.1, 0.5),
    (0.0, 0.5),
]

ARCHITECTURES = [
    ("alexnet",     lambda: AlexNet(dropout=0.3)),
    ("inceptionv3", lambda: Inception3(dropout=0.3)),
    ("resnet50",    lambda: resnet50(dropout=0.3)),
]

if __name__ == "__main__":
    summary = {}

    for arch_name, model_fn in ARCHITECTURES:
        for shape in SHAPES:
            for difficulty, diff_cfg in DIFFICULTY_CONFIGS.items():
                for p_pos, p_neg in CONTAMINATION_RATIOS:
                    run_name = f"{arch_name}_{shape}_{difficulty}_pos{p_pos}_neg{p_neg}"
                    print(f"\n>>> Training {run_name}")
                    try:
                        test_m, _ = run_training(
                            model_fn, run_name,
                            epochs=20,
                            spurious_prob_pos=p_pos,
                            spurious_prob_neg=p_neg,
                            shape=shape,
                            **diff_cfg,
                        )
                        summary[run_name] = test_m
                    except Exception as e:
                        summary[run_name] = {"error": str(e)}
                        print(f"!! {run_name} failed: {e}")

                    with open(os.path.join("..", "results", "summary.json"), "w") as f:
                        json.dump(summary, f, indent=2)
