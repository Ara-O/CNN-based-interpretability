from shared import run_training
from models.alexnet import AlexNet
from models.densenet121 import DenseNet121
from models.resnet50 import resnet50
from models.vgg16 import VGG16
from models.inceptionv3 import Inception3

import os
import json


STAR = dict(
    shape="star",
    star_size=20,
    blend_range=(0.8, 1.0),
)

if __name__ == "__main__":
    EXPERIMENTS = [
        ("alexnet_clean",         lambda: AlexNet(dropout=0.3),
            dict(spurious_prob_pos=0.0, spurious_prob_neg=0.0)),
        ("alexnet_pos0.5_neg0.0", lambda: AlexNet(dropout=0.3),
            dict(spurious_prob_pos=0.5, spurious_prob_neg=0.0)),
        ("alexnet_pos0.0_neg0.5", lambda: AlexNet(dropout=0.3),
            dict(spurious_prob_pos=0.0, spurious_prob_neg=0.5)),
        ("alexnet_pos0.1_neg0.5", lambda: AlexNet(dropout=0.3),
            dict(spurious_prob_pos=0.1, spurious_prob_neg=0.5)),
        ("alexnet_pos0.5_neg0.1", lambda: AlexNet(dropout=0.3),
            dict(spurious_prob_pos=0.5, spurious_prob_neg=0.1)),

        ("densenet121_clean",         lambda: DenseNet121(dropout=0.3),
            dict(spurious_prob_pos=0.0, spurious_prob_neg=0.0)),
        ("densenet121_pos0.5_neg0.0", lambda: DenseNet121(dropout=0.3),
            dict(spurious_prob_pos=0.5, spurious_prob_neg=0.0)),
        ("densenet121_pos0.0_neg0.5", lambda: DenseNet121(dropout=0.3),
            dict(spurious_prob_pos=0.0, spurious_prob_neg=0.5)),
        ("densenet121_pos0.1_neg0.5", lambda: DenseNet121(dropout=0.3),
            dict(spurious_prob_pos=0.1, spurious_prob_neg=0.5)),
        ("densenet121_pos0.5_neg0.1", lambda: DenseNet121(dropout=0.3),
            dict(spurious_prob_pos=0.5, spurious_prob_neg=0.1)),

        ("resnet50_clean",         lambda: resnet50(dropout=0.3),
            dict(spurious_prob_pos=0.0, spurious_prob_neg=0.0)),
        ("resnet50_pos0.5_neg0.0", lambda: resnet50(dropout=0.3),
            dict(spurious_prob_pos=0.5, spurious_prob_neg=0.0)),
        ("resnet50_pos0.0_neg0.5", lambda: resnet50(dropout=0.3),
            dict(spurious_prob_pos=0.0, spurious_prob_neg=0.5)),
        ("resnet50_pos0.1_neg0.5", lambda: resnet50(dropout=0.3),
            dict(spurious_prob_pos=0.1, spurious_prob_neg=0.5)),
        ("resnet50_pos0.5_neg0.1", lambda: resnet50(dropout=0.3),
            dict(spurious_prob_pos=0.5, spurious_prob_neg=0.1)),

        ("vgg16_clean",         lambda: VGG16(dropout=0.3),
            dict(spurious_prob_pos=0.0, spurious_prob_neg=0.0)),
        ("vgg16_pos0.5_neg0.0", lambda: VGG16(dropout=0.3),
            dict(spurious_prob_pos=0.5, spurious_prob_neg=0.0)),
        ("vgg16_pos0.0_neg0.5", lambda: VGG16(dropout=0.3),
            dict(spurious_prob_pos=0.0, spurious_prob_neg=0.5)),
        ("vgg16_pos0.1_neg0.5", lambda: VGG16(dropout=0.3),
            dict(spurious_prob_pos=0.1, spurious_prob_neg=0.5)),
        ("vgg16_pos0.5_neg0.1", lambda: VGG16(dropout=0.3),
            dict(spurious_prob_pos=0.5, spurious_prob_neg=0.1)),

        ("inceptionv3_clean",         lambda: Inception3(dropout=0.3),
            dict(spurious_prob_pos=0.0, spurious_prob_neg=0.0)),
        ("inceptionv3_pos0.5_neg0.0", lambda: Inception3(dropout=0.3),
            dict(spurious_prob_pos=0.5, spurious_prob_neg=0.0)),
        ("inceptionv3_pos0.0_neg0.5", lambda: Inception3(dropout=0.3),
            dict(spurious_prob_pos=0.0, spurious_prob_neg=0.5)),
        ("inceptionv3_pos0.1_neg0.5", lambda: Inception3(dropout=0.3),
            dict(spurious_prob_pos=0.1, spurious_prob_neg=0.5)),
        ("inceptionv3_pos0.5_neg0.1", lambda: Inception3(dropout=0.3),
            dict(spurious_prob_pos=0.5, spurious_prob_neg=0.1)),

    ]

    summary = {}
    for run_name, model_fn, kwargs in EXPERIMENTS:
        print(f"\n>>> Training {run_name}")
        try:
            test_m, _ = run_training(
                model_fn, run_name,
                epochs=20,
                **STAR,
                **kwargs,
            )
            summary[run_name] = test_m
        except Exception as e:
            summary[run_name] = {"error": str(e)}
            print(f"!! {run_name} failed: {e}")

    with open(os.path.join("..", "results", "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)