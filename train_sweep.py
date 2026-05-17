from shared import run_training
from models.alexnet import AlexNet 
from models.resnet50    import ResNet
from models.vgg16       import VGG16
from models.densenet121 import DenseNet121
from models.inceptionv3 import Inception3
import os
import json

ARCHITECTURES = [
    ("alexnet",     lambda: AlexNet(dropout=0.3)),
    ("resnet50",    lambda: ResNet(dropout=0.3)),
    ("vgg16",       lambda: VGG16(dropout=0.3)),
    ("densenet121", lambda: DenseNet121(dropout=0.3)),
    ("inception3",  lambda: Inception3(dropout=0.3)),
]

if __name__ == "__main__":
    SPUR_P_POS, SPUR_P_NEG = 0.5, 0.1
    shapes = ["star", "circle", "wave"]
    sizes  = {"small": 15, "medium": 20, "large": 35}
    blends = {"subtle": (0.4, 0.6), "moderate": (0.6, 0.8)}

    summary = {}
    for arch_name, model_fn in ARCHITECTURES:
        for shape in shapes:
            for size_label, size_px in sizes.items():
                for blend_label, blend_range in blends.items():
                    base = f"{arch_name}_{shape}_{size_label}_{blend_label}"

                    # Model A: trained WITH spurious cue
                    try:
                        spur_m, _ = run_training(
                            model_fn, f"{base}_spur",
                            spurious_prob_pos=SPUR_P_POS,
                            spurious_prob_neg=SPUR_P_NEG,
                            shape=shape, color=None, star_size=size_px, blend_range=blend_range,
                            epochs=10,
                        )
                    except Exception as e:
                        spur_m = {"error": str(e)}

                    # Model B: clean baseline
                    try:
                        clean_m, _ = run_training(
                            model_fn, f"{base}_clean",
                            spurious_prob_pos=0.0, spurious_prob_neg=0.0,
                            shape=shape, color=None, star_size=size_px, blend_range=blend_range,
                            epochs=10,
                        )
                    except Exception as e:
                        clean_m = {"error": str(e)}

                    summary[base] = {"spur": spur_m, "clean": clean_m}
                    
        with open(os.path.join("..", "results", "summary_sweep.json"), "w") as f:
            json.dump(summary, f, indent=2)