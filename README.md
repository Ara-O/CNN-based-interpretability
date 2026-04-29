# Spurious Feature Learning in Chest X-ray Classification

This project studies how CNNs trained on ChestMNIST learn spurious shortcuts and whether
various training interventions reduce reliance on them. A star-shaped marker is drawn on
50% of positive (abnormal) training examples, giving the model an easy textural shortcut.
Models are always evaluated on clean test data so the effect of spurious training is visible.

## Setup

```
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision medmnist scikit-learn tqdm matplotlib pillow numpy
```

ChestMNIST downloads automatically on first run via medmnist. The `data/` directory is
gitignored. You will also need to create `trained_models/` before running any training:

```
mkdir trained_models
```

## Directory Structure

```
deep_learning_project/
├── models/
│   ├── alexnet.py                          # AlexNet (baseline)
│   ├── alexnet_enhanced.py                 # parameter sweep over shape/size/blend
│   ├── resnet50.py
│   ├── vgg16.py
│   ├── inceptionv3.py
│   └── densenet121.py
├── updated_model_experiments/
│   ├── alexnet_data_augmentation.py        # heavy augmentation
│   ├── alexnet_data_heavy_dropout.py       # dropout=0.7 + AvgPool
│   ├── alexnet_logit_magnitude_weighting.py # L2 penalty on logits
│   └── alexnet_saliency_regularization.py  # entropy regularization on saliency
├── interp/
│   ├── interp.py                           # filter, activation, and saliency plots
│   └── filter_visualization.ipynb
├── data_processing/
│   └── data_exploration.ipynb
├── data/                                   # dataset files (gitignored)
├── trained_models/                         # saved checkpoints (gitignored)
├── test_images/                            # sample images for interp.py
└── results/                                # visualization outputs from interp.py
```

## Training

Each script in `models/` is self-contained. Set `SPURIOUS = True` or `False` near the
top of `__main__` to control whether the star marker is injected during training.

```
cd models
python alexnet.py
```

The checkpoint with the best validation AUC is saved to `../trained_models/`.

## Experiments

The scripts in `updated_model_experiments/` are AlexNet variants, each testing a
different approach to reducing spurious feature reliance. Run them from that directory:

```
cd updated_model_experiments
python alexnet_data_augmentation.py
```

| Script | What it tests |
|---|---|
| `alexnet_data_augmentation.py` | random crop, vertical flip, random erasing |
| `alexnet_data_heavy_dropout.py` | dropout=0.7, AvgPool instead of MaxPool |
| `alexnet_logit_magnitude_weighting.py` | adds `0.01 * (logits**2).mean()` to the loss |
| `alexnet_saliency_regularization.py` | entropy loss to disperse gradient attention |

## Interpretability

`interp/interp.py` loads trained checkpoints and produces three sets of visualizations
per model: learned filter weights, activation maps for a test image, and saliency maps
(vanilla gradient and guided backprop).

Edit the `models` list near the bottom of the file to select which checkpoints to analyze,
then run:

```
cd interp
python interp.py
```

Output PNGs are written to `results/<model_id>/`.

## Binary Classification Setup

- Source: ChestMNIST (grayscale chest X-rays, 224x224)
- Label: 1 if any pathology is present, 0 otherwise
- Spurious marker: star shape, alpha-blended at random intensity (0.3-0.7), random position
- Injection rate: 50% of positive examples in train and val; test is always clean
- Optimizer: Adam, lr=1e-4, weight_decay=1e-4
- Scheduler: CosineAnnealingWarmRestarts
- Checkpointing: best val AUC
