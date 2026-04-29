# %%

import argparse
import math
import sys
from pathlib import Path

sys.path.append("../")
 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
 
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms import v2
from PIL import Image

# %%
from models.resnet50 import ResNet
from models.densenet121 import DenseNet121
from models.inceptionv3 import Inception3
from models.vgg16 import VGG16
from models.alexnet import AlexNet

# %%
BG      = "#0d0d0f"
SURFACE = "#16161a"
ACCENT  = "#7f5af0"
TEXT    = "#fffffe"
SUBTLE  = "#72757e"
 
plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    SURFACE,
    "axes.edgecolor":    SUBTLE,
    "axes.labelcolor":   TEXT,
    "xtick.color":       SUBTLE,
    "ytick.color":       SUBTLE,
    "text.color":        TEXT,
    "font.family":       "monospace",
})
 

# %%
def load_model(model_class, weight_path):
    model = model_class()
    model.load_state_dict(torch.load(weight_path, map_location="cpu"))
    model.eval()
    return model

# %%
def get_named_conv_layers(model):
    """Return (name, module) pairs for every Conv2d in the model."""
    return [(name, m) for name, m in model.named_modules()
            if isinstance(m, nn.Conv2d)]

# %%
# model = load_alexnet("../best_alexnet_spurious_0.5.pt")
# print(get_named_conv_layers(model))

# %%
def register_activation_hooks(model):
    """Attach forward hooks to Conv2d/ReLU/pool layers; returns (activations, handles)."""
    activations = {}
    handles = []
 
    def make_hook(name):
        def hook(module, inp, out):
            activations[name] = out.detach()
        return hook
 
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.ReLU, nn.MaxPool2d, nn.AdaptiveAvgPool2d)):
            handles.append(module.register_forward_hook(make_hook(name)))
 
    return activations, handles

# %%
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
 
preprocess = transforms.Compose([
    v2.Grayscale(num_output_channels=1),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])
 

# %%
def load_image(path=None):
    if path:
        pil = Image.open(path).convert("RGB")
    else:
        rng = np.random.default_rng(42)
        arr = rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)
        for c in range(3):
            arr[:, :, c] = (arr[:, :, c] * 0.4 +
                            np.tile(rng.integers(0, 256, (28, 28)),
                                    (8, 8))[:224, :224] * 0.6).astype(np.uint8)
        pil = Image.fromarray(arr)
 
    tensor = preprocess(pil).unsqueeze(0)
    return pil, tensor
 

# %%
def denormalize(tensor):
    """Convert a normalized [C,H,W] tensor back to a displayable numpy array."""
    t = tensor.clone().numpy()
    # for c, (m, s) in enumerate(zip(IMAGENET_MEAN, IMAGENET_STD)):
    #     t[c] = t[c] * s + m
    return np.clip(t.transpose(1, 2, 0), 0, 1)
 

# %%
def visualize_filters(model, save_path="alexnet_filters.png"):
    """Plot learned conv filter weights; each tile is one filter."""
    conv_layers = get_named_conv_layers(model)
    n_layers = len(conv_layers)
 
    fig = plt.figure(figsize=(20, 4 * n_layers))
    fig.suptitle("AlexNet · Learned Filter Weights (per Conv layer)",
                 fontsize=16, fontweight="bold", color=TEXT, y=1.01)
 
    for row_idx, (name, layer) in enumerate(conv_layers):
        weights = layer.weight.data.clone()        # [out_ch, in_ch, kH, kW]
        n_filters = weights.shape[0]
 
        w_min, w_max = weights.min(), weights.max()
        weights = (weights - w_min) / (w_max - w_min + 1e-8)

        n_cols = min(n_filters, 32)
        n_rows_inner = math.ceil(n_filters / n_cols)

        ax_label = fig.add_subplot(n_layers, 1, row_idx + 1)
        ax_label.axis("off")
        ax_label.set_title(
            f"Layer: {name}   |   shape: {list(layer.weight.shape)}   "
            f"|   stride: {layer.stride}   pad: {layer.padding}",
            loc="left", color=ACCENT, fontsize=11, pad=8,
        )

        gs_inner = gridspec.GridSpecFromSubplotSpec(
            n_rows_inner, n_cols, subplot_spec=ax_label.get_subplotspec(),
            hspace=0.05, wspace=0.05,
        )
 
        for fi in range(n_filters):
            ax = fig.add_subplot(gs_inner[fi // n_cols, fi % n_cols])
            f = weights[fi]                          # [in_ch, kH, kW]
            if f.shape[0] == 3:
                img = f[:3].permute(1, 2, 0).numpy()
                img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                ax.imshow(img, interpolation="nearest")
            else:
                img = f.mean(0).numpy()
                img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                ax.imshow(img, cmap="gray", interpolation="nearest")
            ax.axis("off")
 
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close(fig)
    print(f"  ✓  Filter weights  → {save_path}")
 

# %%
def visualize_activations(model, image_tensor, pil_image,
                          save_path="alexnet_activations.png"):
    """Forward pass and plot feature maps at each conv/relu/pool (up to 32 channels)."""
    activations, handles = register_activation_hooks(model)
 
    with torch.no_grad():
        output = model(image_tensor)
    # pred_class = output.argmax(dim=1).item()
 
    for h in handles:
        h.remove()
 
    interesting = {k: v for k, v in activations.items()
                   if any(t in k for t in ("features.", "avgpool"))}
 
    n_layers = len(interesting)
    fig = plt.figure(figsize=(22, 4.5 * n_layers))
    fig.suptitle(
        f"AlexNet · Activation Maps ",
        fontsize=15, fontweight="bold", color=TEXT, y=1.005,
    )
 
    for row_idx, (name, act) in enumerate(interesting.items()):
        act_np = act[0].numpy()                   # [C, H, W]
        n_ch   = act_np.shape[0]
        n_cols = min(n_ch, 32)
        n_rows_inner = math.ceil(min(n_ch, 32) / n_cols)
 
        ax_label = fig.add_subplot(n_layers, 1, row_idx + 1)
        ax_label.axis("off")
        ax_label.set_title(
            f"Layer: {name}   |   output shape: {list(act.shape[1:])}",
            loc="left", color=ACCENT, fontsize=11, pad=8,
        )
 
        gs_inner = gridspec.GridSpecFromSubplotSpec(
            n_rows_inner, n_cols + 1,               # +1 for input thumb
            subplot_spec=ax_label.get_subplotspec(),
            hspace=0.05, wspace=0.05,
        )
 
        ax_in = fig.add_subplot(gs_inner[:, 0])
        ax_in.imshow(pil_image.resize((56, 56)))
        ax_in.set_title("input", fontsize=7, color=SUBTLE, pad=2)
        ax_in.axis("off")
 
        for ci in range(min(n_ch, 32)):
            ax = fig.add_subplot(gs_inner[ci // n_cols, (ci % n_cols) + 1])
            ch = act_np[ci]
            ax.imshow(ch, cmap="inferno", interpolation="nearest")
            ax.axis("off")
 
    plt.tight_layout()
    fig.savefig(save_path, dpi=130, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close(fig)
    print(f"Activation maps → {save_path}")
 
 

# %%

def visualize_saliency(model, image_tensor, pil_image,
                       save_path="alexnet_saliency.png"):
    """Vanilla saliency and guided backprop; saves a 4-panel plot."""
    # ── Vanilla saliency ──────────────────────────────────────────────────────
    inp = image_tensor.clone().requires_grad_(True)
    output = model(inp)
    pred_class = output.argmax(dim=1).item()
    # Just takes the logits, argmax will always be 0
    score = output[0, pred_class]
    model.zero_grad()
    score.backward()
 
    saliency = inp.grad.data.abs()[0]              # [3, H, W]
    saliency_max, _ = saliency.max(dim=0)          # max over channels
    saliency_np = saliency_max.numpy()
 
    # ── Guided Backprop ───────────────────────────────────────────────────────
    # clamp to positive gradients only (guided backprop)
    saved_relu_bwd = {}
 
    def guided_hook_factory(module):
        def backward_hook(module, grad_in, grad_out):
            return (torch.clamp(grad_in[0], min=0),)
        return module.register_backward_hook(backward_hook)
 
    guided_handles = []
    for m in model.modules():
        if isinstance(m, nn.ReLU):
            guided_handles.append(guided_hook_factory(m))
 
    inp_g = image_tensor.clone().requires_grad_(True)
    out_g = model(inp_g)
    model.zero_grad()
    out_g[0, pred_class].backward()
 
    guided_sal = inp_g.grad.data[0]               # [3, H, W]
    guided_sal = guided_sal - guided_sal.min()
    guided_sal = guided_sal / (guided_sal.max() + 1e-8)
    guided_sal_np = guided_sal.permute(1, 2, 0).numpy()
 
    for h in guided_handles:
        h.remove()
 
    # ── Plot ──────────────────────────────────────────────────────────────────
    input_display = denormalize(image_tensor[0])
 
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(
        f"AlexNet · Gradient Saliency",
        fontsize=14, fontweight="bold", color=TEXT,
    )
    fig.patch.set_facecolor(BG)
 
    panels = [
        (input_display,          "Input Image",      "viridis",  False),
        (saliency_np,            "Vanilla Saliency", "hot",      True),
        (guided_sal_np,          "Guided Backprop",  "viridis",  False),
        (input_display * np.expand_dims(saliency_np / saliency_np.max(), -1),
                                 "Overlay",          "viridis",  False),
    ]
 
    for ax, (img, title, cmap, use_cmap) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        ax.imshow(img, cmap=cmap if use_cmap else None)
        ax.set_title(title, color=ACCENT, fontsize=12, pad=6)
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_edgecolor(SUBTLE)
 
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close(fig)
    print(f"Gradient saliency → {save_path}")
 
 

# %%
def print_architecture(model):
    print("\n── AlexNet Architecture ──────────────────────────────────────")
    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear, nn.MaxPool2d,
                          nn.ReLU, nn.Dropout, nn.AdaptiveAvgPool2d)):
            indent = "  " * name.count(".")
            params = (f"  [{sum(p.numel() for p in m.parameters()):,} params]"
                      if list(m.parameters()) else "")
            print(f"  {indent}{name:30s}  {m}{params}")
    print()
 

# %%
models = [
    # Alexnet
    # {
    #     "id": "best_alexnet_spurious_heavy_dropout",
    #     "model": AlexNet,
    #     "weight_pth": "best_alexnet_spurious_heavy_dropout.pt",
    # },
    {
        "id": "alexnet_spurious_randomized_0.5_no_intervention",
        "model": AlexNet,
        "weight_pth": "best_alexnet_spurious_0.5_randomized.pt",
    },
    # # Densenet121
    # {
    #     "id": "densenet121_clean",
    #     "model": DenseNet121,
    #     "weight_pth": "best_densenet121_clean.pt",
    # },
    # {
    #     "id": "densenet121_spurious_randomized_0.5_no_intervention",
    #     "model": DenseNet121,
    #     "weight_pth": "best_densenet121_spurious_0.5_randomized.pt",
    # },
    # # Resnet50
    # {
    #     "id": "resnet50_clean",
    #     "model": ResNet,
    #     "weight_pth": "best_resnet50_clean.pt",
    # },
    # {
    #     "id": "resnet50_spurious_randomized_0.5_no_intervention",
    #     "model": ResNet,
    #     "weight_pth": "best_resnet50_spurious_0.5_randomized.pt",
    # },
    # # VGG16
    # {
    #     "id": "vgg16_clean",
    #     "model": VGG16,
    #     "weight_pth": "best_vgg16_clean.pt",
    # },
    # {
    #     "id": "vgg16_spurious_randomized_0.5_no_intervention",
    #     "model": VGG16,
    #     "weight_pth": "best_vgg16_spurious_0.5_randomized.pt",
    # },
]

pil_image, image_tensor = load_image("../test_images/img7_star_blended.png")

for model in models:
    outdir = Path(f"../results/{model['id']}/")
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading {model['id']} …")
    trained_model = load_model(model_class=model['model'], weight_path=f"../trained_models/{model['weight_pth']}")
    # print_architecture(model)

    print("Loading image …")

    visualize_filters(trained_model, save_path=str(outdir / f"filters_{model['id']}.png"))

    visualize_activations(trained_model, image_tensor, pil_image, save_path=str(outdir / f"activations_{model['id']}.png"))

    # if mode in ("all", "saliency"):
    visualize_saliency(trained_model, image_tensor, pil_image, save_path=str(outdir / f"saliency_{model['id']}.png"))

    print("\nDone. Output files saved to:", outdir.resolve())


