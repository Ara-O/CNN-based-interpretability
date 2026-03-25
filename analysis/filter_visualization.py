"""
Filter Visualization — DenseNet121
Plots the raw learned convolutional filter weights layer by layer,
so you can see how filters evolve from low-level to high-level features.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from train import DenseNet121

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_conv_layers(model: nn.Module) -> list[tuple[str, nn.Conv2d]]:
    """Return all (name, Conv2d) pairs in the model, in forward order."""
    return [(name, m) for name, m in model.named_modules()
            if isinstance(m, nn.Conv2d)]


def normalize_filter(f: np.ndarray) -> np.ndarray:
    """Normalize a single filter to [0, 1] for display."""
    f = f - f.min()
    return f / (f.max() + 1e-8)


def plot_layer_filters(
    name: str,
    weight: torch.Tensor,
    max_filters: int = 32,
    save_path: str | None = None,
):
    """
    Plot up to `max_filters` filters from one Conv2d layer.

    weight shape: (out_channels, in_channels, kH, kW)
    For 1x1 convs (kH=kW=1) we visualise the weight as a bar,
    for 3x3+ we show the spatial pattern of the first input channel.
    """
    weight = weight.detach().cpu().numpy()
    n_filters = min(weight.shape[0], max_filters)

    cols = min(n_filters, 8)
    rows = math.ceil(n_filters / cols)

    fig, axes = plt.subplots(rows, cols,
                             figsize=(cols * 1.4, rows * 1.4),
                             squeeze=False)
    kH, kW = weight.shape[2], weight.shape[3]
    title = (f"{name}  |  {weight.shape[0]} filters  "
             f"[in={weight.shape[1]}, {kH}×{kW}]")
    fig.suptitle(title, fontsize=9)

    for idx in range(rows * cols):
        row, col = divmod(idx, cols)
        ax = axes[row][col]
        ax.axis("off")

        if idx >= n_filters:
            continue

        f = weight[idx, 0]          # take first input channel → (kH, kW)

        if kH == 1 and kW == 1:
            # 1×1 conv: show as a thin vertical bar using all input channels
            bar = weight[idx, :, 0, 0]          # (in_channels,)
            bar = normalize_filter(bar)
            ax.imshow(bar[np.newaxis, :], cmap="RdBu_r", aspect="auto",
                      vmin=0, vmax=1)
        else:
            ax.imshow(normalize_filter(f), cmap="RdBu_r",
                      vmin=0, vmax=1, interpolation="nearest")

        ax.set_title(f"f{idx}", fontsize=5, pad=1)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=130, bbox_inches="tight")
        print(f"  saved → {save_path}")
    else:
        plt.show()
    plt.close()


def visualize_all_filters(
    model: nn.Module,
    max_filters: int = 32,
    only_kernels_larger_than: int = 1,   # set to 1 to include 1×1 convs too
    save_dir: str = ".",
):
    """
    Iterate over every Conv2d in the model and produce one plot per layer.

    Args:
        model                   : trained DenseNet121
        max_filters             : how many filters to show per layer (max 32 looks clean)
        only_kernels_larger_than: skip convs whose kernel is ≤ this value
                                  (set to 2 to skip 1×1, set to 0 to show all)
        save_dir                : directory where PNGs are written
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    layers = get_conv_layers(model)
    print(f"Found {len(layers)} Conv2d layers in the model.\n")

    for layer_idx, (name, module) in enumerate(layers):
        kH, kW = module.weight.shape[2], module.weight.shape[3]
        if max(kH, kW) <= only_kernels_larger_than:
            print(f"  [{layer_idx:03d}] {name} ({kH}×{kW}) — skipped")
            continue

        safe_name = name.replace(".", "_")
        save_path = os.path.join(save_dir, f"{layer_idx:03d}_{safe_name}.png")

        print(f"  [{layer_idx:03d}] {name}  "
              f"shape={tuple(module.weight.shape)}  → plotting …")
        plot_layer_filters(name, module.weight,
                           max_filters=max_filters,
                           save_path=save_path)

    print(f"\nDone. All filter plots saved to '{save_dir}/'")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    CKPT = "best_densenet121_clean.pt"   # swap to spurious checkpoint if needed
    SAVE_DIR    = "filter_plots"
    MAX_FILTERS = 32                     # filters shown per layer
    # Set to 1 to include 1×1 convs, 2 to skip them (less informative visually)
    MIN_KERNEL  = 1

    # ── load model ───────────────────────────────────────────────────────────
    model = DenseNet121(dropout=0.3).to(device)
    try:
        state = torch.load(CKPT, map_location=device)
        model.load_state_dict(state)
        print(f"Loaded weights from '{CKPT}'")
    except FileNotFoundError:
        print(f"Checkpoint '{CKPT}' not found — visualising random-init filters.")

    model.eval()

    # ── run visualisation ────────────────────────────────────────────────────
    visualize_all_filters(
        model,
        max_filters=MAX_FILTERS,
        only_kernels_larger_than=MIN_KERNEL,
        save_dir=SAVE_DIR,
    )
