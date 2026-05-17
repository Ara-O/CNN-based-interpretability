from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image, ImageDraw
from medmnist import ChestMNIST
import torch 

def _local_color(img, cx, cy, size):
    arr = np.array(img)
    x0, y0 = max(cx - size, 0), max(cy - size, 0)
    x1, y1 = min(cx + size, arr.shape[1]), min(cy + size, arr.shape[0])
    local_mean = arr[y0:y1, x0:x1].mean()
    return int(min(local_mean + 30, 255))


def draw_star(img, size=20, pos=(10, 10), color=None, blend_alpha=0.5):
    img = img.copy()
    cx, cy = pos
    if color is None:
        color = _local_color(img, cx, cy, size)
    overlay = img.copy()
    draw = ImageDraw.Draw(overlay)
    outer, inner = size, size / 2.5
    points = []
    for i in range(10):
        angle = i * 36 - 90
        r = outer if i % 2 == 0 else inner
        rad = np.radians(angle)
        points.append((cx + r * np.cos(rad), cy + r * np.sin(rad)))
    draw.polygon(points, fill=color)
    return Image.blend(img, overlay, alpha=blend_alpha)

def draw_circle(img, size=20, pos=(10, 10), color=None, blend_alpha=0.5):
    img = img.copy()
    cx, cy = pos
    if color is None:
        color = _local_color(img, cx, cy, size)
    overlay = img.copy()
    draw = ImageDraw.Draw(overlay)
    r = size
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
    return Image.blend(img, overlay, alpha=blend_alpha)

def draw_wave(img, size=20, pos=(10, 10), color=None, blend_alpha=0.5):
    img = img.copy()
    cx, cy = pos
    if color is None:
        color = _local_color(img, cx, cy, size)
    overlay = img.copy()
    draw = ImageDraw.Draw(overlay)
    w, h = img.size
    amplitude = size * 0.4
    thickness = max(size // 4, 2)
    x_start = max(cx - size, 0)
    x_end = min(cx + size, w)
    freq = 2 * np.pi / size  # one full cycle across `size` pixels
    for x in range(x_start, x_end):
        y_centre = int(cy + amplitude * np.sin(freq * (x - cx)))
        y0 = max(y_centre - thickness // 2, 0)
        y1 = min(y_centre + thickness // 2, h - 1)
        for y in range(y0, y1 + 1):
            draw.point((x, y), fill=color)
    return Image.blend(img, overlay, alpha=blend_alpha)


SHAPE_FNS = {
    "star":   draw_star,
    "circle": draw_circle,
    "wave":   draw_wave,
}

class BinaryChestMNIST(Dataset):
    def __init__(self, split,
                 spurious_prob_pos=0.0, spurious_prob_neg=0.0,
                 shape="star", star_size=20,
                 color=None, blend_range=(0.5, 0.5),
                 randomize_star_pos=False, transform=None,
                 download=True, size=224):
        self.dataset = ChestMNIST(split=split, download=download, size=size)
        self.spurious_prob_pos = spurious_prob_pos
        self.spurious_prob_neg = spurious_prob_neg
        self.shape = shape
        self.star_size = star_size
        self.color = color                   # None → local adaptive; int → fixed
        self.blend_range = blend_range       # (1.0, 1.0) → fully opaque
        self.randomize_star_pos = randomize_star_pos
        self.transform = transform
        self.draw_fn = SHAPE_FNS[shape]

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        y = 1 if label.any() else 0

        prob = self.spurious_prob_pos if y == 1 else self.spurious_prob_neg
        if prob > 0 and np.random.random() < prob:
            w, h = img.size
            margin = self.star_size + 4
            if self.randomize_star_pos:
                cx = np.random.randint(margin, w - margin)
                cy = np.random.randint(margin, h - margin)
            else:
                cx, cy = margin, margin
            lo, hi = self.blend_range
            alpha = np.random.uniform(lo, hi) if hi > lo else lo
            img = self.draw_fn(img, size=self.star_size, pos=(cx, cy),
                               color=self.color, blend_alpha=alpha)

        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.dataset)