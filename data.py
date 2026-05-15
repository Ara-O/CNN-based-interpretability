import numpy as np
from PIL import Image, ImageDraw
from medmnist import ChestMNIST
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2


def draw_star(img: Image.Image, size=20, pos=(10, 10), color=255) -> Image.Image:
    img = img.copy()
    draw = ImageDraw.Draw(img)
    cx, cy = pos
    outer, inner = size, size // 2.5
    points = []
    for i in range(10):
        angle = i * 36 - 90
        r = outer if i % 2 == 0 else inner
        rad = np.radians(angle)
        points.append((cx + r * np.cos(rad), cy + r * np.sin(rad)))
    draw.polygon(points, fill=color)
    return img


class BinaryChestMNIST(Dataset):
    def __init__(self, split, spurious=False, spurious_prob=1.0,
                 star_size=20, randomize_star_pos=False, transform=None, **kwargs):
        self.dataset = ChestMNIST(split=split, **kwargs)
        self.spurious = spurious
        self.spurious_prob = spurious_prob
        self.star_size = star_size
        self.randomize_star_pos = randomize_star_pos
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def _get_star_pos(self, img_w, img_h):
        margin = self.star_size + 4
        if self.randomize_star_pos:
            cx = np.random.randint(margin, img_w - margin)
            cy = np.random.randint(margin, img_h - margin)
        else:
            cx, cy = margin, margin
        return cx, cy

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        binary_label = 1 if label.any() else 0

        if self.spurious and binary_label == 1:
            if np.random.random() < self.spurious_prob:
                w, h = img.size
                pos = self._get_star_pos(w, h)
                img = draw_star(img, size=self.star_size, pos=pos)

        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(binary_label, dtype=torch.long)

