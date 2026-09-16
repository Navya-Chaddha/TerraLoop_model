"""Image-level fruit type and freshness classification for folder datasets.

The supplied dataset contains one fruit per image and folder names rather than
bounding boxes.  This module intentionally solves that problem as an image
classifier; it does not pretend the data can train an object detector.
"""

from __future__ import annotations

import math
from pathlib import Path
from collections.abc import Iterable

import torch
from PIL import Image
from torch import Tensor, nn
from torch.utils.data import Dataset
from torchvision import models, transforms

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
GENERATED_PREFIXES = ("rotated_by_", "saltandpepper_", "vertical_flip_", "translation_")
FRESHNESS_LABELS = ("fresh", "stale")


def label_from_folder(folder: str) -> tuple[str, str]:
    """Convert the source folder convention to (fruit, freshness)."""
    name = "".join(char for char in folder.lower() if char.isalpha())
    if name.startswith("rotten"):
        fruit, freshness = name.removeprefix("rotten"), "stale"
    elif name.startswith("fresh"):
        fruit, freshness = name.removeprefix("fresh"), "fresh"
    else:
        fruit, freshness = name, "fresh"
    if fruit.endswith("ies"):
        fruit = fruit[:-3] + "y"
    else:
        fruit = fruit.rstrip("s")
    return fruit, freshness


def real_image_paths(roots: Iterable[str | Path]) -> list[Path]:
    """Return only source photographs, excluding the dataset's augmented copies."""
    paths: list[Path] = []
    for root in map(Path, roots):
        if root.exists():
            paths.extend(
                p for p in root.rglob("*")
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
                and not p.name.lower().startswith(GENERATED_PREFIXES)
            )
    return sorted(set(paths))


class RealFruitDataset(Dataset):
    """Dataset backed only by real uploaded photos, never generated augmentations."""

    def __init__(self, paths: list[Path], fruit_to_idx: dict[str, int], transform=None):
        self.paths = paths
        self.fruit_to_idx = fruit_to_idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        fruit, freshness = label_from_folder(path.parent.name)
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(self.fruit_to_idx[fruit]), torch.tensor(FRESHNESS_LABELS.index(freshness))


def training_transform():
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(models.ResNet50_Weights.IMAGENET1K_V2.transforms().mean,
                             models.ResNet50_Weights.IMAGENET1K_V2.transforms().std),
    ])


def evaluation_transform():
    return transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize(models.ResNet50_Weights.IMAGENET1K_V2.transforms().mean,
                             models.ResNet50_Weights.IMAGENET1K_V2.transforms().std),
    ])


class FruitFreshnessNet(nn.Module):
    """MobileNetV3 classifier with fruit-type and freshness heads."""

    def __init__(self, num_fruits: int, pretrained: bool = True):
        super().__init__()
        weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.mobilenet_v3_large(weights=weights)
        feature_dim = backbone.classifier[0].in_features
        backbone.classifier = nn.Identity()
        self.backbone = backbone
        self.dropout = nn.Dropout(0.25)
        self.fruit_head = nn.Linear(feature_dim, num_fruits)
        self.freshness_head = nn.Linear(feature_dim, len(FRESHNESS_LABELS))

    def forward(self, images: Tensor) -> dict[str, Tensor]:
        features = self.dropout(self.backbone(images))
        return {"fruit_logits": self.fruit_head(features), "freshness_logits": self.freshness_head(features)}


def prediction_entropy(probabilities: Tensor) -> Tensor:
    """Normalized Shannon entropy: 0 = certain, 1 = maximally uncertain."""
    probabilities = probabilities.clamp_min(1e-8)
    return -(probabilities * probabilities.log()).sum(dim=-1) / math.log(probabilities.shape[-1])
