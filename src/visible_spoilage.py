"""Explainable image indicators for visible surface deterioration.

These indicators are heuristics, not a mold classifier or food-safety test.
They supplement the learned freshness prediction without replacing it.
"""

from __future__ import annotations

import math

import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor


def _entropy(channel: torch.Tensor, bins: int = 32) -> float:
    histogram = torch.histc(channel, bins=bins, min=0.0, max=1.0)
    probabilities = histogram / histogram.sum().clamp_min(1.0)
    probabilities = probabilities[probabilities > 0]
    return float(-(probabilities * probabilities.log()).sum() / math.log(bins))


def visible_spoilage_indicators(image: Image.Image) -> dict:
    """Return image-wide color/texture cues that can flag visible deterioration."""
    rgb = pil_to_tensor(image.convert("RGB").resize((256, 256))).float() / 255.0
    red, green, blue = rgb
    maximum = rgb.max(dim=0).values
    minimum = rgb.min(dim=0).values
    chroma = maximum - minimum
    saturation = chroma / maximum.clamp_min(1e-6)
    safe_chroma = chroma.clamp_min(1e-6)
    hue = torch.zeros_like(maximum)
    red_is_max = maximum == red
    green_is_max = (maximum == green) & ~red_is_max
    blue_is_max = ~(red_is_max | green_is_max)
    hue[red_is_max] = (((green - blue) / safe_chroma) % 6)[red_is_max]
    hue[green_is_max] = (((blue - red) / safe_chroma) + 2)[green_is_max]
    hue[blue_is_max] = (((red - green) / safe_chroma) + 4)[blue_is_max]
    hue = hue / 6.0
    grayscale = 0.299 * red + 0.587 * green + 0.114 * blue
    blurred = torch.nn.functional.avg_pool2d(
        grayscale[None, None], kernel_size=9, stride=1, padding=4
    )[0, 0]
    local_variation = (grayscale - blurred).abs()

    dark = maximum < 0.24
    # Brown occupies a narrow orange/brown hue band. Using hue here avoids the
    # previous false positive where healthy red tomatoes matched a broad RGB rule.
    brown = (
        (hue >= 0.035) & (hue <= 0.14) & (saturation >= 0.25)
        & (maximum >= 0.12) & (maximum <= 0.72)
    )
    desaturated_rough = (
        (saturation < 0.24) & (maximum > 0.22) & (maximum < 0.88)
        & (local_variation > 0.075)
    )

    dark_ratio = float(dark.float().mean())
    brown_ratio = float(brown.float().mean())
    rough_patch_ratio = float(desaturated_rough.float().mean())
    texture_variation = float(local_variation.mean())
    color_entropy = sum(_entropy(channel) for channel in rgb) / 3

    entropy_texture = max(0.0, color_entropy - 0.72) * min(
        rough_patch_ratio / 0.08, 1.0
    )
    risk_score = min(
        100.0,
        35.0 * min(dark_ratio / 0.18, 1.0)
        + 30.0 * min(brown_ratio / 0.28, 1.0)
        + 25.0 * min(rough_patch_ratio / 0.10, 1.0)
        + 10.0 * min(entropy_texture / 0.20, 1.0),
    )
    level = "high" if risk_score >= 65 else "medium" if risk_score >= 35 else "low"
    return {
        "visible_spoilage_risk": level,
        "visible_spoilage_score": round(risk_score, 1),
        "color_entropy": round(color_entropy, 3),
        "dark_pixel_ratio": round(dark_ratio, 3),
        "brown_pixel_ratio": round(brown_ratio, 3),
        "rough_patch_ratio": round(rough_patch_ratio, 3),
        "texture_variation": round(texture_variation, 3),
        "heuristic_notice": "Visible surface cues only; not proof of mold, internal spoilage, or food safety.",
    }
