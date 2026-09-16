"""Train the image-level MobileNetV3 model on the real fruit photos.

Example: python -m src.train_freshness --epochs 20
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.freshness_classifier import (FruitFreshnessNet, RealFruitDataset,
    evaluation_transform, label_from_folder, real_image_paths, training_transform)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", nargs="+", default=["data/research/original", "data/collected_uploads/labelled"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(42); random.seed(42)
    source_paths = real_image_paths(args.data)
    if not source_paths:
        raise SystemExit("No real photos found. Check --data paths.")
    fruits = sorted({label_from_folder(p.parent.name)[0] for p in source_paths})
    fruit_to_idx = {fruit: i for i, fruit in enumerate(fruits)}
    # Split original source photos per class.  Generated augmentations never enter
    # validation, so validation reflects unseen original photographs.
    paths_by_label: dict[tuple[str, str], list[Path]] = {}
    for path in source_paths:
        paths_by_label.setdefault(label_from_folder(path.parent.name), []).append(path)
    train_paths: list[Path] = []
    validation_paths: list[Path] = []
    for paths in paths_by_label.values():
        random.Random(42).shuffle(paths)
        n_val = max(1, round(len(paths) * 0.2))
        validation_paths.extend(paths[:n_val])
        train_paths.extend(paths[n_val:])
    train_set = RealFruitDataset(train_paths, fruit_to_idx, training_transform())
    val_set = RealFruitDataset(validation_paths, fruit_to_idx, evaluation_transform())
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = FruitFreshnessNet(len(fruits), not args.no_pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    best_accuracy = -1.0; checkpoint = Path("checkpoints/freshness_cnn.pth")
    checkpoint.parent.mkdir(exist_ok=True)
    print(f"Using {len(source_paths)} source photos ({len(train_paths)} train, {len(validation_paths)} validation).")
    for epoch in range(1, args.epochs + 1):
        model.train()
        for images, fruit, fresh in train_loader:
            images, fruit, fresh = images.to(device), fruit.to(device), fresh.to(device)
            output = model(images)
            loss = loss_fn(output["fruit_logits"], fruit) + loss_fn(output["freshness_logits"], fresh)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        model.eval(); correct = total = 0
        with torch.no_grad():
            for images, fruit, fresh in val_loader:
                output = model(images.to(device))
                correct += ((output["fruit_logits"].argmax(1).cpu() == fruit) & (output["freshness_logits"].argmax(1).cpu() == fresh)).sum().item()
                total += len(fruit)
        accuracy = correct / max(total, 1)
        print(f"epoch {epoch:02d}: joint validation accuracy={accuracy:.3f}")
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save({"model_state_dict": model.state_dict(), "fruit_classes": fruits,
                        "best_joint_accuracy": accuracy}, checkpoint)
    print(f"Saved best model to {checkpoint}")
    print(json.dumps({"source_images": len(source_paths), "best_joint_accuracy": best_accuracy}, indent=2))


if __name__ == "__main__":
    main()
