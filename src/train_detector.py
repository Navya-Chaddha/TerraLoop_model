"""Fine-tune a YOLO detector on the LVIS fruits-and-vegetables dataset."""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO


def _safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    resolved_destination = destination.resolve()
    with zipfile.ZipFile(archive) as zipped:
        for member in zipped.infolist():
            target = (destination / member.filename).resolve()
            if resolved_destination not in target.parents and target != resolved_destination:
                raise ValueError(f"Unsafe archive member: {member.filename}")
        zipped.extractall(destination)


def prepare_dataset(root: Path) -> Path:
    """Extract nested archives and create a validated local YOLO data YAML."""
    for archive in sorted(root.rglob("*.zip")):
        marker = archive.with_suffix(archive.suffix + ".extracted")
        if not marker.exists():
            _safe_extract(archive, archive.parent)
            marker.touch()
    candidates = [
        candidate
        for candidate in sorted(root.rglob("*.yaml")) + sorted(root.rglob("*.yml"))
        if candidate.name != "data.resolved.yaml"
    ]
    for candidate in candidates:
        text = candidate.read_text(encoding="utf-8", errors="ignore")
        if "train:" in text and ("names:" in text or "nc:" in text):
            config = yaml.safe_load(text)
            dataset_root = candidate.parent.resolve()
            required = {
                "train": dataset_root / "images" / "train",
                "val": dataset_root / "images" / "val",
                "test": dataset_root / "images" / "test",
            }
            missing = [str(path) for path in required.values() if not path.is_dir()]
            if missing:
                raise FileNotFoundError(
                    "Dataset YAML was found, but these split directories are missing: "
                    + ", ".join(missing)
                )

            # The Kaggle release currently points `val` at images/test. Generate a
            # corrected local file so validation metrics use the actual 1,500-image
            # validation split without changing the downloaded source metadata.
            config.update({
                "path": str(dataset_root),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
            })
            resolved = dataset_root / "data.resolved.yaml"
            resolved.write_text(
                yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )
            return resolved
    raise FileNotFoundError(f"No YOLO dataset YAML found under {root}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/lvis_categories"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--model", default="yolo11n.pt")
    args = parser.parse_args()

    data_yaml = prepare_dataset(args.data)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = YOLO(args.model)
    result = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.image_size,
        batch=args.batch_size,
        device=device,
        project="checkpoints/detector_runs",
        name="lvis_produce",
        patience=8,
        exist_ok=True,
    )
    best = Path(result.save_dir) / "weights" / "best.pt"
    destination = Path("checkpoints/produce_detector.pt")
    shutil.copy2(best, destination)
    print(f"Saved best detector to {destination}")


if __name__ == "__main__":
    main()
