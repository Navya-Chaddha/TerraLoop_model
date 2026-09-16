"""Optional YOLO produce detector trained on the LVIS category dataset."""

from __future__ import annotations

from pathlib import Path

from PIL import Image


def canonical_category(name: str) -> str:
    """Return the first normalized alias from an LVIS slash-separated label."""
    return name.split("/", 1)[0].strip().lower()


class ProduceDetector:
    def __init__(self, checkpoint: str | Path = "checkpoints/produce_detector.pt"):
        from ultralytics import YOLO

        self.model = YOLO(str(checkpoint))

    def detect(self, image: Image.Image, confidence: float = 0.25) -> list[dict]:
        result = self.model.predict(source=image, conf=confidence, verbose=False)[0]
        detections = []
        if result.boxes is None:
            return detections
        for box, score, class_index in zip(
            result.boxes.xyxy.cpu().tolist(),
            result.boxes.conf.cpu().tolist(),
            result.boxes.cls.cpu().tolist(),
        ):
            detections.append({
                "category": str(result.names[int(class_index)]),
                "confidence": round(float(score) * 100, 1),
                "box": [round(float(value), 1) for value in box],
            })
        return sorted(detections, key=lambda item: item["confidence"], reverse=True)
