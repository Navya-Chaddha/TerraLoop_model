"""Simple camera-first interface for the existing trained models."""
from __future__ import annotations

import hashlib
import io
import json
import shutil
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from uuid import uuid4

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from src.freshness_classifier import (
    FRESHNESS_LABELS,
    FruitFreshnessNet,
    evaluation_transform,
    prediction_entropy,
)
from src.produce_detector import ProduceDetector, canonical_category
from src.visible_spoilage import visible_spoilage_indicators

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data/collected_uploads"
RAW = DATASET / "raw"
CROPS = DATASET / "crops"
RECORDS = DATASET / "records"
SUSPECTED = DATASET / "suspected"
STATIC = Path(__file__).with_name("static")
engine: Optional["FreshnessEngine"] = None
produce_detector: Optional[ProduceDetector] = None
model_errors: list[str] = []
MAX_BYTES = 15 * 1024 * 1024


class FreshnessEngine:
    def __init__(self, checkpoint: str | Path):
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.fruits = saved["fruit_classes"]
        self.validation_accuracy = float(saved.get("best_joint_accuracy", 0.0))
        self.model = FruitFreshnessNet(len(self.fruits), pretrained=False)
        self.model.load_state_dict(saved["model_state_dict"])
        self.model.eval()
        self.transform = evaluation_transform()

    @torch.inference_mode()
    def predict(self, image: Image.Image) -> dict:
        output = self.model(self.transform(image.convert("RGB")).unsqueeze(0))
        fruit_probs = torch.softmax(output["fruit_logits"], 1)[0]
        fresh_probs = torch.softmax(output["freshness_logits"], 1)[0]
        uncertainty = float(
            (prediction_entropy(fruit_probs[None]) + prediction_entropy(fresh_probs[None])).item() / 2
        )
        return {
            "fruit": self.fruits[int(fruit_probs.argmax())],
            "fruit_confidence": round(float(fruit_probs.max()) * 100, 1),
            "freshness": FRESHNESS_LABELS[int(fresh_probs.argmax())],
            "freshness_score": round(float(fresh_probs[0]) * 100, 1),
            "freshness_confidence": round(float(fresh_probs.max()) * 100, 1),
            "uncertainty_entropy": round(uncertainty, 3),
            "review_recommended": uncertainty > 0.55 or float(fresh_probs.max()) < 0.65,
            "validation_joint_accuracy": round(self.validation_accuracy, 3),
        }


def _decode(contents: bytes) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(contents))
        if image.width * image.height > 24_000_000:
            raise HTTPException(413, "Image is too large. Use a smaller camera frame.")
        return ImageOps.exif_transpose(image).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise HTTPException(400, "Camera frame could not be read. Try another snapshot.") from exc


def _suspected_regions(crop: Image.Image) -> list[dict]:
    """Return visible dark/brown/rough connected areas for optional snapshots."""
    rgb = np.asarray(crop.resize((256, 256)).convert("RGB"))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    brown = (hsv[..., 0] >= 9) & (hsv[..., 0] <= 50) & (hsv[..., 1] >= 70) & (hsv[..., 2] <= 205)
    dark = hsv[..., 2] < 65
    texture = cv2.absdiff(gray, cv2.GaussianBlur(gray, (9, 9), 0)) > 22
    mask = ((brown | dark) & texture).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    regions = []
    for index in range(1, count):
        x, y, width, height, area = stats[index]
        if area < 80 or area / (width * height) < 0.12:
            continue
        regions.append({
            "box": [round(x / 256, 4), round(y / 256, 4), round((x + width) / 256, 4), round((y + height) / 256, 4)],
            "area_ratio": round(float(area / (256 * 256)), 4),
            "factor": "dark/brown textured patch",
        })
    return sorted(regions, key=lambda region: region["area_ratio"], reverse=True)[:5]


def analyze_camera_frame(image: Image.Image) -> tuple[dict, list[Image.Image]]:
    if engine is None or produce_detector is None:
        raise HTTPException(503, "Models are not ready. Check /health and try again.")
    detections = produce_detector.detect(image)
    objects, crops = [], []
    for item in detections:
        left, top, right, bottom = map(int, item["box"])
        left, top = max(0, left), max(0, top)
        right, bottom = min(image.width, right), min(image.height, bottom)
        if right <= left or bottom <= top:
            continue
        crop = image.crop((left, top, right, bottom))
        category = canonical_category(item["category"])
        surface = visible_spoilage_indicators(crop)
        supported = category in engine.fruits
        prediction = engine.predict(crop) if supported else {}
        objects.append({
            "object_id": str(len(objects)),
            "category": category,
            "raw_category": item["category"],
            "box": [left, top, right, bottom],
            "detector_confidence": item["confidence"],
            "freshness": prediction.get("freshness"),
            "freshness_score": prediction.get("freshness_score"),
            "freshness_confidence": prediction.get("freshness_confidence"),
            "uncertainty_entropy": prediction.get("uncertainty_entropy"),
            "freshness_status": "predicted" if supported else "not_model_supported",
            "analysis_mode": "trained_freshness_model" if supported else "visible_heuristic_only",
            "visible_surface_anomaly_score": surface["visible_spoilage_score"],
            "visible_surface_risk": surface["visible_spoilage_risk"],
            "surface_features": surface,
            "suspected_regions": _suspected_regions(crop),
            "review_recommended": prediction.get("review_recommended", True) or item["confidence"] < 50,
        })
        crops.append(crop)
    result = {
        "objects": objects,
        "detections": [{"category": o["category"], "confidence": o["detector_confidence"], "box": o["box"]} for o in objects],
        "supported_fruits": engine.fruits,
        "message": "Each detected fruit was analyzed independently." if objects else "No produce detected. Move closer and try again.",
        "limitation": "Visible surface patterns only; camera images cannot confirm internal spoilage, mold or food safety.",
        "detector_metrics": {"precision": 0.597, "recall": 0.226, "map50": 0.264, "map50_95": 0.166},
    }
    return result, crops


def _save_capture(contents: bytes, image: Image.Image, result: dict, crops: list[Image.Image]) -> str:
    sample_id = uuid4().hex
    for folder in (RAW, CROPS, RECORDS):
        folder.mkdir(parents=True, exist_ok=True)
    raw_path = RAW / f"{sample_id}.png"
    image.save(raw_path)
    for index, crop in enumerate(crops):
        crop.save(CROPS / f"{sample_id}_{index}.png")
        result["objects"][index]["crop_path"] = str(CROPS / f"{sample_id}_{index}.png")
    record = {
        "sample_id": sample_id,
        "source": "camera_capture",
        "stored_at": datetime.now(timezone.utc).isoformat(),
        "sha256": hashlib.sha256(contents).hexdigest(),
        "raw_image": str(raw_path),
        "prediction": result,
        "label_status": "unlabelled",
    }
    (RECORDS / f"{sample_id}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return sample_id


@asynccontextmanager
async def lifespan(_: FastAPI):
    global engine, produce_detector, model_errors
    model_errors = []
    for name, factory, filename in (
        ("freshness", FreshnessEngine, "freshness_cnn.pth"),
        ("category", ProduceDetector, "produce_detector.pt"),
    ):
        try:
            loaded = await run_in_threadpool(factory, ROOT / "checkpoints" / filename)
        except Exception as exc:
            model_errors.append(f"{name} model failed to load: {exc}")
            loaded = None
        if name == "freshness": engine = loaded
        else: produce_detector = loaded
    yield


app = FastAPI(title="Fruit freshness camera", version="2.1", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def home():
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store"})


@app.get("/health")
def health():
    return {"status": "ready" if engine and produce_detector else "model_loading_failed",
            "model_loaded": engine is not None, "category_detector_loaded": produce_detector is not None,
            "category_count": len(produce_detector.model.names) if produce_detector else 0,
            "freshness_supported_categories": engine.fruits if engine else [], "errors": model_errors}


@app.post("/camera/preview")
async def camera_preview(file: UploadFile = File(...)):
    contents = await file.read(MAX_BYTES + 1)
    if len(contents) > MAX_BYTES:
        raise HTTPException(413, "Preview frame is larger than 15 MB.")
    image = _decode(contents)

    def detect_preview():
        if produce_detector is None:
            raise HTTPException(503, "Category model is not ready.")
        return {"detections": produce_detector.detect(image), "saved": False}

    return await run_in_threadpool(detect_preview)


@app.post("/camera/analyze")
@app.post("/analyze")
async def camera_analyze(file: UploadFile = File(...)):
    contents = await file.read(MAX_BYTES + 1)
    if len(contents) > MAX_BYTES:
        raise HTTPException(413, "Snapshot is larger than 15 MB.")
    image = _decode(contents)
    result, crops = await run_in_threadpool(analyze_camera_frame, image)
    sample_id = _save_capture(contents, image, result, crops)
    result["sample_id"] = sample_id
    result["source"] = "camera_capture"
    return result


@app.get("/camera/crop/{sample_id}/{object_id}")
def camera_crop(sample_id: str, object_id: int):
    try:
        record = json.loads((RECORDS / f"{sample_id}.json").read_text())
        path = Path(record["prediction"]["objects"][object_id]["crop_path"])
        if not path.is_relative_to(CROPS) or not path.is_file():
            raise FileNotFoundError
        return FileResponse(path, headers={"Cache-Control": "no-store"})
    except (FileNotFoundError, IndexError, KeyError, ValueError) as exc:
        raise HTTPException(404, "Crop not found.") from exc


@app.post("/camera/suspected")
async def save_suspected_region(sample_id: str = Form(...), object_id: int = Form(...), region_index: int = Form(...)):
    try:
        record = json.loads((RECORDS / f"{sample_id}.json").read_text())
        obj = record["prediction"]["objects"][object_id]
        region = obj["suspected_regions"][region_index]
        raw = Image.open(record["raw_image"]).convert("RGB")
        left, top, right, bottom = obj["box"]
        width, height = right - left, bottom - top
        box = (left + int(region["box"][0] * width), top + int(region["box"][1] * height),
               left + int(region["box"][2] * width), top + int(region["box"][3] * height))
        output = SUSPECTED / f"{sample_id}_object{object_id}_region{region_index}.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        raw.crop(box).save(output)
        region["saved_snapshot"] = str(output)
        (RECORDS / f"{sample_id}.json").write_text(json.dumps(record, indent=2))
        return {"saved": True, "path": str(output), "factor": region["factor"]}
    except (FileNotFoundError, IndexError, ValueError) as exc:
        raise HTTPException(404, "That suspected region is no longer available.") from exc


def main():
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
