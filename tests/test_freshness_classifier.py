from src.freshness_classifier import label_from_folder
from src.visible_spoilage import visible_spoilage_indicators
from src.produce_detector import canonical_category
from PIL import Image
import numpy as np


def test_research_dataset_folder_labels():
    assert label_from_folder("FreshApple") == ("apple", "fresh")
    assert label_from_folder("RottenOrange") == ("orange", "stale")
    assert label_from_folder("FreshStrawberry") == ("strawberry", "fresh")
    assert label_from_folder("RottenPomegranate") == ("pomegranate", "stale")


def test_legacy_plural_folder_labels_remain_supported():
    assert label_from_folder("rottenbananas") == ("banana", "stale")


def test_visible_spoilage_indicators_are_bounded_and_explained():
    result = visible_spoilage_indicators(Image.new("RGB", (32, 32), "white"))
    assert 0 <= result["visible_spoilage_score"] <= 100
    assert 0 <= result["color_entropy"] <= 1
    assert result["visible_spoilage_risk"] in {"low", "medium", "high"}


def test_healthy_red_is_not_mistaken_for_brown_discoloration():
    result = visible_spoilage_indicators(Image.new("RGB", (32, 32), (210, 30, 25)))
    assert result["brown_pixel_ratio"] == 0


def test_brown_color_band_is_measured():
    result = visible_spoilage_indicators(Image.new("RGB", (32, 32), (120, 70, 30)))
    assert result["brown_pixel_ratio"] == 1
    assert "not proof" in result["heuristic_notice"]


def test_lvis_category_aliases_are_normalized():
    assert canonical_category("orange/orange fruit") == "orange"
    assert canonical_category("Strawberry") == "strawberry"


def test_camera_frame_analysis_keeps_unsupported_freshness_empty(monkeypatch):
    import deployment.app as app_module

    class Detector:
        def detect(self, image):
            return [{"category": "tomato", "confidence": 91.0, "box": [0, 0, image.width, image.height]}]

    class Engine:
        fruits = ["apple"]

        def predict(self, image):
            raise AssertionError("unsupported category must not call the freshness model")

    # Replace the active globals only for this isolated assertion.
    monkeypatch.setattr(app_module, "produce_detector", Detector())
    monkeypatch.setattr(app_module, "engine", Engine())
    result, crops = app_module.analyze_camera_frame(Image.new("RGB", (100, 100), (200, 30, 25)))
    assert len(crops) == 1
    assert result["objects"][0]["freshness"] is None
    assert result["objects"][0]["analysis_mode"] == "visible_heuristic_only"


def test_suspected_regions_are_bounded():
    import deployment.app as app_module
    image = Image.fromarray(np.full((256, 256, 3), (180, 40, 20), dtype=np.uint8))
    regions = app_module._suspected_regions(image)
    assert all(0 <= coordinate <= 1 for region in regions for coordinate in region["box"])
