# Fruit and vegetable freshness analyzer

## Team setup

Use Python 3.10 or newer. From a terminal:

```bash
git clone https://github.com/Navya-Chaddha/TerraLoop_model.git
cd TerraLoop_model
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
mkdir -p checkpoints
```

On Windows, activate with `.venv\Scripts\activate` instead. Obtain the trusted team model files separately and place them at:

- `checkpoints/freshness_cnn.pth` — trained fruit/freshness classifier.
- `checkpoints/produce_detector.pt` — trained produce category detector.

Model files and datasets are excluded from Git. A fresh clone does not include either checkpoint; both are required for camera analysis. Do not substitute a generic YOLO checkpoint for the produce-trained detector. The team must supply the model artifacts; no automatic model download is configured.

```bash
python -m uvicorn deployment.app:app --host 127.0.0.1 --port 8000 --reload
```

Open http://127.0.0.1:8000/, allow camera access, and capture a fruit. Visit `/health` to check model readiness and loading errors. Open the server URL rather than the HTML file directly. If port 8000 is occupied, stop the earlier server or use `--port 8001` and open that port.

## File structure

```text
deployment/
  app.py                  # FastAPI routes, inference orchestration, capture storage
  static/                 # Camera UI: index.html, styles.css, app.js
  Dockerfile              # Container build definition
src/
  freshness_classifier.py # Classifier architecture, labels, transforms
  produce_detector.py     # YOLO inference and category aliases
  visible_spoilage.py      # RGB, entropy, texture indicators
  train_freshness.py       # Freshness training entry point
  train_detector.py        # Produce detector training entry point
scripts/download_lvis.py   # Category dataset download
tests/                    # Python regression tests
setup.py                  # Package, dependencies, command entry points
requirements.txt          # Dependency list for container installation
CONTRIBUTING.md            # Branches, pull requests, verification
data/                     # Local only: research data and camera captures
checkpoints/              # Local only: model weights and training output
```

Run `python -m pytest -q` before submitting changes. See [CONTRIBUTING.md](CONTRIBUTING.md) for collaboration and camera testing. Keep runtime artifacts outside any manual GitHub upload: upload the tracked project files, not the entire local folder.

## Freshness training

This project trains from the supplied research dataset in `data/research/original`. It does not use generated image archives as validation data.

The source folder names define the labels: `FreshApple` / `RottenApple`, and the equivalent folders for banana, grape, guava, jujube, orange, pomegranate, and strawberry. MobileNetV3-Large provides shared image features for two learned heads: fruit type and freshness.

```bash
python -m src.train_freshness --epochs 20
uvicorn deployment.app:app --reload
```

Open http://127.0.0.1:8000 and choose **Open camera**. The interface reports the detected produce category, detector confidence, supported freshness results, and explainable visible-surface indicators. It intentionally has no file-upload control.

Every camera snapshot is retained locally in `data/collected_uploads/raw`, with crops in `data/collected_uploads/crops` and a JSON record containing its SHA-256 fingerprint, timestamp and model output. Model predictions are never treated as ground-truth labels; the existing labelling/training pipeline can use human-verified records later.

The training script reads the supplied original-image dataset and can also read human-labelled captures from `data/collected_uploads/labelled`. This does not establish which sources were used for a particular checkpoint; retain the training configuration with each shared model. Validation accuracy is metadata for that split, not a guarantee of accuracy on unrelated real-world photos.

## Category expansion

Category expansion uses [Henning Heyen's Kaggle LVIS fruits-and-vegetables dataset](https://www.kaggle.com/datasets/henningheyen/lvis-fruits-and-vegetables-dataset). It contains 6,721 training, 1,500 validation, and 180 manually labelled test images with YOLO bounding boxes for 63 produce categories. It does **not** contain freshness labels.

```bash
python scripts/download_lvis.py
python -m src.train_detector --epochs 30
```

`src.train_detector` generates `data.resolved.yaml` so the Kaggle release's incorrect `val: images/test` entry does not contaminate validation. The active `checkpoints/produce_detector.pt` is the published YOLOv8m baseline trained on this exact dataset. Its local held-out evaluation on the 180-image test split was: precision 0.597, recall 0.226, mAP50 0.264, and mAP50-95 0.166. Those metrics are why the interface exposes confidence and manual-review warnings.

When `checkpoints/produce_detector.pt` exists, the app uses YOLO to localize one or more produce items and analyzes each detected crop independently. The learned freshness head is applied only to apple, banana, grape, guava, jujube, orange, pomegranate, and strawberry. All additional categories are explicitly reported as `visible_heuristic_only`; unsupported categories do not receive made-up classifier scores.

## Camera workflow

The browser camera is requested with `getUserMedia` and no server-side camera access. Start the camera, keep one fruit in view, and use **Capture snapshot** when the preview detects it. The snapshot is saved under one sample ID with its detected crops and measurements. The interface can save individual suspected dark/brown textured regions for later review. Preview frames are not saved. Stop the camera when finished; the page also releases camera tracks when it closes.

## Visible surface indicators

Visible-spoilage indicators use normalized RGB-channel Shannon entropy, dark and brown/discoloration color-band ratios, rough desaturated patch coverage, and local texture variation. The detector crop reduces background interference. These parameters are saved with every camera snapshot so future human-verified data can be audited and used in retraining.

These indicators can flag visible deterioration patterns, but they are not a trained mold detector. They cannot prove mold, internal spoilage, smell, softness, or food safety. A single photo also cannot measure change over time; that requires labelled photo sequences of the same item.
