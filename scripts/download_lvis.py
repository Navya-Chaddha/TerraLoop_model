"""Download the LVIS produce-detection dataset from Kaggle."""

from pathlib import Path

import kagglehub


def main() -> None:
    destination = Path("data/lvis_categories")
    path = kagglehub.dataset_download(
        "henningheyen/lvis-fruits-and-vegetables-dataset",
        output_dir=str(destination),
    )
    print(f"Dataset downloaded to: {path}")


if __name__ == "__main__":
    main()
