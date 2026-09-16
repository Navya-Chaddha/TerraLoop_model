"""Fruit and vegetable category/freshness analysis application."""

from setuptools import setup, find_packages

setup(
    name="fruit-freshness",
    version="2.0.0",
    description="Produce detection with supported fruit freshness analysis",
    python_requires=">=3.10",
    packages=find_packages(exclude=("tests",)),
    package_data={"deployment": ["static/*"]},
    install_requires=[
        "torch>=2.2.0",
        "torchvision>=0.17.0",
        "Pillow>=10.0.0",
        "fastapi>=0.110.0",
        "uvicorn[standard]>=0.27.0",
        "python-multipart>=0.0.6",
        "kagglehub>=1.0.2",
        "ultralytics>=8.4.0",
        "numpy>=1.26.0",
        "opencv-contrib-python>=4.8.0",
    ],
    extras_require={"dev": ["pytest>=8.0.0", "httpx>=0.27.0"]},
    entry_points={
        "console_scripts": [
            "fruit-freshness-train=src.train_freshness:main",
            "fruit-category-train=src.train_detector:main",
            "fruit-freshness-serve=deployment.app:main",
        ],
    },
)
