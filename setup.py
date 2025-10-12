from pathlib import Path
from setuptools import setup, find_packages

deps=[
    "numpy",
    "Pillow",
    "opencv-python",
    "transformers",
    "huggingface-hub",
    "torch",
    "opencv-python",
    "pandas",
    "scikit-learn",
    "scikit-image",
    "Pillow",
    "tqdm",
    "pytoshop",
    "onnx",
    "onnxruntime-gpu",
    "huggingface_hub",
    "segment_anything",
    "requests",
    "numpy",
    "einops",
    "torch",
]


README = Path(__file__).parent / "README.md"
long_description = README.read_text(encoding="utf-8") if README.exists() else "Image2Live2d"

setup(
    name="Image2Live2d",
    version="0.0.1",
    description="Generate resources for Live2D: simple parts split and HF-based semantic masks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="suzukimain",
    python_requires=">=3.10",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=list(deps),  # use requirements.txt for runtime deps
    extras_require={
        "mediapipe": ["mediapipe>=0.10.14"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
