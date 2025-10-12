from pathlib import Path
from setuptools import setup, find_packages


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
    install_requires=[],  # use requirements.txt for runtime deps
    extras_require={
        "mediapipe": ["mediapipe>=0.10.14"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
