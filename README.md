# Image2Live2d

Stage 1 focuses on quickly exporting rough parts as transparent PNGs from a single character image.

Current outputs (template):
- Face base (skin): `face_skin.png`
- Eyes (combined L/R):
  - Sclera: `eye_white.png`
  - Iris: `eye_iris.png`
  - Eyelids (top/bottom): `eyelid_top.png`, `eyelid_bottom.png`
- Mouth (closed): `mouth_closed.png`
- Hair (front/back): `hair_front.png`, `hair_back.png`
- Torso (neck + body): `torso.png`
- Arms (combined L/R): `arms.png`

Note: This stage is a scaffold using simple geometry rules plus MediaPipe (optional). Segmentation can be coarse depending on the style/image. Tune thresholds or rules as needed.

## Setup

1) Python 3.10+ is recommended (Windows PowerShell)

2) Install dependencies

```
pip install -r requirements.txt
```

Proto-buf conflict friendly: MediaPipe is optional and commented out in `requirements.txt` to avoid protobuf conflicts. Install it only if you need face landmarks/selfie segmentation.

## How to run (Windows PowerShell)

Create and activate a virtual environment, install dependencies, then run either the Stage 1 script or the Transformers-based API.

1) Create venv and install deps

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

2-a) Run Stage 1 (rough parts PNG export)

```powershell
python src/image2live2d/stage1_simple_parts.py .\sample.png --outdir .\outputs\stage1
```

2-b) Run Transformers-based semantic masks (hair/face/clothes/background)

```powershell
python -c "from image2live2d import SplitImage; SplitImage.run('input.png', outdir='outputs\\masks')"
```

Optional: specify a model id (default: nvidia/segformer-b0-finetuned-ade-512-512)

```powershell
python -c "from image2live2d import SplitImage; SplitImage.run('input.png', outdir='outputs\\masks', model_id='nvidia/segformer-b0-finetuned-ade-512-512')"
```

## Usage (Stage 1 script)

```
python src/image2live2d/stage1_simple_parts.py <input_image_path> --outdir outputs/stage1
```

Example:

```
python src/image2live2d/stage1_simple_parts.py sample.png --outdir outputs/stage1
```

The directory `outputs/stage1` will contain each part as a trimmed transparent PNG.

## SplitImage quick API (Transformers)
You can also generate semantic masks (hair/face/clothes/background) via Hugging Face Transformers:

```python
from image2live2d import SplitImage
SplitImage.run('input.png', outdir='outputs/masks')
```

This will save:
- `outputs/masks/hair.png`
- `outputs/masks/face.png`
- `outputs/masks/clothes.png`
- `outputs/masks/background.png`

## Implementation notes
- Optional: MediaPipe Face Mesh for face landmarks (fallback to a heuristic box when missing)
- Optional: Selfie Segmentation for person mask
- Eyes/Mouth/Hair are approximated by ratios within the face bounding box (template)
- PIL applies masks to alpha and crops to the bounding box for each part

## Known limitations / Troubleshooting
- MediaPipe can be relatively heavy on first run due to model downloads.
- If face detection fails, heuristic regions are used and accuracy may drop; try images closer to frontal faces.
- Hair and arms are coarse in this stage; later stages can refine using SAM/HF and custom rules.