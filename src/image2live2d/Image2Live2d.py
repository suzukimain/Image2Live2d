from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image


@dataclass
class SamMask:
	id: int
	bbox: Tuple[int, int, int, int]  # x, y, w, h
	area: int
	stability_score: float
	category: Optional[str]
	mask: np.ndarray  # HxW uint8 {0,255}


class Split_image:
	"""Split a character image into masks using Segment Anything (SAM).

	Usage:
		Split_image.run(img="input.png", outdir="outputs/masks", sam_model="vit_h", sam_checkpoint="path/to/sam_vit_h.pth")
	"""

	@staticmethod
	def _load_image(img: Any) -> Image.Image:
		"""Load image from PIL, OpenCV ndarray (BGR/RGB), or file path."""
		if isinstance(img, Image.Image):
			return img.convert("RGB")
		if isinstance(img, np.ndarray):
			arr: np.ndarray = img
			if arr.ndim == 2:
				arr = np.stack([arr] * 3, axis=-1)
			if arr.ndim == 3 and arr.shape[-1] == 3:
				# Assume BGR -> RGB
				arr = np.flip(arr, axis=2)
			arr = arr.astype(np.uint8, copy=False)
			return Image.fromarray(arr, mode="RGB")
		p = Path(img)
		if not p.exists():
			raise FileNotFoundError(f"Input not found: {p}")
		with Image.open(p) as im:
			return im.convert("RGB")

	@staticmethod
	def _ensure_outdir(outdir: str | Path) -> Path:
		od = Path(outdir)
		od.mkdir(parents=True, exist_ok=True)
		return od

	@staticmethod
	def _to_uint8_mask(seg: np.ndarray) -> np.ndarray:
		if seg.dtype != np.uint8:
			seg = seg.astype(np.uint8)
		# Convert boolean -> 0/255 if needed
		if seg.max() <= 1:
			seg = seg * 255
		return seg

	@staticmethod
	def _run_sam(image: Image.Image, model_name: str, checkpoint_path: Optional[str]) -> List[SamMask]:
		try:
			import os
			import torch
			from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
		except Exception as e:
			raise RuntimeError("segment-anything/torch are required for SAM segmentation") from e

		np_img = np.array(image.convert("RGB"))
		device = "cuda" if hasattr(torch, "cuda") and torch.cuda.is_available() else "cpu"

		ckpt = checkpoint_path or os.getenv("SAM_CHECKPOINT")
		if not ckpt or not Path(ckpt).exists():
			raise FileNotFoundError("SAM checkpoint not found. Provide sam_checkpoint or set SAM_CHECKPOINT env var.")

		sam = sam_model_registry[model_name](checkpoint=str(ckpt))
		sam.to(device=device)
		generator = SamAutomaticMaskGenerator(sam)
		raw = generator.generate(np_img)

		masks: List[SamMask] = []
		for i, m in enumerate(raw):
			seg = m.get("segmentation")
			if seg is None:
				continue
			seg = (seg.astype(np.uint8)) * 255
			x, y, w, h = m.get("bbox", (0, 0, seg.shape[1], seg.shape[0]))
			masks.append(
				SamMask(
					id=i,
					bbox=(int(x), int(y), int(w), int(h)),
					area=int(m.get("area", int((seg > 0).sum()))),
					stability_score=float(m.get("stability_score", 0.0)),
					category=None,
					mask=seg,
				)
			)
		return masks

	@staticmethod
	def _guess_category(mask: SamMask, image_size: Tuple[int, int]) -> str:
		# Very simple heuristic placeholders; refine later with CLIP/semantic models
		h, w = image_size[1], image_size[0]
		x, y, bw, bh = mask.bbox
		area_ratio = mask.area / float(w * h + 1e-6)
		# Heuristic: large masks covering bottom -> background/clothes/torso
		if area_ratio > 0.6:
			return "background"
		# Upper region likely hair or head
		if y < h * 0.2:
			return "hair"
		# Center-top: face region
		if y < h * 0.4 and bh < h * 0.6:
			return "face"
		# Mid-lower: clothes/torso
		if y + bh > h * 0.5:
			return "clothes"
		return "unknown"

	@staticmethod
	def _save_masks(masks: List[SamMask], outdir: Path) -> Dict[str, str]:
		saved: Dict[str, str] = {}
		for m in masks:
			name = f"mask_{m.id:03d}"
			if m.category:
				name = f"{m.category}_{m.id:03d}"
			fp = outdir / f"{name}.png"
			Image.fromarray(m.mask, mode="L").save(fp)
			saved[name] = str(fp)
		return saved

	@staticmethod
	def run(
		img: Union[str, Image.Image, np.ndarray],
		outdir: str | Path = "outputs/masks",
		sam_model: str = "vit_h",
		sam_checkpoint: Optional[str] = None,
		categorize: bool = True,
		min_area_ratio: float = 0.001,
	) -> Dict[str, str]:
		"""Run SAM and save mask PNGs with English names.

		Returns a dict of {basename: filepath}.
		"""
		image = Split_image._load_image(img)
		out = Split_image._ensure_outdir(outdir)

		masks = Split_image._run_sam(image, sam_model, sam_checkpoint)

		# Filter small masks
		W, H = image.width, image.height
		min_area = int(W * H * min_area_ratio)
		masks = [m for m in masks if m.area >= min_area]

		# Optional: heuristic category names (hair/face/clothes/background)
		if categorize:
			for m in masks:
				m.category = Split_image._guess_category(m, (W, H))

		return Split_image._save_masks(masks, out)

