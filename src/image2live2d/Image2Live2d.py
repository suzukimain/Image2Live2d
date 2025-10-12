from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
from PIL import Image


@dataclass
class SplitConfig:
    model_id: str = "nvidia/segformer-b0-finetuned-ade-512-512"
    target_labels: Optional[Dict[str, List[str]]] = None


class SplitImage:
    def __init__(self, model_id: str | None = None, target_labels: Optional[Dict[str, List[str]]] = None) -> None:
        self.config = SplitConfig()
        if model_id:
            self.config.model_id = model_id
        # 抽出したいパーツ名 -> モデル側のラベル文字列のリスト（必ずdictに）
        self.target_labels: Dict[str, List[str]] = target_labels or {
            "hair": ["hair"],
            "face": ["face", "person head", "head"],
            "clothes": ["clothes", "cloth", "upperclothes", "coat", "jacket", "shirt", "dress", "skirt", "pants", "trousers"],
            "background": ["background", "wall", "sky", "floor", "ceiling", "building"],
        }

    def _load_pipeline(self):
        from transformers import pipeline
        return pipeline("semantic-segmentation", model=self.config.model_id)

    def _labels_to_id_map(self, pipe) -> Dict[str, int]:
        # pipelineのconfigまたはmodel.config.id2labelからid->labelを取得
        id2label = None
        if hasattr(pipe.model, "config") and hasattr(pipe.model.config, "id2label"):
            id2label = pipe.model.config.id2label
        if id2label is None and hasattr(pipe, "id2label"):
            id2label = pipe.id2label
        if id2label is None:
            # フォールバック: 既知のADE20K 150クラスを仮想（ここでは空）
            id2label = {}
        # label(lower) -> id
        label2id = {str(v).lower(): int(k) for k, v in id2label.items()}
        return label2id

    def segment(self, img: Image.Image) -> Dict[str, np.ndarray]:
        pipe = self._load_pipeline()
        out = pipe(img)
        # transformersのsemantic-segmentationは、各ラベルごとにmaskを返す or 全体マップ返す等、実装差あり
        # ここでは最も一般的な"labels"+"masks"形式を想定し、fall-backでscore順の辞書を構築
        label2id = self._labels_to_id_map(pipe)

        # 統一マスク辞書（0/255のuint8）
        masks: Dict[str, Optional[np.ndarray]] = {k: None for k in self.target_labels.keys()}

        def to_binary_mask(m) -> np.ndarray:
            if isinstance(m, Image.Image):
                arr = np.array(m.convert("L"))
                return (arr > 127).astype(np.uint8) * 255
            if isinstance(m, np.ndarray):
                if m.dtype != np.uint8:
                    m = m.astype(np.uint8)
                # 値が0/1なら255スケールに
                if m.max() == 1:
                    m = m * 255
                return (m > 0).astype(np.uint8) * 255
            if isinstance(m, str):
                # base64 PNG文字列'
                import base64, io
                try:
                    b = m.split(",")[-1]
                    data = base64.b64decode(b)
                    with Image.open(io.BytesIO(data)) as im:
                        arr = np.array(im.convert("L"))
                        return (arr > 127).astype(np.uint8) * 255
                except Exception:
                    return None  # type: ignore
            # base64等の場合は未対応
            return None  # type: ignore

        # パイプライン出力ケース1: list of dicts with 'label' and 'mask'
        if isinstance(out, list) and len(out) > 0 and isinstance(out[0], dict) and "label" in out[0]:
            for item in out:
                label = str(item.get("label", "")).lower()
                m = to_binary_mask(item.get("mask"))
                if m is None:
                    continue
                for key, aliases in self.target_labels.items():
                    if any(label == a for a in aliases):
                        masks[key] = m if masks[key] is None else np.maximum(masks[key], m)

        # パイプライン出力ケース2: dict with 'segmentation'
        elif isinstance(out, dict) and "segmentation" in out:
            seg = out["segmentation"]  # HxW int class id
            if isinstance(seg, Image.Image):
                seg = np.array(seg)
            seg = seg.astype(np.int32)
            for key, aliases in self.target_labels.items():
                wanted_ids = [label2id[a] for a in aliases if a in label2id]
                if not wanted_ids:
                    continue
                m = np.isin(seg, wanted_ids).astype(np.uint8) * 255
                masks[key] = m if masks[key] is None else np.maximum(masks[key], m)

        # Noneを空マスクに
        W, H = img.size
        for k, v in list(masks.items()):
            if v is None:
                masks[k] = np.zeros((H, W), dtype=np.uint8)

        return masks

    def save_masks(self, masks: Dict[str, np.ndarray], out_dir: str) -> Dict[str, str]:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        saved: Dict[str, str] = {}
        for key, m in masks.items():
            im = Image.fromarray(m.astype(np.uint8), mode="L")
            fp = out_path / f"{key}.png"
            im.save(fp)
            saved[key] = str(fp)
        return saved

    @classmethod
    def run(
        cls,
        img: Union[str, Path, Image.Image],
        outdir: Union[str, Path] = "outputs/masks",
        model_id: Optional[str] = None,
        target_labels: Optional[Dict[str, List[str]]] = None,
    ) -> Dict[str, str]:
        """One-shot API: load image (if path), segment, and save masks.

        Returns mapping: part name -> saved file path.
        """
        if isinstance(img, (str, Path)):
            with Image.open(str(img)) as im:
                pil_img = im.convert("RGB")
        elif isinstance(img, Image.Image):
            pil_img = img.convert("RGB")
        else:
            raise TypeError("img must be a file path or PIL.Image.Image")

        inst = cls(model_id=model_id, target_labels=target_labels)
        masks = inst.segment(pil_img)
        return inst.save_masks(masks, out_dir=str(outdir))
