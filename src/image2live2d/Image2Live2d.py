import os
from PIL import Image
import cv2
import numpy as np

from pytoshop.enums import BlendMode

# layerdivider 内ユーティリティ
from .config import (
    pil2cv,
    get_seg_base,
    get_composite_layer,
    get_normal_layer,
    save_psd,
    divide_folder,
    load_seg_model,
    get_mask_generator,
    get_masks,
)

# 作業ディレクトリ（demo.py と同様の構成）
BASE = os.getcwd()
OUTPUT_DIR = os.path.join(BASE, "output")
INPUT_DIR  = os.path.join(BASE, "input")
MODEL_DIR  = os.path.join(BASE, "segment_model")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def segment_to_psd(
    image_path: str,
    outdir: str | None = None,
    layer_mode: str = "composite",  # "composite" or "normal"
    area_th: int = 2000,            # 小片除去のしきい値（画像に応じて調整）
):
    """
    画像をセグメンテーションし、レイヤー分割したPSDを保存します。

    Parameters
    - image_path: 入力画像のパス
    - outdir: 出力先ディレクトリ。未指定時はカレント配下の "output" を使用
    - layer_mode: "composite" または "normal"
    - area_th: 小片除去のしきい値

    Returns
    - 生成されたPSDファイルのパス
    """
    # 出力先の決定（引数優先）
    out_dir = os.path.abspath(outdir) if outdir else OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    # 画像読み込み（RGBAにする）
    pil_img = Image.open(image_path).convert("RGBA")
    cv_img = pil2cv(pil_img)
    input_rgba = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGBA)

    # SAM への入力は 3ch RGB かつ長辺 1024 を要求される場合があるため正規化する
    if cv_img.ndim == 3 and cv_img.shape[2] == 4:
        sam_img = cv2.cvtColor(cv_img, cv2.COLOR_BGRA2RGB)
    else:
        sam_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

    H0, W0 = sam_img.shape[:2]
    target_long = 1024
    long_side = max(H0, W0)
    resized = False
    if long_side != target_long:
        scale = target_long / float(long_side)
        new_w = max(1, int(round(W0 * scale)))
        new_h = max(1, int(round(H0 * scale)))
        sam_img_resized = cv2.resize(sam_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        resized = True
    else:
        sam_img_resized = sam_img

    # セグメンテーションモデル読み込み（初回は自動DLの実装がある想定）
    load_seg_model(MODEL_DIR)
    mask_gen = get_mask_generator(
        pred_iou_thresh=0.88,
        stability_score_thresh=0.95,
        min_mask_region_area=1000,
        model_path=MODEL_DIR,
        exe_mode="standalone",  
    )
    # SAM 実行（RGB, HxWx3, uint8）
    masks = get_masks(sam_img_resized, mask_gen)

    # マスクを元解像度に合わせる（最近傍）。area も再計算。
    if resized and isinstance(masks, list):
        upscaled_masks = []
        for m in masks:
            seg = m.get("segmentation")
            if seg is None:
                upscaled_masks.append(m)
                continue
            seg_u8 = seg.astype(np.uint8)
            seg_up = cv2.resize(seg_u8, (W0, H0), interpolation=cv2.INTER_NEAREST).astype(bool)
            m2 = dict(m)
            m2["segmentation"] = seg_up
            m2["area"] = int(seg_up.sum())
            upscaled_masks.append(m2)
        masks = upscaled_masks

    # マスク群から領域ベースを作成
    df = get_seg_base(input_rgba, masks, area_th)

    if layer_mode == "composite":
        base_layer_list, shadow_layer_list, bright_layer_list, addition_layer_list, subtract_layer_list = get_composite_layer(input_rgba, df)

        psd_path = save_psd(
            input_rgba,
            [base_layer_list, bright_layer_list, shadow_layer_list, subtract_layer_list, addition_layer_list],
            ["base", "screen", "multiply", "subtract", "addition"],
            [BlendMode.normal, BlendMode.screen, BlendMode.multiply, BlendMode.subtract, BlendMode.linear_dodge],
            out_dir,
            layer_mode,
        )
    else:
        base_layer_list, bright_layer_list, shadow_layer_list = get_normal_layer(input_rgba, df)
        psd_path = save_psd(
            input_rgba,
            [base_layer_list, bright_layer_list, shadow_layer_list],
            ["base", "bright", "shadow"],
            [BlendMode.normal, BlendMode.normal, BlendMode.normal],
            out_dir,
            layer_mode,
        )


    try:
        psd_path = divide_folder(psd_path, INPUT_DIR, layer_mode)
    except Exception as e:
        print(f"divide_folder をスキップしました（理由: {e}）")

    print(f"PSD saved to: {psd_path}")
    return psd_path
