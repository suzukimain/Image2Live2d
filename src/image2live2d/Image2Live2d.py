import os
from PIL import Image
import numpy as np
import cv2

from pytoshop.enums import BlendMode

# layerdivider 内ユーティリティ
from ldivider.ld_convertor import pil2cv
from ldivider.ld_processor import get_seg_base, get_composite_layer, get_normal_layer
from ldivider.ld_utils import save_psd, divide_folder, load_seg_model
from ldivider.ld_segment import get_mask_generator, get_masks

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
    layer_mode: str = "composite",  # "composite" or "normal"
    area_th: int = 2000,            # 小片除去のしきい値（画像に応じて調整）
):
    # 画像読み込み（RGBAにする）
    pil_img = Image.open(image_path).convert("RGBA")
    cv_img = pil2cv(pil_img)
    input_rgba = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGBA)

    # セグメンテーションモデル読み込み（初回は自動DLの実装がある想定）
    load_seg_model(MODEL_DIR)
    mask_gen = get_mask_generator(
        pred_iou_thresh=0.88,
        stability_score_thresh=0.95,
        min_mask_region_area=1000,
        model_dir=MODEL_DIR,
        mode="extension",  
    )
    masks = get_masks(cv_img, mask_gen)

    # マスク群から領域ベースを作成
    df = get_seg_base(input_rgba, masks, area_th)

    if layer_mode == "composite":
        base_layer_list, shadow_layer_list, bright_layer_list, addition_layer_list, subtract_layer_list = get_composite_layer(input_rgba, df)

        psd_path = save_psd(
            input_rgba,
            [base_layer_list, bright_layer_list, shadow_layer_list, subtract_layer_list, addition_layer_list],
            ["base", "screen", "multiply", "subtract", "addition"],
            [BlendMode.normal, BlendMode.screen, BlendMode.multiply, BlendMode.subtract, BlendMode.linear_dodge],
            OUTPUT_DIR,
            layer_mode,
        )
    else:
        base_layer_list, bright_layer_list, shadow_layer_list = get_normal_layer(input_rgba, df)
        psd_path = save_psd(
            input_rgba,
            [base_layer_list, bright_layer_list, shadow_layer_list],
            ["base", "bright", "shadow"],
            [BlendMode.normal, BlendMode.normal, BlendMode.normal],
            OUTPUT_DIR,
            layer_mode,
        )

    # フォルダレイヤー差し込み（input/empty.psd がある場合のみ）
    try:
        psd_path = divide_folder(psd_path, INPUT_DIR, layer_mode)
    except Exception as e:
        print(f"divide_folder をスキップしました（理由: {e}）")

    print(f"PSD saved to: {psd_path}")
    return psd_path

if __name__ == "__main__":
    # 使用例
    segment_to_psd("person.png", layer_mode="composite", area_th=2000)