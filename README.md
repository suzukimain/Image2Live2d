# Image2Live2d

ステージ1: 1枚のキャラクター画像から、主要パーツをざっくり分割して透過PNGで書き出します。

現段階の出力（雛形）
- 顔ベース（肌）: `face_skin.png`
- 目（左右まとめ）
	- 白目: `eye_white.png`
	- 黒目（虹彩）: `eye_iris.png`
	- まぶた（上/下）: `eyelid_top.png`, `eyelid_bottom.png`
- 口（閉じ）: `mouth_closed.png`
- 髪（前髪・後ろ髪）: `hair_front.png`, `hair_back.png`
- 胴体（首＋体）: `torso.png`
- 腕（左右まとめ）: `arms.png`

注意: 本ステージは幾何ルール＋MediaPipeの簡易推定に基づく雛形であり、画像や絵柄によっては領域推定が荒くなります。必要に応じてルールや閾値を調整してください。

## セットアップ

1) Python 3.10+ 推奨（Windows PowerShell）

2) 依存インストール

```
pip install -r requirements.txt
```

## 使い方（ステージ1）

```
python src/image2live2d/stage1_simple_parts.py <input_image_path> --outdir outputs/stage1
```

例:

```
python src/image2live2d/stage1_simple_parts.py sample.png --outdir outputs/stage1
```

出力先 `outputs/stage1` に各パーツの透過PNGが保存されます。

## 実装メモ
- MediaPipe Face Meshで顔ランドマークを取得（未検出の場合は画像中央の仮領域を使用）
- SelfieSegmentationで人物マスクを取得
- 目・口・髪は顔BBoxからの相対比率で矩形分割（雛形）
- PILでマスクをアルファに適用し、外接矩形でトリムして保存

## 既知の制約 / トラブルシュート
- MediaPipeがGPU不要でも少し重いことがあります。初回実行でダウンロードが発生する場合があります。
- 顔が検出できない場合は仮の顔領域で分割するため精度が下がります。顔が正面に近い画像で試すと安定します。
- 髪や腕の分割は大雑把です。今後のステージでSAM/HFや追加ルールで精密化します。