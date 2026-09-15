# 自前データでの推論に必要な入力

この文書では、公開モデル `meteor_v157c3Z.onnx` を使って、自前のカメラデータで推論するために準備する入力を説明します。

基本的な入力処理は [../hf/onnx_smoke_test.py](../hf/onnx_smoke_test.py) を参照してください。

## 必須データ

推論用データは、シーンごとに次の構成にします。

```text
my_data/
└── scene_name/
    ├── manifest.json
    ├── ego_motion.npz
    └── img/
        ├── 0000_CAM_FRONT_WIDE.jpg
        ├── 0000_CAM_FRONT_LEFT.jpg
        ├── 0000_CAM_FRONT_RIGHT.jpg
        ├── 0000_CAM_BACK_WIDE.jpg
        ├── 0000_CAM_BACK_LEFT.jpg
        ├── 0000_CAM_BACK_RIGHT.jpg
        ├── 0000_CAM_FRONT_NARROW.jpg
        └── 0000_CAM_BACK_NARROW.jpg
```

### 1. 8台分の同期カメラ画像

公開モデルは、次の8カメラをこの順序で使用します。

```text
CAM_FRONT_WIDE
CAM_FRONT_LEFT
CAM_FRONT_RIGHT
CAM_BACK_WIDE
CAM_BACK_LEFT
CAM_BACK_RIGHT
CAM_FRONT_NARROW
CAM_BACK_NARROW
```

画像の条件は以下です。

- 1フレームにつき8枚
- 8枚は同一時刻に撮影されていること
- `768 x 432` に変換できること
- 画像データはRGB、`uint8`、画素値 `0..255`
- 元画像が別解像度の場合、参照実装は `768 x 432` にリサイズする

モデルに渡すテンソル形状は `[1, 8, 3, 432, 768]` です。

### 2. カメラキャリブレーション

`manifest.json` の `cams` に、各カメラの内部パラメータと外部パラメータを記載します。

```json
{
  "cams": {
    "CAM_FRONT_WIDE": {
      "K": [
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
      ],
      "T_ego_cam": [
        [r00, r01, r02, tx],
        [r10, r11, r12, ty],
        [r20, r21, r22, tz],
        [0, 0, 0, 1]
      ]
    }
  }
}
```

- `K`: `3 x 3` のカメラ内部パラメータ
- `K` は画像を `768 x 432` にした後の値
- `T_ego_cam`: `4 x 4` の外部パラメータ
- 参照実装は `T_ego_cam` の逆行列を作り、モデル入力 `T_cam_ego` として渡す
- 座標系、軸方向、単位は学習時のデータセットと合わせる

カメラごとに異なるキャリブレーション値を用意してください。

### 3. フレーム一覧

`manifest.json` の `frames` に、フレーム番号と8枚の画像パスを記載します。

```json
{
  "scene": "scene_name",
  "img_hw": [432, 768],
  "cams": {
    "CAM_FRONT_WIDE": {
      "K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
      "T_ego_cam": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    }
  },
  "frames": [
    {
      "frame": 0,
      "imgs": {
        "CAM_FRONT_WIDE": "img/0000_CAM_FRONT_WIDE.jpg",
        "CAM_FRONT_LEFT": "img/0000_CAM_FRONT_LEFT.jpg",
        "CAM_FRONT_RIGHT": "img/0000_CAM_FRONT_RIGHT.jpg",
        "CAM_BACK_WIDE": "img/0000_CAM_BACK_WIDE.jpg",
        "CAM_BACK_LEFT": "img/0000_CAM_BACK_LEFT.jpg",
        "CAM_BACK_RIGHT": "img/0000_CAM_BACK_RIGHT.jpg",
        "CAM_FRONT_NARROW": "img/0000_CAM_FRONT_NARROW.jpg",
        "CAM_BACK_NARROW": "img/0000_CAM_BACK_NARROW.jpg"
      }
    }
  ]
}
```

画像パスは `scene_name/` からの相対パスです。`frames` の順序と画像の時刻順を一致させてください。

### 4. 車速

`ego_motion.npz` に、フレームごとの車速配列 `v0` を保存します。

```python
import numpy as np

np.savez(
    "ego_motion.npz",
    v0=np.asarray(speeds_mps, dtype=np.float32),
)
```

- 配列形状: `[フレーム数]`
- 型: `float32`
- 単位: `m/s`
- `v0[i]` は `manifest.json` の `frames[i]` に対応

## 不要なデータ

カメラ専用の公開ONNXモデルで推論するだけなら、次のデータは不要です。

- Ground Truth
- 3D bounding box
- セグメンテーション正解ラベル
- 深度正解データ
- Occupancy正解データ
- HD map
- LiDARデータ

## 5カメラで推論する場合

公開されている `meteor_v157c3Z.onnx` は入力形状が8カメラ固定なので、5枚の画像をそのまま渡すことはできません。リポジトリのエクスポータはカメラ数を変更できるため、`meteor_v157.pt` から5カメラ用ONNXを作成します。

```bash
python3 deploy/export_onnx.py \
  --ckpt models/meteor_v157.pt \
  --model v52 \
  --n-cams 5 \
  --uint8-in \
  --argmax-out \
  --lane-logits \
  --no-hist \
  --depth-mean \
  --out out/meteor_v157_5cam.onnx
```

このコマンドにはPyTorch、ONNX、ONNX Runtimeなど、再エクスポート用の依存パッケージが必要です。`--n-cams 5` は、標準カメラ順序の先頭5スロットを入力にする設定です。

```text
CAM_FRONT_WIDE
CAM_FRONT_LEFT
CAM_FRONT_RIGHT
CAM_BACK_WIDE
CAM_BACK_LEFT
```

手持ちの5台がこの構成と異なる場合は、次のどちらかにします。

- 画像を上記5スロットの意味に合わせて用意し、各画像に対応する実際の `K` と `T_ego_cam` を設定する
- モデルを自前のカメラ構成で再学習し、その構成で再エクスポートする

8カメラ用の学習済み重みをそのまま5カメラ化すれば、計算グラフは動かせます。ただし、視野の欠落、カメラ配置の違い、自車データと学習データのドメイン差によって、検出・BEV・経路推定の精度が低下する可能性があります。実運用の精度が必要な場合は、5カメラ構成のデータでファインチューニングまたは再学習してください。

5カメラ用ONNXを作成した後は、通常のONNX確認コマンドで実行できます。

```bash
python3 hf/onnx_smoke_test.py \
  --onnx out/meteor_v157_5cam.onnx \
  --root my_data/scene_name \
  --frame 0
```

この確認スクリプトは現在8カメラ前提のため、5カメラ用ONNXを使う場合は、`hf/onnx_smoke_test.py` のカメラ配列と `load_frame()` の入力作成を5カメラ用に変更する必要があります。画像・`K`・`T_cam_ego` の順序を、エクスポート時の5スロットと必ず一致させてください。

TensorRTを使う場合は、5カメラ用ONNXから別のエンジンをビルドします。8カメラ用エンジンを5カメラ入力で実行することはできません。

## LiDAR入力版を使う場合

LiDAR入力版エンジンでは、追加で各フレームの次の入力が必要です。

```text
lidar_bev:  float32 [1, 4, 400, 250]
lidar_flag: float32 [1]
```

通常の `meteor_v157c3Z.onnx` はカメラ専用なので、LiDARデータは読み込まれません。

## カメラ台数について

公開モデルの `meteor_v157c3Z.onnx` 自体は8カメラ構成で固定されています。5カメラや7カメラで使う場合は、`deploy/export_onnx.py` の `--n-cams` で、そのカメラ数のONNXを別途作成してください。カメラ数を減らしただけでは学習時と異なる入力になるため、精度評価と必要に応じたファインチューニングを行ってください。

モデルの入力仕様は [../models/meteor_v157.param.yaml](../models/meteor_v157.param.yaml) にあります。

## t4datasetからの変換

t4dataset形式の元データを持っている場合は、次の変換ツールで推論用形式を作成できます。

```bash
python3 bevlane/t4_to_demo.py \
  --scene /path/to/t4_scene \
  --out out/t4demo
```

このツールは、画像を `768 x 432` に変換し、`manifest.json`、カメラキャリブレーション、`ego_motion.npz` を生成します。詳細は [../bevlane/t4_to_demo.py](../bevlane/t4_to_demo.py) を参照してください。

## 簡易チェック

作成したデータは、まず1フレームだけONNXで確認できます。

```bash
python3 hf/onnx_smoke_test.py \
  --onnx models/meteor_v157c3Z.onnx \
  --root my_data/scene_name \
  --frame 0
```

正常に実行できると、最後に `SMOKE PASS` が表示されます。

なお、Ground Truthは推論には不要です。推論結果を動画として描画する場合も、まずはカメラ画像、キャリブレーション、車速だけで確認できます。
