# METEOR 推論時の中間生成物と認識クラス

## 1. 中間生成物のDump候補

`deploy/runtime.py` の推論出力には、以下のテンソルがあります。

| 出力名 | 内容 | 用途 |
|---|---|---|
| `lane` | BEVセマンティック認識 | 道路、車線、停止線などの認識 |
| `lane_logit` | BEVセグメンテーションの生Logit | 各クラスの確信度確認 |
| `depth` | カメラごとの深度ビン | カメラ画像上の距離推定 |
| `depth_mean` | メートル単位の期待深度 | 物体までの距離確認 |
| `seg2d` | カメラごとの2Dセマンティック認識 | 画像内の道路・車両・歩行者など |
| `hm` | BEV 3D物体検出ヒートマップ | 物体中心の候補位置 |
| `reg` | 3D物体の回帰値 | 位置、サイズ、Yawなど |
| `hm2d_s0..s2` | 2D物体検出ヒートマップ | 画像内の物体候補 |
| `reg2d_s0..s2` | 2D物体検出回帰値 | 2D bboxの位置・サイズ |
| `ego` | 自車の軌道候補3本 | 自車の未来6点、操舵、加減速など |
| `traj` | 検出物体の未来軌道 | 各物体の未来位置 |
| `stationary` | 停止物体判定Logit | 停車・走行の判定 |
| `occ` | 3D占有グリッド | 空間内の物体・道路・建物など |
| `risk` | BEVリスクマップ | 周囲の危険度 |
| `tl` | 交通信号状態 | 赤・黄・青・信号なし |
| `flow` | Occupancy等のフロー | 物体・空間の動き |
| `lg_pts` | ベクトル車線グラフの点列 | 車線形状 |
| `lg_meta` | 車線グラフのメタ情報 | 存在確率・クラス |
| `lg_adj` | 車線グラフの隣接関係 | 車線同士の接続 |
| `unk` | Unknown物体検出 | 未知障害物候補 |
| `raw_bev` | 時系列融合用のBEV特徴 | 次フレームの履歴入力 |

## 2. 現在Dump対象から除外されている出力

`deploy/orin_realtime.py` では、現在以下の出力を `skip_outputs` に指定しています。

```python
skip_outputs = (
    "flow",
    "unk",
    "pl",
    "tl",
    "lg_pts",
    "lg_meta",
    "lg_adj",
)
```

そのため、以下の出力は現状CPU側に転送されず、Dumpするには設定変更が必要です。

```text
flow
unk
tl
lg_pts
lg_meta
lg_adj
```

`raw_bev` は通常のCPU出力ではなく、時系列推論のためGPU上のリングバッファで保持されています。

## 3. 現在の動画で確認できるもの

現在の `exec_dmp_data.sh` は以下の設定です。

```bash
METEOR_TH2D=0.30
METEOR_SEG2D_OVERLAY=0
METEOR_OCC_PANEL=0
METEOR_2D_HIDE=7
```

この設定では、主に以下が動画に表示されます。

- 各カメラ画像
- 3D物体検出ボックス
- 2D物体検出ボックス
- 深度画像
- BEVレーン認識
- 自車軌道
- 検出物体の未来軌道
- BEVリスクマップ

以下は非表示です。

```text
2Dセマンティックセグメンテーションの色付き重畳
Occupancyパネル
クラスID 7 の2D検出表示
```

画像上で認識結果を確認したい場合は、次の設定が有効です。

```bash
METEOR_SEG2D_OVERLAY=1
METEOR_OCC_PANEL=1
```

## 4. 認識クラス

### 4.1 BEVレーン認識: 9クラス

定義元:

```text
bevlane/train.py
```

| ID | クラス名 |
|---:|---|
| 0 | unlabeled |
| 1 | road |
| 2 | sidewalk |
| 3 | crosswalk |
| 4 | laneline |
| 5 | stopline |
| 6 | road_edge |
| 7 | marking |
| 8 | parking |

### 4.2 2Dセマンティック認識: 21クラス

定義元:

```text
comlops-21cls-autolabel-2504.csv
```

| ID | クラス名 |
|---:|---|
| 0 | background |
| 1 | misc obstacle |
| 2 | car |
| 3 | truck |
| 4 | bus |
| 5 | motorcycle |
| 6 | bicycle |
| 7 | pedestrian |
| 8 | marking |
| 9 | traffic light |
| 10 | traffic sign |
| 11 | road |
| 12 | sidewalk |
| 13 | lane marking |
| 14 | crosswalk |
| 15 | unused |
| 16 | wall/fence |
| 17 | building |
| 18 | vegetation/terrain |
| 19 | sky |
| 20 | pole |

### 4.3 2D物体検出: 10クラス

定義元:

```text
bevlane/extract_bbox2d.py
deploy/viz_np.py
```

| ID | 略称 | クラス名 |
|---:|---|---|
| 0 | obs | unknown / obstacle |
| 1 | car | car |
| 2 | trk | truck |
| 3 | bus | bus |
| 4 | bcy | bicycle |
| 5 | mcy | motorcycle |
| 6 | ped | pedestrian |
| 7 | pnt | point obstacle |
| 8 | tl | traffic light |
| 9 | ts | traffic sign |

### 4.4 BEV 3D物体検出: 2クラス

現在の3D検出ヘッドは、以下の2クラスです。

| ID | クラス名 |
|---:|---|
| 0 | vehicle |
| 1 | vru |

`vru` は、主に以下のような交通参加者をまとめたクラスです。

```text
pedestrian
bicycle
motorcycle
その他のVRU
```

### 4.5 Occupancy: 10クラス

定義元:

```text
bevlane/extract_occ.py
```

| ID | クラス名 |
|---:|---|
| 0 | free |
| 1 | obstacle |
| 2 | vehicle |
| 3 | 2wheel |
| 4 | pedestrian |
| 5 | road |
| 6 | sidewalk |
| 7 | vegetation |
| 8 | building |
| 9 | pole/sign |

## 5. Dumpの優先順位

認識器が何を見ているのかを確認する目的では、次の順番でDumpするのが有効です。

### 1. `seg2d`

各カメラ画像上で、認識器が以下をどう分類しているか確認できます。

```text
道路
車両
歩行者
信号
標識
車線
建物
空
植生
電柱
```

### 2. `depth_mean`

各画素の距離推定を確認できます。

特に以下の確認に有効です。

```text
物体までの距離が正しいか
遠方物体の距離が伸びていないか
左右カメラで距離に差がないか
```

### 3. `hm` とデコード後の3Dボックス

`hm` は、物体検出ヘッドがBEV上のどこに反応しているかを確認するための出力です。

```text
ヒートマップが存在する場所
閾値前の弱い検出候補
複数ピークや重複検出
```

を調べられます。

### 4. `lane` または `lane_logit`

道路、車線、停止線、道路端の判断を確認できます。

特に生の反応を見る場合は、最終クラスIDの `lane` よりも `lane_logit` の方が有用です。

### 5. `ego` と `traj`

以下を確認できます。

```text
自車がどの未来軌道を選択したか
候補軌道3本の違い
他車両が将来どちらへ移動すると予測されたか
停止物体の未来軌道
```

### 6. `occ` と `risk`

以下を確認できます。

```text
車両・歩行者・障害物の3D占有
道路や歩道の認識
建物・植生・電柱の認識
周囲の危険度推定
```

## 6. 注意点

これらはニューラルネットワークのタスク別出力です。

そのため、

```text
モデルが画像のどの画素を直接見ているか
どの部分を根拠に判断したか
```

を直接表すAttention mapやGrad-CAMではありません。

画素単位で「何を見ているか」を調べるには、別途以下の解析が必要です。

```text
Grad-CAM
遮蔽テスト
カメラ単位の入力アブレーション
画像領域のマスク実験
カメラ間の出力比較
```

特に実装が簡単で効果を確認しやすいのは、次の2つです。

```text
カメラを1台ずつ黒画像に置き換えて出力差分を見る
画像の一部をマスクして検出・レーン出力の変化を見る
```