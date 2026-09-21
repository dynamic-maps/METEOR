# Conda/Pip 環境の保存と復元方法

本プロジェクトでは、Conda 管理のパッケージと Pip 管理のパッケージを分離して管理します。

## 環境のエクスポート

### 1. Conda パッケージの出力

インストール履歴のみを出力します。

```bash
conda env export --from-history > environment.yml
```

このファイルには以下のような情報が保存されます。

```yaml
name: myenv
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - pytorch
  - cudatoolkit
```

### 2. Pip パッケージの出力

```bash
pip list --format=freeze > requirements.txt
```

このファイルには以下のような情報が保存されます。

```text
mmcv==2.1.0
mmengine==0.10.5
mmdet==3.3.0
```

`pip freeze` を使用すると、

```text
packaging @ file:///home/conda/...
```

のようなローカルビルド情報が含まれる場合があります。

そのため、本プロジェクトでは以下を推奨します。

```bash
pip list --format=freeze > requirements.txt
```

---

## 環境の復元方法

### 1. Conda 環境作成

environment.yml から環境を作成します。

```bash
conda env create -f environment.yml
```

### 2. 環境を有効化

環境名を確認します。

```bash
grep "^name:" environment.yml
```

例:

```text
name: segformer
```

有効化:

```bash
conda activate segformer
```

### 3. Pip パッケージのインストール

```bash
pip install -r requirements.txt
```

---

## 既存環境への適用

既に Conda 環境が存在する場合は、

```bash
conda activate <env_name>
pip install -r requirements.txt
```

のみ実行してください。

---

## 動作確認

インストール後に以下を確認してください。

### Python

```bash
python --version
```

### Condaパッケージ

```bash
conda list
```

### Pipパッケージ

```bash
pip list
```

### GPU認識確認（必要な場合）

```bash
nvidia-smi
```

### PyTorch GPU確認

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

期待値:

```text
True
```

---

## 注意事項

- Conda パッケージは `environment.yml` で管理する
- Pip パッケージは `requirements.txt` で管理する
- 新しいライブラリを追加した場合は両ファイルを更新する
- CUDA ドライバは別途インストールが必要
- GPU サーバ間で環境を共有する場合は Python のメジャーバージョンを合わせることを推奨
