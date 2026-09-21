#!/usr/bin/env bash
set -euo pipefail

# デフォルト設定 (引数や環境変数で上書き可能)
SCENES="${1:-out/demo6/scenes.txt}"
OUT="${2:-out/autolabel}"
ROOT="${BEVLANE_ROOT:-out/demo6}"
WORKERS="${WORKERS:-16}"
STRIDE="${STRIDE:-1}"

# 出力先ディレクトリの作成
mkdir -p "$OUT"

echo "=== Running BEV autolabel batch ==="
echo "Scenes:  $SCENES"
echo "Root:    $ROOT"
echo "Out:     $OUT"
echo "Workers: $WORKERS"
echo "Stride:  $STRIDE"
echo "==================================="

BEVLANE_ROOT="$ROOT" python3 run_batch.py \
  --scenes "$SCENES" \
  --out "$OUT" \
  --workers "$WORKERS" \
  --stride "$STRIDE"
