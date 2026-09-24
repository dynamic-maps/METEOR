#!/bin/bash
set -e
ONNX_FILE="out/meteor_v157_5cam.onnx"
ENGINE_FILE="out/meteor_v157_5cam_fp16.engine"

if [[ ! -f "$ONNX_FILE" ]]; then
    echo "[INFO] ONNX file not found. Building $ONNX_FILE"
    python3 deploy/export_onnx.py \
        --ckpt models/meteor_v157.pt \
        --model v52 \
        --cam-layout 5cam \
        --uint8-in \
        --argmax-out \
        --lane-logits \
        --no-hist \
        --depth-mean \
        --out "$ONNX_FILE"

    echo "[INFO] Export ONNX file completed."
else
    echo "[INFO] ONNX file already exists."
fi

if [[ ! -f "$ENGINE_FILE" ]]; then
    echo "[INFO] Engine not found. Building..."

    python3 deploy/build_engine_fp16.py \
        "$ONNX_FILE" \
        "$ENGINE_FILE" \
        8

    echo "[INFO] Engine build completed."
else
    echo "[INFO] Engine already exists."
fi

echo "[INFO] Running inference..."

mkdir -p out/dmp

METEOR_TH2D=0.30 METEOR_SEG2D_OVERLAY=1 METEOR_OCC_PANEL=0 METEOR_2D_HIDE=7 PYTHONPATH=. \
python3 deploy/orin_realtime.py \
  --engine "$ENGINE_FILE" \
  --cam-layout 5cam \
  --root out/custom_dataset \
  --dump-dir out/dmp_intermediates \
  --out out/dmp.mp4

