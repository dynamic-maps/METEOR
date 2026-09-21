#!/bin/bash
set -e

ENGINE_FILE="out/meteor_v157_5cam_fp16.engine"

if [[ ! -f "$ENGINE_FILE" ]]; then
    echo "[INFO] Engine not found. Building..."

    python3 deploy/build_engine_fp16.py \
        out/meteor_v157_5cam.onnx \
        "$ENGINE_FILE" \
        8

    echo "[INFO] Engine build completed."
else
    echo "[INFO] Engine already exists."
fi

echo "[INFO] Running inference..."

mkdir -p out/dmp

METEOR_TH2D=0.30 METEOR_SEG2D_OVERLAY=0 METEOR_OCC_PANEL=0 METEOR_2D_HIDE=7 PYTHONPATH=. \
python3 deploy/orin_realtime.py \
    --engine "$ENGINE_FILE" \
    --root out/custom_dataset \
    --out out/dmp.mp4
