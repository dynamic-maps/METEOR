#!/usr/bin/env python3
"""Visualize intermediate METEOR dump NPZ files.

Expected layout:
    <dump-dir>/<scene>/<frame>.npz

The output video contains BEV lane, lane-logit confidence, BEV detection
heatmap, per-camera 2D segmentation, and per-camera 2D detection heatmaps.
Raw tensors remain available in the source NPZ files.
"""
import argparse
import glob
import os

import cv2
import numpy as np


BEV_NAMES = ["unlabeled", "road", "sidewalk", "crosswalk", "laneline",
             "stopline", "road_edge", "marking", "parking"]
SEG_NAMES = ["background", "misc", "car", "truck", "bus", "motorcycle",
             "bicycle", "pedestrian", "marking", "traffic_light",
             "traffic_sign", "road", "sidewalk", "lane", "crosswalk",
             "unused", "wall", "building", "vegetation", "sky", "pole"]

# BGR colors, kept distinct from the model's RGB palette for readable panels.
BEV_COLORS = np.array([
    (0, 0, 0), (90, 90, 90), (160, 90, 140), (255, 200, 0),
    (255, 255, 255), (0, 0, 220), (0, 140, 255), (60, 220, 220),
    (120, 80, 40),
], np.uint8)
SEG_COLORS = np.array([
    (0, 0, 0), (180, 180, 180), (0, 0, 220), (0, 100, 180),
    (0, 60, 220), (0, 0, 255), (120, 40, 180), (40, 40, 255),
    (220, 220, 220), (0, 180, 255), (0, 220, 220), (128, 64, 128),
    (232, 35, 244), (255, 0, 0), (255, 0, 255), (0, 0, 0),
    (156, 102, 102), (70, 70, 70), (35, 142, 107), (180, 130, 70),
    (128, 196, 128),
], np.uint8)
CAM_NAMES = ["FRONT_WIDE", "FRONT_RIGHT", "BACK_RIGHT", "BACK_LEFT",
             "FRONT_LEFT", "CAM5"]


def as_array(z, name):
    value = z[name]
    return np.asarray(value)


def squeeze_batch(value):
    return value[0] if value.ndim > 0 and value.shape[0] == 1 else value


def class_map(value, classes):
    value = squeeze_batch(value)
    if value.ndim >= 3 and value.shape[0] == classes:
        return value.argmax(0).astype(np.uint8)
    return value.astype(np.uint8)


def color_map(labels, colors):
    return colors[np.clip(labels.astype(np.int32), 0, len(colors) - 1)]


def fit_panel(image, width, height):
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_NEAREST)
    return image


def label_panel(canvas, image, x, y, width, height, title):
    panel = fit_panel(image, width, height)
    canvas[y:y + height, x:x + width] = panel
    cv2.putText(canvas, title, (x + 8, y + 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 1, cv2.LINE_AA)


def montage(images, names, width, height, cols=3):
    if not images:
        return np.zeros((height, width, 3), np.uint8)
    rows = int(np.ceil(len(images) / cols))
    tile_w = width // cols
    tile_h = height // rows
    out = np.zeros((height, width, 3), np.uint8)
    for i, image in enumerate(images):
        r, c = divmod(i, cols)
        x, y = c * tile_w, r * tile_h
        tile = cv2.resize(image, (tile_w, tile_h),
                          interpolation=cv2.INTER_NEAREST)
        out[y:y + tile_h, x:x + tile_w] = tile
        cv2.putText(out, names[i], (x + 6, y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1,
                    cv2.LINE_AA)
    return out


def render_frame(path):
    z = np.load(path)
    canvas = np.full((1080, 1920, 3), 18, np.uint8)
    scene = os.path.basename(os.path.dirname(path))
    frame = int(np.asarray(z["frame"]).item()) if "frame" in z else 0
    cv2.putText(canvas, f"METEOR intermediates  {scene}  f{frame:04d}",
                (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (230, 230, 230),
                2, cv2.LINE_AA)

    if "lane" in z:
        lane = class_map(as_array(z, "lane"), len(BEV_NAMES))
        label_panel(canvas, color_map(lane, BEV_COLORS), 16, 48, 610, 400,
                    "lane: BEV semantic class")
    if "lane_logit" in z:
        logits = squeeze_batch(as_array(z, "lane_logit")).astype(np.float32)
        logits -= logits.max(0, keepdims=True)
        prob = np.exp(logits)
        prob /= np.maximum(prob.sum(0, keepdims=True), 1e-6)
        conf = (prob.max(0) * 255).astype(np.uint8)
        heat = cv2.applyColorMap(conf, cv2.COLORMAP_TURBO)
        label_panel(canvas, heat, 646, 48, 610, 400,
                    "lane_logit: max softmax confidence")
    if "hm" in z:
        hm = squeeze_batch(as_array(z, "hm")).astype(np.float32)
        if hm.ndim == 3:
            hm = 1.0 / (1.0 + np.exp(-hm))
            hm = hm.max(0)
        hm_img = cv2.applyColorMap((np.clip(hm, 0, 1) * 255).astype(np.uint8),
                                   cv2.COLORMAP_TURBO)
        label_panel(canvas, hm_img, 1276, 48, 628, 400,
                    "hm: BEV 3D detection response")

    seg_images, seg_names = [], []
    if "seg2d" in z:
        seg = squeeze_batch(as_array(z, "seg2d"))
        if seg.ndim == 4 and seg.shape[1] == len(SEG_NAMES):
            seg = seg.argmax(1)
        for i in range(seg.shape[0]):
            seg_images.append(color_map(seg[i], SEG_COLORS))
            seg_names.append(CAM_NAMES[i] if i < len(CAM_NAMES) else f"CAM{i}")
    label_panel(canvas, montage(seg_images, seg_names, 930, 560),
                16, 480, 930, 560, "seg2d: per-camera semantic class")

    hm_images, hm_names = [], []
    hm_scales = [as_array(z, f"hm2d_s{i}") for i in range(3)
                 if f"hm2d_s{i}" in z]
    if hm_scales:
        ncam = squeeze_batch(hm_scales[0]).shape[0]
        for cam in range(ncam):
            maps = []
            for scale in hm_scales:
                value = squeeze_batch(scale)[cam].astype(np.float32)
                if value.ndim == 3:
                    value = 1.0 / (1.0 + np.exp(-value)).max(0)
                maps.append(cv2.resize(value, (300, 220),
                                       interpolation=cv2.INTER_LINEAR))
            hm_images.append(cv2.applyColorMap(
                (np.clip(np.maximum.reduce(maps), 0, 1) * 255).astype(np.uint8),
                cv2.COLORMAP_TURBO))
            hm_names.append(CAM_NAMES[cam] if cam < len(CAM_NAMES)
                            else f"CAM{cam}")
    label_panel(canvas, montage(hm_images, hm_names, 930, 560),
                974, 480, 930, 560,
                "hm2d_s0..s2: max sigmoid response across scales")
    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-dir", default="out/dmp_intermediates")
    parser.add_argument("--out", default="out/dmp_intermediates.mp4")
    parser.add_argument("--scene", default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--fps", type=float, default=10)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    pattern = os.path.join(args.dump_dir, args.scene or "*", "*.npz")
    paths = sorted(glob.glob(pattern))
    if not paths:
        parser.error(f"no dump files found: {pattern}")
    paths = paths[::max(args.stride, 1)]
    if args.limit:
        paths = paths[:args.limit]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"),
                             args.fps, (1920, 1080))
    if not writer.isOpened():
        raise RuntimeError(f"cannot open video writer: {args.out}")
    try:
        for i, path in enumerate(paths):
            writer.write(render_frame(path))
            if i % 50 == 0:
                print(f"{i + 1}/{len(paths)} {path}", flush=True)
    finally:
        writer.release()
    print(f"done: {len(paths)} frames -> {args.out}", flush=True)


if __name__ == "__main__":
    main()