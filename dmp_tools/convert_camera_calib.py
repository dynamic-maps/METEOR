#!/usr/bin/env python3
"""Convert vehicle-frame camera mounts into METEOR manifest calibration.

The input uses the vehicle frame x=forward, y=left, z=up and OpenCV camera
axes x=right, y=down, z=forward. The manifest stores the camera-to-ego
transform under the historical key ``T_ego_cam``.
"""
import argparse
import json
import math
import os
import re
import shutil

import cv2
import numpy as np
import yaml


CAMERA_IDS = {
    0: "CAM_FRONT_WIDE",
    1: "CAM_FRONT_RIGHT",
    2: "CAM_BACK_RIGHT",
    3: "CAM_BACK_LEFT",
    4: "CAM_FRONT_LEFT",
}
IMAGE_RE = re.compile(
    r"^(?P<frame>.+)_Camera_(?P<camera>[0-4])\.(?P<ext>jpg|jpeg|png)$",
    re.IGNORECASE,
)
TIMESTAMP_RE = re.compile(
    r"^Record(?P<record>\d+)_"
    r"(?P<date>\d{6})_(?P<time>\d{9})_Camera_[0-4]\.",
    re.IGNORECASE,
)


def _rot_about_axis(axis, angle):
    """Return a Rodrigues rotation for a unit 3-vector."""
    x, y, z = axis
    c, s = math.cos(angle), math.sin(angle)
    v = 1.0 - c
    return np.array([
        [c + x * x * v, x * y * v - z * s, x * z * v + y * s],
        [y * x * v + z * s, c + y * y * v, y * z * v - x * s],
        [z * x * v - y * s, z * y * v + x * s, c + z * z * v],
    ], dtype=np.float64)


def resize_intrinsic(K, source_hw, target_hw):
    source_h, source_w = map(float, source_hw)
    target_h, target_w = map(float, target_hw)

    K = np.asarray(K, dtype=np.float64).copy()
    K[0, :] *= target_w / source_w
    K[1, :] *= target_h / source_h
    return K


def ego_to_camera(position, ypr_deg):
    """Build an ego->camera matrix from position and yaw/pitch/roll."""
    yaw, pitch, roll = np.deg2rad(ypr_deg)
    forward = np.array([
        math.cos(pitch) * math.cos(yaw),
        math.cos(pitch) * math.sin(yaw),
        math.sin(pitch),
    ])
    right = np.array([math.sin(yaw), -math.cos(yaw), 0.0])
    down = np.cross(forward, right)
    roll_rot = _rot_about_axis(forward, roll)
    right, down = roll_rot @ right, roll_rot @ down

    # OpenCV camera coordinates are x=right, y=down, z=forward.
    R = np.stack((right, down, forward), axis=0)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = -R @ np.asarray(position, dtype=np.float64)
    return T


def convert(config, target_hw=(432, 768)):
    cameras = config.get("cameras", [])
    if not cameras:
        raise ValueError("cameras must contain at least one camera")

    result = {}
    source_sizes = {tuple(camera["image_hw"]) for camera in cameras}
    if len(source_sizes) != 1:
        raise ValueError("all cameras must use the same source image_hw")

    source_hw = next(iter(source_sizes))
    target_h, target_w = target_hw

    for camera in cameras:
        name = camera["id"]
        position = camera["position_ego_m"]
        ypr = camera["ypr_deg"]

        if len(position) != 3 or len(ypr) != 3:
            raise ValueError(
                f"{name}: position and ypr must each have 3 values"
            )

        K = np.asarray(camera["intrinsic"], dtype=np.float64)
        if K.shape != (3, 3):
            raise ValueError(f"{name}: intrinsic must be a 3x3 matrix")

        K = resize_intrinsic(K, source_hw, target_hw)

        T_cam_ego = ego_to_camera(position, ypr)
        T_ego_cam = np.linalg.inv(T_cam_ego)

        result[name] = {
            "K": K.tolist(),
            "T_ego_cam": T_ego_cam.tolist(),
            "position_ego_m": list(map(float, position)),
            "ypr_deg": list(map(float, ypr)),
        }

    return {
        "img_hw": [int(target_h), int(target_w)],
        "cams": result,
        "frames": [],
    }


def load_manifest(path):
    with open(path, encoding="utf-8") as stream:
        manifest = json.load(stream)
    if "cams" not in manifest:
        raise ValueError(f"{path}: manifest must contain cams")
    if "img_hw" not in manifest or len(manifest["img_hw"]) != 2:
        raise ValueError(f"{path}: manifest must contain img_hw=[height,width]")
    missing = [name for name in CAMERA_IDS.values()
               if name not in manifest["cams"]]
    if missing:
        raise ValueError(f"{path}: missing camera calibration: {', '.join(missing)}")
    return manifest


def resize_manifest_calibration(manifest, target_hw):
    source_hw = tuple(map(int, manifest["img_hw"]))
    target_hw = tuple(map(int, target_hw))
    if source_hw != target_hw:
        for camera in manifest["cams"].values():
            camera["K"] = resize_intrinsic(
                camera["K"], source_hw, target_hw
            ).tolist()
    manifest["img_hw"] = [target_hw[0], target_hw[1]]
    return manifest


def collect_images(images_dir):
    """Group regular Camera_0..4 images by their shared frame prefix."""
    grouped = {}
    for name in sorted(os.listdir(images_dir)):
        match = IMAGE_RE.match(name)
        if match is None:
            continue
        frame_key = match.group("frame")
        camera_name = CAMERA_IDS[int(match.group("camera"))]
        grouped.setdefault(frame_key, {})[camera_name] = name
    return grouped


def frame_timestamp(frame_key):
    match = TIMESTAMP_RE.match(frame_key + "_Camera_0.jpg")
    if match is None:
        raise ValueError(f"cannot parse timestamp from frame key: {frame_key}")
    date = match.group("date")
    clock = match.group("time")
    # YYMMDD_HHMMSSmmm -> seconds from an arbitrary epoch; only differences matter.
    return (
        int(date[:2]) * 365.25 * 24 * 3600
        + int(date[2:4]) * 31 * 24 * 3600
        + int(date[4:6]) * 24 * 3600
        + int(clock[:2]) * 3600
        + int(clock[2:4]) * 60
        + int(clock[4:6])
        + int(clock[6:9]) / 1000.0
    )


def populate_frames(manifest, images_dir, output_dir, frame_keys=None):
    target_h, target_w = map(int, manifest["img_hw"])
    os.makedirs(os.path.join(output_dir, "img"), exist_ok=True)
    grouped = collect_images(images_dir)
    required = set(CAMERA_IDS.values())
    frames = []
    skipped = 0

    selected = set(frame_keys) if frame_keys is not None else None
    timestamps = []
    for frame_key in sorted(grouped):
        if selected is not None and frame_key not in selected:
            continue
        images = grouped[frame_key]
        if set(images) != required:
            skipped += 1
            continue

        frame_index = len(frames)
        frame_images = {}
        failed = False
        for camera_name in CAMERA_IDS.values():
            source_name = images[camera_name]
            source_path = os.path.join(images_dir, source_name)
            image = cv2.imread(source_path, cv2.IMREAD_COLOR)
            if image is None:
                failed = True
                break
            same_size = image.shape[:2] == (target_h, target_w)
            output_name = source_name
            output_path = os.path.join(output_dir, "img", output_name)
            if same_size:
                shutil.copy2(source_path, output_path)
            else:
                image = cv2.resize(
                    image, (target_w, target_h), interpolation=cv2.INTER_AREA
                )
                if not cv2.imwrite(output_path, image):
                    failed = True
                    break
            frame_images[camera_name] = os.path.join("img", output_name)
        if failed:
            skipped += 1
            continue
        frames.append({"frame": frame_index, "imgs": frame_images})
        timestamps.append(frame_timestamp(frame_key))

    manifest["frames"] = frames
    return len(frames), skipped, timestamps


def read_runs(path):
    runs = []
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            values = [value.strip() for value in line.strip().split(",")]
            if len(values) < 2 or not values[0]:
                continue
            records = {f"Record{value.zfill(3)}" for value in values[1:] if value}
            if records:
                runs.append((values[0], records))
    return runs


def write_ego_motion(timestamps, output_dir, capture_distance_m):
    timestamps = np.asarray(timestamps, dtype=np.float64)
    v0 = np.zeros(len(timestamps), dtype=np.float32)
    if len(timestamps) > 1:
        dt = np.diff(timestamps)
        speed = np.divide(
            float(capture_distance_m), dt,
            out=np.zeros_like(dt), where=dt > 0,
        ).astype(np.float32)
        v0[1:] = speed
        v0[0] = speed[0]
    np.savez(
        os.path.join(output_dir, "ego_motion.npz"),
        v0=v0,
        stamp=timestamps,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="camera mount YAML")
    parser.add_argument("--images", required=True,
                        help="directory containing Camera_0..4 images")
    parser.add_argument("--output", required=True,
                        help="METEOR output directory")
    parser.add_argument("--run-data", required=True,
                        help="CSV run definition file")
    parser.add_argument(
        "--capture-distance-m",
        type=float,
        default=2.0,
        help="distance represented by one capture interval in metres",
    )
    parser.add_argument(
        "--img-hw",
        default="432,768",
        help="output image size as HEIGHT,WIDTH",
    )
    args = parser.parse_args()

    target_hw = tuple(int(v) for v in args.img_hw.split(","))
    if len(target_hw) != 2:
        raise ValueError("--img-hw must be HEIGHT,WIDTH")

    if args.capture_distance_m <= 0:
        parser.error("--capture-distance-m must be positive")
    with open(args.config, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    manifest_template = convert(config, target_hw=target_hw)
    grouped = collect_images(args.images)
    runs = read_runs(args.run_data)
    os.makedirs(args.output, exist_ok=True)
    scenes = []
    for run_id, records in runs:
        run_manifest = json.loads(json.dumps(manifest_template))
        frame_keys = [
            key for key in grouped
            if key.split("_", 1)[0] in records
        ]
        run_dir = os.path.join(args.output, run_id)
        os.makedirs(run_dir, exist_ok=True)
        frame_count, skipped, timestamps = populate_frames(
            run_manifest, args.images, run_dir, frame_keys
        )
        run_manifest["scene"] = run_id

        write_ego_motion(timestamps, run_dir, args.capture_distance_m)
        output_manifest = os.path.join(run_dir, "manifest.json")
        with open(output_manifest, "w", encoding="utf-8") as stream:
            json.dump(run_manifest, stream, indent=2)
            stream.write("\n")
        scenes.append(run_id)
        print(f"wrote {output_manifest}: {frame_count} frames")
        if skipped:
            print(f"skipped {skipped} incomplete frames in {run_id}")
    with open(os.path.join(args.output, "scenes.txt"), "w", encoding="utf-8") as stream:
        stream.write("\n".join(scenes) + ("\n" if scenes else ""))


if __name__ == "__main__":
    main()
