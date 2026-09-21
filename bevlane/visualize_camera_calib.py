#!/usr/bin/env python3
import argparse
import json
import math

import matplotlib.pyplot as plt
import numpy as np


def project_point(camera, point_ego):
    K = np.asarray(camera["K"], dtype=float)
    T_ego_cam = np.asarray(camera["T_ego_cam"], dtype=float)

    # manifest の T_ego_cam は camera -> ego なので反転する
    T_cam_ego = np.linalg.inv(T_ego_cam)

    point = np.r_[point_ego, 1.0]
    point_cam = T_cam_ego @ point
    x, y, z = point_cam[:3]

    if z <= 0:
        return None, point_cam[:3]

    uv = K @ np.array([x, y, z])
    return uv[:2] / uv[2], point_cam[:3]


def camera_center_and_axis(camera):
    T_ego_cam = np.asarray(camera["T_ego_cam"], dtype=float)

    # camera -> ego 行列の原点がカメラ位置
    center = T_ego_cam[:3, 3]

    # OpenCV camera の optical axis は +Z
    axis = T_ego_cam[:3, :3] @ np.array([0.0, 0.0, 1.0])
    axis /= np.linalg.norm(axis)

    return center, axis


def optical_axis_point(camera, distance):
    center, axis = camera_center_and_axis(camera)
    return center + axis * distance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("--out", default="camera_calib_check.png")
    parser.add_argument(
        "--forward-distance",
        type=float,
        default=10.0,
        help="vehicle forward test point distance in metres",
    )
    args = parser.parse_args()

    with open(args.manifest, encoding="utf-8") as stream:
        manifest = json.load(stream)

    height, width = manifest["img_hw"]
    forward_point = np.array([args.forward_distance, 0.0, 0.0])

    camera_names = list(manifest["cams"])
    figure, axes = plt.subplots(
        2,
        len(camera_names),
        figsize=(4 * len(camera_names), 8),
        squeeze=False,
    )

    for index, name in enumerate(camera_names):
        camera = manifest["cams"][name]
        K = np.asarray(camera["K"], dtype=float)
        cx, cy = K[0, 2], K[1, 2]

        axis_point = optical_axis_point(camera, args.forward_distance)
        axis_projected, axis_cam = project_point(camera, axis_point)

        forward_projected, forward_cam = project_point(camera, forward_point)

        image_axis = axes[0, index]
        image_axis.set_xlim(0, width)
        image_axis.set_ylim(height, 0)
        image_axis.set_aspect("equal")
        image_axis.axvline(cx, color="gray", linestyle="--", linewidth=1)
        image_axis.axhline(cy, color="gray", linestyle="--", linewidth=1)
        image_axis.scatter(
            [cx],
            [cy],
            color="lime",
            marker="*",
            s=100,
            label="optical axis / principal point",
        )

        if axis_projected is not None:
            u, v = axis_projected
            image_axis.scatter([u], [v], color="white", marker="+", s=100)
            image_axis.text(
                u,
                v,
                f"  axis ({u:.0f}, {v:.0f})",
                color="white",
            )

        if forward_projected is not None:
            u, v = forward_projected
            visible = 0 <= u < width and 0 <= v < height
            color = "blue" if visible else "orange"
            image_axis.scatter([u], [v], color=color, s=60)
            image_axis.text(
                u,
                v,
                f"  vehicle +X ({u:.0f}, {v:.0f})",
                color=color,
            )

        image_axis.set_title(name)
        image_axis.set_xlabel("u [pixel]")
        image_axis.set_ylabel("v [pixel]")

        # BEV表示
        bev_axis = axes[1, index]
        center, optical_axis = camera_center_and_axis(camera)
        axis_point = center + optical_axis * args.forward_distance

        bev_axis.scatter([0], [0], color="black", marker="x")
        bev_axis.scatter([center[0]], [center[1]], color="blue")
        bev_axis.arrow(
            center[0],
            center[1],
            optical_axis[0] * 2.0,
            optical_axis[1] * 2.0,
            color="red",
            width=0.015,
            head_width=0.12,
            length_includes_head=True,
        )
        bev_axis.scatter(
            [axis_point[0]],
            [axis_point[1]],
            color="lime",
            marker="*",
            s=100,
        )
        bev_axis.scatter(
            [forward_point[0]],
            [forward_point[1]],
            color="blue",
            marker="x",
            s=70,
        )

        plot_points = np.vstack([center, axis_point, forward_point])
        x_margin = max(3.0, 0.25 * np.ptp(plot_points[:, 0]))
        y_margin = max(3.0, 0.25 * np.ptp(plot_points[:, 1]))
        bev_axis.set_xlim(plot_points[:, 0].min() - x_margin,
                          plot_points[:, 0].max() + x_margin)
        bev_axis.set_ylim(plot_points[:, 1].min() - y_margin,
                          plot_points[:, 1].max() + y_margin)
        bev_axis.set_aspect("equal")
        bev_axis.grid(True)
        bev_axis.set_xlabel("ego X: forward [m]")
        bev_axis.set_ylabel("ego Y: left [m]")
        bev_axis.set_title(
            f"camera center=({center[0]:.2f}, {center[1]:.2f})"
        )

    figure.suptitle(
        f"Calibration check: lime=optical axis, blue=vehicle +X "
        f"({args.forward_distance:g} m)"
    )
    figure.tight_layout()
    figure.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
