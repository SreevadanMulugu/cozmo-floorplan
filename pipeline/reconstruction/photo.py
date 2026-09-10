"""
Photo tier reconstruction.

Input:  Per-room photo folders, 2-8 images each.
        Uses COLMAP SfM for poses + ZoeDepth for metric depth.

Design philosophy: this is the WEAKEST tier. We produce honest WIDE confidence
intervals rather than fake accuracy. The gate is ±8% with calibrated intervals.
"""

from __future__ import annotations
import logging
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

from pipeline.input_parser import ParsedInput
from pipeline.reconstruction.lidar import (
    RoomCloud, PlaneSegment, _preprocess, _segment_planes
)

log = logging.getLogger(__name__)

# Per-image uncertainty in metres (calibrated from ZoeDepth indoor benchmarks)
DEPTH_UNCERTAINTY_PER_M = 0.05   # 5% of depth value
MIN_IMAGES_FOR_SFM = 2


def reconstruct(parsed: ParsedInput, output_dir: Path | None = None) -> dict[str, RoomCloud]:
    """
    Returns a dict of room_label → RoomCloud.
    Multi-room stitching is handled by stitcher.py.
    """
    work_dir = output_dir or (parsed.root / "_photo_work")
    work_dir.mkdir(parents=True, exist_ok=True)

    room_clouds: dict[str, RoomCloud] = {}
    for room_label, image_paths in parsed.photo_rooms.items():
        log.info(f"Processing photo room: {room_label} ({len(image_paths)} images)")
        room_dir = work_dir / room_label
        room_dir.mkdir(exist_ok=True)
        cloud = _reconstruct_room(room_label, image_paths, room_dir)
        room_clouds[room_label] = cloud

    return room_clouds


def _reconstruct_room(
    room_label: str,
    image_paths: list[Path],
    work_dir: Path
) -> RoomCloud:
    n_images = len(image_paths)
    log.info(f"  {n_images} images")

    # 1. Estimate metric depth per image
    depths = _estimate_depths(image_paths, work_dir)

    # 2. Estimate poses via COLMAP SfM (best effort — may fail with few images)
    poses, intrinsic = _estimate_poses(image_paths, work_dir)

    # 3. Build point cloud from depth + poses
    pcd = _build_point_cloud(image_paths, depths, poses, intrinsic)
    if len(pcd.points) < 100:
        log.warning(f"  Very sparse point cloud ({len(pcd.points)} pts) for {room_label}")

    pcd = _preprocess(pcd)
    room_cloud = RoomCloud(pcd=pcd)
    _segment_planes(room_cloud)

    # Attach photo-tier uncertainty metadata
    room_cloud._n_images = n_images  # type: ignore
    room_cloud._tier = "photo"       # type: ignore

    return room_cloud


def _estimate_depths(image_paths: list[Path], work_dir: Path) -> list[np.ndarray]:
    """Run ZoeDepth on each image to get metric depth maps."""
    model = _load_zoe_depth()
    depths = []

    from PIL import Image as PILImage
    for img_path in image_paths:
        cache = work_dir / (img_path.stem + "_depth.npy")
        if cache.exists():
            depths.append(np.load(str(cache)))
            continue
        try:
            if model is not None:
                import torch
                img = PILImage.open(img_path).convert("RGB")
                depth = model.infer_pil(img)   # H×W float32, metres
            else:
                # Fallback: constant 2.5m depth
                img = PILImage.open(img_path)
                depth = np.full((img.height, img.width), 2.5, dtype=np.float32)
            depth = depth.astype(np.float32)
            np.save(str(cache), depth)
            depths.append(depth)
        except Exception as e:
            log.warning(f"  Depth estimation failed for {img_path.name}: {e}")
            depths.append(np.full((480, 640), 2.5, dtype=np.float32))

    return depths


def _load_zoe_depth():
    try:
        import torch
        # ZoeDepth is loaded via torch.hub
        device = "mps" if torch.backends.mps.is_available() else \
                 "cuda" if torch.cuda.is_available() else "cpu"
        model = torch.hub.load("isl-org/ZoeDepth", "ZoeD_N", pretrained=True)
        model = model.to(device).eval()
        log.info(f"Loaded ZoeDepth on {device}")
        return model
    except Exception as e:
        log.warning(f"Could not load ZoeDepth: {e}; using constant depth fallback")
        return None


def _estimate_poses(
    image_paths: list[Path],
    work_dir: Path
) -> tuple[dict[str, np.ndarray], o3d.camera.PinholeCameraIntrinsic]:
    if len(image_paths) < MIN_IMAGES_FOR_SFM:
        log.info("  Too few images for SfM; using identity pose")
        return (
            {p.name: np.eye(4) for p in image_paths},
            _default_intrinsic(image_paths[0])
        )

    # Create a temporary image directory with symlinks
    img_dir = work_dir / "images"
    img_dir.mkdir(exist_ok=True)
    for img_path in image_paths:
        link = img_dir / img_path.name
        if not link.exists():
            try:
                link.symlink_to(img_path)
            except Exception:
                import shutil
                shutil.copy(img_path, link)

    try:
        import pycolmap
        db_path = work_dir / "colmap.db"
        sparse_dir = work_dir / "sparse"
        sparse_dir.mkdir(exist_ok=True)

        pycolmap.extract_features(db_path, img_dir)
        pycolmap.match_exhaustive(db_path)
        maps = pycolmap.incremental_mapping(db_path, img_dir, sparse_dir)

        if not maps:
            raise RuntimeError("COLMAP produced no reconstruction")

        reconstruction = maps[0]
        cam = list(reconstruction.cameras.values())[0]

        try:
            fx = fy = cam.focal_length
            cx, cy = cam.principal_point_x, cam.principal_point_y
            w, h = int(cam.width), int(cam.height)
        except AttributeError:
            w, h = 1920, 1080
            fx = fy = max(w, h) * 0.8
            cx, cy = w / 2, h / 2

        intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)

        poses: dict[str, np.ndarray] = {}
        for img_id, image in reconstruction.images.items():
            R = image.rotation_matrix()
            t = image.tvec
            T_wc = np.eye(4)
            T_wc[:3, :3] = R.T
            T_wc[:3, 3] = -R.T @ t
            poses[image.name] = T_wc

        log.info(f"  COLMAP registered {len(poses)}/{len(image_paths)} images")
        return poses, intrinsic

    except Exception as e:
        log.warning(f"  COLMAP SfM failed: {e}; using identity poses")
        return (
            {p.name: np.eye(4) for p in image_paths},
            _default_intrinsic(image_paths[0])
        )


def _default_intrinsic(img_path: Path) -> o3d.camera.PinholeCameraIntrinsic:
    try:
        from PIL import Image as PILImage
        w, h = PILImage.open(img_path).size
        fx = fy = max(w, h) * 0.8
        return o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, w / 2, h / 2)
    except Exception:
        return o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault
        )


def _build_point_cloud(
    image_paths: list[Path],
    depths: list[np.ndarray],
    poses: dict[str, np.ndarray],
    intrinsic: o3d.camera.PinholeCameraIntrinsic
) -> o3d.geometry.PointCloud:
    from PIL import Image as PILImage

    combined = o3d.geometry.PointCloud()

    for img_path, depth in zip(image_paths, depths):
        color = np.array(PILImage.open(img_path).convert("RGB"), dtype=np.uint8)

        # Resize depth to match colour
        if color.shape[:2] != depth.shape[:2]:
            from PIL import Image as PilI
            depth = np.array(
                PilI.fromarray(depth).resize(
                    (color.shape[1], color.shape[0]), PilI.BILINEAR
                )
            )

        color_o3d = o3d.geometry.Image(color)
        depth_o3d = o3d.geometry.Image(depth.astype(np.float32))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d, depth_o3d,
            depth_scale=1.0, depth_trunc=5.0, convert_rgb_to_intensity=False
        )

        # Adjust intrinsics to actual image size
        h, w = color.shape[:2]
        actual_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            w, h,
            intrinsic.intrinsic_matrix[0, 0],
            intrinsic.intrinsic_matrix[1, 1],
            intrinsic.intrinsic_matrix[0, 2],
            intrinsic.intrinsic_matrix[1, 2],
        )

        frame_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, actual_intrinsic)
        T = poses.get(img_path.name, np.eye(4))
        frame_pcd.transform(T)
        combined += frame_pcd

    return combined
