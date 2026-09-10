"""
Input parser: detects tier and normalizes inputs to a common intermediate format.

Supported inputs:
  LiDAR  → .ply file (from 3D Scanner App or ARKit app)
           OR directory with depth_*.tiff + rgb_*.png + poses.txt
           OR ARKit RGBD directory (odometry.csv + depth/*.png + rgb.mp4)
  Video  → .mp4 / .mov file
           OR directory with frames/ + poses.txt (from COLMAP)
  Photo  → directory with *.jpg / *.png (one folder per room, or single folder)
"""

from __future__ import annotations
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

Tier = Literal["lidar", "video", "photo"]


@dataclass
class ParsedInput:
    tier: Tier
    root: Path
    # LiDAR
    ply_files: list[Path] = field(default_factory=list)
    depth_frames: list[Path] = field(default_factory=list)   # 32F TIFF per frame
    color_frames: list[Path] = field(default_factory=list)   # paired RGB
    pose_file: Path | None = None                            # camera poses (N×4×4 txt)
    # ARKit RGBD format (odometry.csv + depth/*.png + confidence/*.png + rgb.mp4)
    odometry_file: Path | None = None
    confidence_frames: list[Path] = field(default_factory=list)
    # Video
    video_file: Path | None = None
    extracted_frames_dir: Path | None = None
    # Photo (keyed by room label)
    photo_rooms: dict[str, list[Path]] = field(default_factory=dict)
    # Metadata
    intrinsics: dict | None = None   # fx, fy, cx, cy in pixels


def detect_tier(input_path: str) -> Tier:
    p = Path(input_path)
    if p.is_file():
        ext = p.suffix.lower()
        if ext == ".ply":
            return "lidar"
        if ext in (".mp4", ".mov", ".m4v"):
            return "video"
        if ext in (".jpg", ".jpeg", ".png"):
            return "photo"
    if p.is_dir():
        # Recurse one level to find ARKit RGBD sessions (sub-dirs with odometry.csv)
        for candidate in [p] + [d for d in p.iterdir() if d.is_dir()]:
            if (candidate / "odometry.csv").exists() and (candidate / "depth").is_dir():
                return "lidar"

        files = list(p.rglob("*"))
        exts = {f.suffix.lower() for f in files}
        if ".ply" in exts:
            return "lidar"
        has_video = bool({".mp4", ".mov", ".m4v"} & exts)
        has_images = bool({".jpg", ".jpeg", ".png"} & exts)
        depth_tiffs = [f for f in files if re.search(r"depth", f.name, re.I) and f.suffix.lower() in (".tiff", ".tif")]
        if depth_tiffs:
            return "lidar"
        if has_video:
            return "video"
        if has_images:
            return "photo"
    raise ValueError(f"Cannot detect tier for: {input_path}")


def parse_input(input_path: str, tier: Tier | None = None) -> ParsedInput:
    p = Path(input_path).resolve()
    if tier is None:
        tier = detect_tier(input_path)

    result = ParsedInput(tier=tier, root=p)

    if tier == "lidar":
        _parse_lidar(p, result)
    elif tier == "video":
        _parse_video(p, result)
    elif tier == "photo":
        _parse_photo(p, result)

    return result


def _parse_lidar(p: Path, result: ParsedInput) -> None:
    if p.is_file() and p.suffix.lower() == ".ply":
        result.ply_files = [p]
        return

    # Check for ARKit RGBD session format: odometry.csv + depth/*.png + rgb.mp4
    # May be in p itself or in a single sub-directory
    arkit_candidates = [p]
    if p.is_dir():
        arkit_candidates += [d for d in p.iterdir() if d.is_dir()]
    for candidate in arkit_candidates:
        if (candidate / "odometry.csv").exists() and (candidate / "depth").is_dir():
            result.odometry_file = candidate / "odometry.csv"
            result.depth_frames = sorted((candidate / "depth").glob("*.png"))
            conf_dir = candidate / "confidence"
            if conf_dir.is_dir():
                result.confidence_frames = sorted(conf_dir.glob("*.png"))
            # Video for RGB frames
            for vname in ("rgb.mp4", "video.mp4", "rgb.mov"):
                vpath = candidate / vname
                if vpath.exists():
                    result.video_file = vpath
                    break
            # Global camera matrix as fallback intrinsics
            cam_csv = candidate / "camera_matrix.csv"
            if cam_csv.exists():
                result.intrinsics = _load_camera_matrix_csv(cam_csv)
            return

    # Directory: collect PLYs
    plys = sorted(p.rglob("*.ply"))
    if plys:
        result.ply_files = plys

    # Look for legacy ARKit-style RGB-D frames (depth_*.tiff / rgb_*.png)
    depths = sorted(p.rglob("depth_*.tiff")) + sorted(p.rglob("depth_*.tif"))
    colors = sorted(p.rglob("rgb_*.png")) + sorted(p.rglob("rgb_*.jpg"))
    result.depth_frames = depths
    result.color_frames = colors

    # Poses file
    for name in ("poses.txt", "camera_poses.txt", "trajectory.txt"):
        pose_candidate = p / name
        if pose_candidate.exists():
            result.pose_file = pose_candidate
            break

    result.intrinsics = _load_intrinsics(p)


def _parse_video(p: Path, result: ParsedInput) -> None:
    if p.is_file():
        result.video_file = p
        return

    videos = list(p.rglob("*.mp4")) + list(p.rglob("*.mov"))
    if videos:
        result.video_file = videos[0]

    frames_dir = p / "frames"
    if frames_dir.exists():
        result.extracted_frames_dir = frames_dir

    result.pose_file = (p / "poses.txt") if (p / "poses.txt").exists() else None
    result.intrinsics = _load_intrinsics(p)


def _parse_photo(p: Path, result: ParsedInput) -> None:
    image_exts = {".jpg", ".jpeg", ".png"}

    if p.is_file():
        result.photo_rooms = {"room_0": [p]}
        return

    # Check for per-room subdirectories
    subdirs = [d for d in p.iterdir() if d.is_dir()]
    if subdirs:
        for subdir in sorted(subdirs):
            imgs = sorted(
                f for f in subdir.iterdir()
                if f.suffix.lower() in image_exts
            )
            if imgs:
                result.photo_rooms[subdir.name] = imgs
    else:
        # Flat directory — all images are one room
        imgs = sorted(f for f in p.iterdir() if f.suffix.lower() in image_exts)
        result.photo_rooms["room_0"] = imgs


def _load_intrinsics(base: Path) -> dict | None:
    """Try to load camera intrinsics from intrinsics.txt or camera_matrix.txt."""
    for name in ("intrinsics.txt", "camera_matrix.txt", "K.txt"):
        candidate = base / name
        if candidate.exists():
            try:
                import numpy as np
                K = np.loadtxt(candidate)
                if K.shape == (3, 3):
                    return {"fx": K[0, 0], "fy": K[1, 1], "cx": K[0, 2], "cy": K[1, 2]}
                if K.shape == (4,):
                    return {"fx": K[0], "fy": K[1], "cx": K[2], "cy": K[3]}
            except Exception:
                pass
    return None


def _load_camera_matrix_csv(cam_csv: Path) -> dict | None:
    """Load intrinsics from ARKit camera_matrix.csv (3×3 K matrix, comma-separated)."""
    try:
        import numpy as np
        K = np.loadtxt(str(cam_csv), delimiter=",")
        if K.shape == (3, 3):
            return {"fx": float(K[0, 0]), "fy": float(K[1, 1]),
                    "cx": float(K[0, 2]), "cy": float(K[1, 2])}
    except Exception:
        pass
    return None
