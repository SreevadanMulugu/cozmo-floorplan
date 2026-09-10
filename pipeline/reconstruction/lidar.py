"""
LiDAR tier reconstruction.

Input:  PLY point cloud (from 3D Scanner App / ARKit / provided sample data)
        Optionally: per-frame depth TIFF + RGB pairs + poses.txt

Output: RoomCloud dataclass with per-room point clouds and aligned global cloud
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import open3d as o3d

from pipeline.input_parser import ParsedInput

log = logging.getLogger(__name__)

VOXEL_SIZE = 0.02          # 2 cm voxel downsampling
RANSAC_DIST = 0.02         # 2 cm plane inlier threshold
MIN_WALL_INLIERS = 200     # minimum points to consider a wall plane valid
MIN_PLANE_AREA = 0.5       # m² — discard tiny planes
FLOOR_NORMAL_TOL = 0.15    # max deviation from vertical for floor/ceiling normals
WALL_NORMAL_TOL = 0.15     # max deviation from horizontal for wall normals


@dataclass
class PlaneSegment:
    normal: np.ndarray        # unit normal vector (3,)
    center: np.ndarray        # centroid (3,)
    inlier_cloud: o3d.geometry.PointCloud
    d: float                  # plane equation: n·x + d = 0
    label: str = "unknown"    # floor | ceiling | wall | other


@dataclass
class RoomCloud:
    pcd: o3d.geometry.PointCloud          # full (possibly multi-room) point cloud
    planes: list[PlaneSegment] = field(default_factory=list)
    floor: PlaneSegment | None = None
    ceiling: PlaneSegment | None = None
    walls: list[PlaneSegment] = field(default_factory=list)
    # Populated downstream by geometry.py
    room_bounds_min: np.ndarray | None = None
    room_bounds_max: np.ndarray | None = None


def reconstruct(parsed: ParsedInput) -> RoomCloud:
    """Main entry point for LiDAR tier."""
    if parsed.ply_files:
        pcd = _load_and_merge_plys(parsed.ply_files)
    elif parsed.depth_frames:
        pcd = _reconstruct_from_rgbd(parsed)
    else:
        raise ValueError("LiDAR input has neither PLY files nor depth frames")

    log.info(f"Loaded point cloud: {len(pcd.points):,} points")
    pcd = _preprocess(pcd)
    log.info(f"After preprocessing: {len(pcd.points):,} points")

    room_cloud = RoomCloud(pcd=pcd)
    _segment_planes(room_cloud)
    return room_cloud


def _load_and_merge_plys(ply_files: list[Path]) -> o3d.geometry.PointCloud:
    clouds = []
    for f in ply_files:
        pcd = o3d.io.read_point_cloud(str(f))
        if len(pcd.points) == 0:
            log.warning(f"Empty PLY: {f}")
            continue
        clouds.append(pcd)
    if not clouds:
        raise ValueError("All PLY files were empty")
    if len(clouds) == 1:
        return clouds[0]
    merged = clouds[0]
    for c in clouds[1:]:
        merged += c
    return merged


def _reconstruct_from_rgbd(parsed: ParsedInput) -> o3d.geometry.PointCloud:
    """Build point cloud from ARKit-style per-frame depth TIFF + RGB pairs."""
    from PIL import Image

    if parsed.intrinsics:
        ix = parsed.intrinsics
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=1920, height=1440,
            fx=ix["fx"], fy=ix["fy"], cx=ix["cx"], cy=ix["cy"]
        )
    else:
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault
        )

    # Load poses if available
    poses = None
    if parsed.pose_file:
        raw = np.loadtxt(parsed.pose_file)
        if raw.ndim == 2 and raw.shape[1] == 16:
            poses = raw.reshape(-1, 4, 4)
        elif raw.ndim == 2 and raw.shape[0] % 4 == 0:
            poses = raw.reshape(-1, 4, 4)

    combined = o3d.geometry.PointCloud()
    pairs = list(zip(parsed.depth_frames, parsed.color_frames))

    for i, (depth_path, color_path) in enumerate(pairs):
        # Load 32F depth TIFF (meters)
        depth_img = np.array(Image.open(depth_path), dtype=np.float32)
        color_img = np.array(Image.open(color_path).convert("RGB"))

        depth_o3d = o3d.geometry.Image(depth_img)
        color_o3d = o3d.geometry.Image(color_img.astype(np.uint8))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d, depth_o3d,
            depth_scale=1.0,
            depth_trunc=5.0,
            convert_rgb_to_intensity=False
        )
        frame_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)

        if poses is not None and i < len(poses):
            frame_pcd.transform(poses[i])

        combined += frame_pcd

    return combined


def _preprocess(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    # Voxel downsample
    pcd = pcd.voxel_down_sample(VOXEL_SIZE)
    # Remove statistical outliers
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    # Estimate normals for later plane fitting
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    pcd.orient_normals_consistent_tangent_plane(10)
    return pcd


def _segment_planes(room_cloud: RoomCloud) -> None:
    """Iteratively extract planes from point cloud and classify them."""
    remaining = room_cloud.pcd
    pts_array = np.asarray(remaining.points)
    max_planes = 20
    planes: list[PlaneSegment] = []

    for _ in range(max_planes):
        if len(remaining.points) < MIN_WALL_INLIERS * 2:
            break

        plane_model, inliers = remaining.segment_plane(
            distance_threshold=RANSAC_DIST,
            ransac_n=3,
            num_iterations=500
        )
        if len(inliers) < MIN_WALL_INLIERS:
            break

        a, b, c, d = plane_model
        normal = np.array([a, b, c])
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-6:
            break
        normal = normal / norm_len

        inlier_cloud = remaining.select_by_index(inliers)
        center = np.mean(np.asarray(inlier_cloud.points), axis=0)

        seg = PlaneSegment(
            normal=normal,
            center=center,
            inlier_cloud=inlier_cloud,
            d=d / norm_len,
            label="unknown",   # will be set in _assign_structural_planes
        )
        planes.append(seg)

        remaining = remaining.select_by_index(inliers, invert=True)

    room_cloud.planes = planes
    _assign_structural_planes(room_cloud)


def _classify_plane_for_axis(
    normal: np.ndarray, center: np.ndarray, up_axis: int, bounds: tuple
) -> str:
    """
    Classify a plane as floor, ceiling, wall, or other given a specific up-axis.
    Does NOT rely on normal sign — uses centroid position instead.
    """
    abs_n = np.abs(normal)
    lo, hi = bounds

    if abs_n[up_axis] > (1 - FLOOR_NORMAL_TOL):
        cy = center[up_axis]
        span = hi - lo
        rel = (cy - lo) / max(span, 0.01)
        if rel < 0.25:
            return "floor"
        elif rel > 0.75:
            return "ceiling"
        return "other"

    if abs_n[up_axis] < WALL_NORMAL_TOL:
        return "wall"

    return "other"


def _assign_structural_planes(room_cloud: RoomCloud) -> None:
    pts = np.asarray(room_cloud.pcd.points)

    # Always try Y-up first (ARKit/Open3D standard); fall back to Z-up
    for up_axis in [1, 2]:
        lo = pts[:, up_axis].min() if len(pts) > 0 else 0.0
        hi = pts[:, up_axis].max() if len(pts) > 0 else 3.0
        bounds = (lo, hi)

        for seg in room_cloud.planes:
            seg.label = _classify_plane_for_axis(seg.normal, seg.center, up_axis, bounds)

        floors = [p for p in room_cloud.planes if p.label == "floor"]
        ceilings = [p for p in room_cloud.planes if p.label == "ceiling"]

        if floors or ceilings:
            # Validate: height between floor/ceiling should be realistic (1.5–5m)
            floor_h = min(floors, key=lambda p: p.center[up_axis]).center[up_axis] if floors else lo
            ceil_h = max(ceilings, key=lambda p: p.center[up_axis]).center[up_axis] if ceilings else hi
            room_h = abs(ceil_h - floor_h)
            if 1.5 <= room_h <= 5.0:
                break   # good axis found
            # Height unrealistic — try other axis

    # Derive up_axis from final classification
    floors = [p for p in room_cloud.planes if p.label == "floor"]
    ceilings = [p for p in room_cloud.planes if p.label == "ceiling"]
    walls = [p for p in room_cloud.planes if p.label == "wall"]

    floors = [p for p in room_cloud.planes if p.label == "floor"]
    ceilings = [p for p in room_cloud.planes if p.label == "ceiling"]
    walls = [p for p in room_cloud.planes if p.label == "wall"]

    # Pick floor with lowest centroid along up-axis, ceiling highest
    if floors:
        room_cloud.floor = min(floors, key=lambda p: p.center[up_axis])
    if ceilings:
        room_cloud.ceiling = max(ceilings, key=lambda p: p.center[up_axis])
    room_cloud.walls = walls

    # Enforce parallel-plane constraint: floor/ceiling normals must agree within 2°
    if room_cloud.floor and room_cloud.ceiling:
        cos_sim = abs(np.dot(room_cloud.floor.normal, room_cloud.ceiling.normal))
        angle_deg = np.degrees(np.arccos(np.clip(cos_sim, -1, 1)))
        if angle_deg > 2.0:
            log.warning(
                f"Floor/ceiling normals diverge by {angle_deg:.1f}° — "
                "possible drift; forcing ceiling normal to match floor"
            )
            # Force ceiling normal to be antiparallel to floor
            room_cloud.ceiling.normal = -room_cloud.floor.normal

    log.info(
        f"Segments: {len(floors)} floor, {len(ceilings)} ceiling, "
        f"{len(walls)} walls, {len([p for p in room_cloud.planes if p.label=='other'])} other"
    )
