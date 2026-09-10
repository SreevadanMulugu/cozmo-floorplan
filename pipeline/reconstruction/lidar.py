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
    elif parsed.odometry_file:
        # ARKit RGBD format with per-frame poses, depth PNGs, and confidence maps
        pcd = _reconstruct_from_arkit(parsed)
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


def _depth_to_points(
    depth: np.ndarray, fx: float, fy: float, cx: float, cy: float,
    max_depth: float = 6.0
) -> np.ndarray:
    """Unproject a depth map (H×W float32 metres) to (N, 3) XYZ points in camera space."""
    h, w = depth.shape
    u = np.arange(w, dtype=np.float32)
    v = np.arange(h, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)

    mask = (depth > 0.05) & (depth < max_depth)
    z = depth[mask]
    x = (uu[mask] - cx) * z / fx
    y = (vv[mask] - cy) * z / fy
    return np.stack([x, y, z], axis=1)


def _reconstruct_from_arkit(parsed: ParsedInput) -> o3d.geometry.PointCloud:
    """
    Memory-efficient ARKit RGBD reconstruction.
    Lazy CSV reading (only sampled rows), pure-numpy accumulation, one Open3D
    PointCloud created at the end from a single stacked array.
    """
    import csv
    from PIL import Image as PILImage

    depth_list = sorted(parsed.depth_frames)
    conf_list = sorted(parsed.confidence_frames) if parsed.confidence_frames else []
    has_conf = (len(conf_list) == len(depth_list))

    n_total = len(depth_list)
    # 50 frames: good balance of coverage and memory; gives ~50K pts/frame → 2.5M raw
    STRIDE = max(1, n_total // 50)
    n_sampled = max(1, n_total // STRIDE)
    log.info(f"ARKit RGBD: {n_total} frames, stride={STRIDE} → {n_sampled} sampled")

    # Determine which frame stems we need, then read only those rows from CSV
    use_stems = {depth_list[i].stem for i in range(0, n_total, STRIDE)}

    pose_lut: dict = {}   # stem → (T_wc 4x4, fx, fy, cx, cy)
    with open(str(parsed.odometry_file), newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row = {k.strip(): v.strip() for k, v in row.items()}
            stem = row["frame"]
            if stem not in use_stems:
                continue
            qx, qy, qz, qw = float(row["qx"]), float(row["qy"]), float(row["qz"]), float(row["qw"])
            x, y, z = float(row["x"]), float(row["y"]), float(row["z"])
            fx = float(row["fx"]) if row.get("fx") else (parsed.intrinsics["fx"] if parsed.intrinsics else 1600.0)
            fy = float(row["fy"]) if row.get("fy") else (parsed.intrinsics["fy"] if parsed.intrinsics else 1600.0)
            cx = float(row["cx"]) if row.get("cx") else (parsed.intrinsics["cx"] if parsed.intrinsics else 960.0)
            cy = float(row["cy"]) if row.get("cy") else (parsed.intrinsics["cy"] if parsed.intrinsics else 720.0)
            pose_lut[stem] = (_quat_trans_to_matrix(qx, qy, qz, qw, x, y, z), fx, fy, cx, cy)

    # Accumulate world-space points as plain numpy arrays (no Open3D per-frame objects)
    all_pts: list[np.ndarray] = []

    for i in range(0, n_total, STRIDE):
        depth_path = depth_list[i]
        stem = depth_path.stem
        if stem not in pose_lut:
            continue
        T_wc, fx_full, fy_full, cx_full, cy_full = pose_lut[stem]

        # Depth: uint16 PNG → float32 metres
        depth_img = np.array(PILImage.open(depth_path), dtype=np.uint16).astype(np.float32) / 1000.0

        # Confidence filter: keep pixels with confidence ≥ 1 (medium or high)
        if has_conf:
            conf_img = np.array(PILImage.open(conf_list[i]), dtype=np.uint8)
            depth_img[conf_img < 1] = 0.0

        dh, dw = depth_img.shape
        # Scale per-frame intrinsics from full-res (1920×1440) down to depth resolution (256×192)
        fx_d = fx_full * dw / 1920.0
        fy_d = fy_full * dh / 1440.0
        cx_d = cx_full * dw / 1920.0
        cy_d = cy_full * dh / 1440.0

        pts_cam = _depth_to_points(depth_img, fx_d, fy_d, cx_d, cy_d, max_depth=6.0)
        if len(pts_cam) == 0:
            continue

        # Transform to world coords in numpy only
        ones = np.ones((len(pts_cam), 1), dtype=np.float32)
        pts_world = (T_wc @ np.hstack([pts_cam, ones]).T)[:3].T.astype(np.float32)
        all_pts.append(pts_world)
        del pts_cam, pts_world, depth_img   # free immediately

    if not all_pts:
        raise ValueError("No valid depth frames found in ARKit dataset")

    # One shot: stack all frames, convert to float64 only at Open3D boundary
    import gc
    all_xyz_f32 = np.vstack(all_pts)   # already float32 from per-frame code
    del all_pts
    gc.collect()
    log.info(f"Raw cloud: {len(all_xyz_f32):,} points (f32 {all_xyz_f32.nbytes//1024}KB)")

    # np.vstack result may not be C-contiguous; Open3D Vector3dVector requires it
    log.info("Converting to contiguous float64...")
    all_xyz_f64 = np.ascontiguousarray(all_xyz_f32, dtype=np.float64)
    del all_xyz_f32
    gc.collect()
    log.info(f"Contiguous float64 array: {all_xyz_f64.nbytes//1024}KB C={all_xyz_f64.flags['C_CONTIGUOUS']}; creating PointCloud...")

    combined = o3d.geometry.PointCloud()
    combined.points = o3d.utility.Vector3dVector(all_xyz_f64)
    del all_xyz_f64
    gc.collect()
    log.info(f"PointCloud created: {len(combined.points):,} pts; voxel-downsampling...")

    combined = combined.voxel_down_sample(VOXEL_SIZE)
    log.info(f"After voxel-downsample: {len(combined.points):,} points")
    return combined


def _quat_trans_to_matrix(qx: float, qy: float, qz: float, qw: float,
                           tx: float, ty: float, tz: float) -> np.ndarray:
    """Convert quaternion + translation to 4×4 world-from-camera transform."""
    # Normalise quaternion
    n = (qx**2 + qy**2 + qz**2 + qw**2) ** 0.5
    if n < 1e-10:
        return np.eye(4)
    qx, qy, qz, qw = qx/n, qy/n, qz/n, qw/n

    R = np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),     1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw),     1 - 2*(qx**2 + qy**2)],
    ])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [tx, ty, tz]
    return T


def _extract_rgb_frames(video_path: Path, work_dir: Path) -> Path | None:
    """Extract RGB frames from mp4 to a directory, named by frame index (000000.png)."""
    import subprocess
    out_dir = work_dir / "_rgb_frames"
    out_dir.mkdir(exist_ok=True)
    # Only extract if not already done
    existing = list(out_dir.glob("*.png"))
    if len(existing) > 10:
        return out_dir
    try:
        result = subprocess.run(
            ["ffmpeg", "-i", str(video_path),
             "-vf", "fps=30",   # extract all frames at native fps
             "-q:v", "3",
             str(out_dir / "%06d.png")],
            capture_output=True, timeout=300
        )
        if result.returncode != 0:
            log.warning(f"ffmpeg RGB extraction failed: {result.stderr.decode()[:200]}")
            return None
        return out_dir
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        log.warning(f"ffmpeg not available or timed out: {e}")
        return None


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
    """Build point cloud from per-frame depth TIFF + RGB pairs with numpy unprojection."""
    from PIL import Image

    ix = parsed.intrinsics or {}
    fx = ix.get("fx", 1600.0)
    fy = ix.get("fy", 1600.0)
    cx = ix.get("cx", 960.0)
    cy = ix.get("cy", 720.0)

    poses = None
    if parsed.pose_file:
        raw = np.loadtxt(parsed.pose_file)
        if raw.ndim == 2 and raw.shape[1] == 16:
            poses = raw.reshape(-1, 4, 4)
        elif raw.ndim == 2 and raw.shape[0] % 4 == 0:
            poses = raw.reshape(-1, 4, 4)

    combined = o3d.geometry.PointCloud()
    pairs = list(zip(parsed.depth_frames, parsed.color_frames or [None]*len(parsed.depth_frames)))
    STRIDE = max(1, len(pairs) // 150)

    for i, (depth_path, _) in enumerate(pairs[::STRIDE]):
        depth_img = np.array(Image.open(depth_path), dtype=np.float32)
        if depth_img.max() > 100:   # likely uint16 mm encoding
            depth_img = depth_img / 1000.0

        dh, dw = depth_img.shape
        fx_s = fx * dw / 1920.0
        fy_s = fy * dh / 1440.0
        cx_s = cx * dw / 1920.0
        cy_s = cy * dh / 1440.0
        pts = _depth_to_points(depth_img, fx_s, fy_s, cx_s, cy_s, max_depth=6.0)
        if len(pts) == 0:
            continue

        idx = i * STRIDE
        if poses is not None and idx < len(poses):
            pts_h = np.hstack([pts, np.ones((len(pts), 1))])
            pts = (poses[idx] @ pts_h.T)[:3].T

        frame_pcd = o3d.geometry.PointCloud()
        frame_pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
        combined += frame_pcd
        if (i + 1) % 20 == 0:
            combined = combined.voxel_down_sample(VOXEL_SIZE)

    return combined.voxel_down_sample(VOXEL_SIZE)


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
            num_iterations=1000
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


def _merge_coplanar_walls(walls: list[PlaneSegment]) -> list[PlaneSegment]:
    """
    Merge near-parallel wall planes that represent the same physical wall.
    Groups by similar normal (±5°) AND similar plane offset (±15cm).
    Result is deterministic: geometry-based grouping, not RANSAC ordering.
    """
    if not walls:
        return walls

    ANG_TOL_DEG = 5.0
    DIST_TOL_M = 0.15

    used = [False] * len(walls)
    merged: list[PlaneSegment] = []

    for i, w in enumerate(walls):
        if used[i]:
            continue
        group = [w]
        used[i] = True

        for j, w2 in enumerate(walls):
            if used[j] or j == i:
                continue
            cos_sim = abs(float(np.dot(w.normal, w2.normal)))
            angle_deg = float(np.degrees(np.arccos(np.clip(cos_sim, 0.0, 1.0))))
            if angle_deg > ANG_TOL_DEG:
                continue
            # Perpendicular distance between parallel planes; adjust sign for anti-parallel normals
            d2_adj = -w2.d if float(np.dot(w.normal, w2.normal)) < 0 else w2.d
            if abs(w.d - d2_adj) > DIST_TOL_M:
                continue
            group.append(w2)
            used[j] = True

        if len(group) == 1:
            merged.append(group[0])
            continue

        # Combine inlier clouds into one canonical segment
        combined_pts = np.vstack([np.asarray(g.inlier_cloud.points) for g in group])
        merged_cloud = o3d.geometry.PointCloud()
        merged_cloud.points = o3d.utility.Vector3dVector(combined_pts.astype(np.float64))
        biggest = max(group, key=lambda g: len(g.inlier_cloud.points))
        merged.append(PlaneSegment(
            normal=biggest.normal,
            center=combined_pts.mean(axis=0),
            inlier_cloud=merged_cloud,
            d=biggest.d,
            label="wall",
        ))

    log.info(f"Wall merge: {len(walls)} RANSAC planes → {len(merged)} canonical walls")
    return merged


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

    # Merge near-parallel/coplanar wall segments → deterministic canonical walls
    room_cloud.walls = _merge_coplanar_walls(walls)

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
