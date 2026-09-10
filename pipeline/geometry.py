"""
Geometry normalizer: extracts structured room geometry from a RoomCloud.

Produces:
  - Wall line segments in 2D (top-down XZ projection)
  - Room bounding polygon
  - Floor area
  - Ceiling height (with spread for repeatability)
  - Opening detection (gaps in wall planes)
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import ConvexHull

from pipeline.reconstruction.lidar import RoomCloud, PlaneSegment

log = logging.getLogger(__name__)

MIN_WALL_LENGTH = 0.3    # metres — discard wall segments shorter than this
OPENING_GAP_M = 0.5      # a gap ≥ 0.5m in a wall projection is an opening candidate
OPENING_MIN_W = 0.5      # opening must be ≥ 0.5m wide
OPENING_MAX_W = 3.5      # opening must be ≤ 3.5m wide (door/window, not full gap)


@dataclass
class WallSegment:
    p1: np.ndarray         # 2D start (x, z) in metres
    p2: np.ndarray         # 2D end
    length_m: float
    height_m: float
    wall_id: str = ""
    normal_2d: np.ndarray = field(default_factory=lambda: np.zeros(2))


@dataclass
class Opening:
    wall_id: str
    p_center: np.ndarray   # 2D centre of opening (x, z)
    width_m: float
    type: str = "unknown"  # door | window | opening


@dataclass
class RoomGeometry:
    walls: list[WallSegment] = field(default_factory=list)
    openings: list[Opening] = field(default_factory=list)
    floor_polygon_2d: np.ndarray | None = None   # (N, 2) in metres
    floor_area_m2: float = 0.0
    ceiling_height_m: float = 0.0
    room_center_2d: np.ndarray | None = None


def extract_geometry(room_cloud: RoomCloud) -> RoomGeometry:
    geo = RoomGeometry()

    # Ceiling height
    geo.ceiling_height_m = _compute_ceiling_height(room_cloud)

    # Walls → 2D line segments
    for i, wall_plane in enumerate(room_cloud.walls):
        seg = _wall_plane_to_segment(wall_plane, room_cloud, wall_id=f"w{i+1}")
        if seg and seg.length_m >= MIN_WALL_LENGTH:
            geo.walls.append(seg)

    # Floor polygon and area from point cloud top-down projection
    geo.floor_polygon_2d, geo.floor_area_m2 = _compute_floor_polygon(room_cloud)

    if geo.floor_polygon_2d is not None and len(geo.floor_polygon_2d) > 0:
        geo.room_center_2d = geo.floor_polygon_2d.mean(axis=0)

    # Opening detection per wall
    for wall_seg in geo.walls:
        openings = _detect_openings(wall_seg, room_cloud)
        geo.openings.extend(openings)

    log.info(
        f"Geometry: {len(geo.walls)} walls, {len(geo.openings)} openings, "
        f"area={geo.floor_area_m2:.2f}m², height={geo.ceiling_height_m:.2f}m"
    )
    return geo


def _compute_ceiling_height(room_cloud: RoomCloud) -> float:
    """Height = distance between floor and ceiling planes along the dominant up-axis."""
    if room_cloud.floor and room_cloud.ceiling:
        floor_n = np.abs(room_cloud.floor.normal)
        up_axis = int(np.argmax(floor_n))
        h = abs(room_cloud.ceiling.center[up_axis] - room_cloud.floor.center[up_axis])
        if 1.5 <= h <= 5.0:
            return h
    # Fallback: use point cloud vertical extents (try Y then Z)
    pts = np.asarray(room_cloud.pcd.points)
    if len(pts) < 2:
        return 2.4
    for ax in [1, 2]:
        rng = pts[:, ax].max() - pts[:, ax].min()
        if 1.5 <= rng <= 5.0:
            return float(rng)
    return 2.4


def _wall_plane_to_segment(
    plane: PlaneSegment,
    room_cloud: RoomCloud,
    wall_id: str
) -> WallSegment | None:
    """Project wall inlier points to 2D (XZ) and fit a line segment."""
    pts = np.asarray(plane.inlier_cloud.points)
    if len(pts) < 20:
        return None

    # Project to top-down (X, Z) — ignore Y
    xz = pts[:, [0, 2]]

    # PCA to find main axis direction
    xz_centered = xz - xz.mean(axis=0)
    cov = xz_centered.T @ xz_centered / len(xz)
    _, vecs = np.linalg.eigh(cov)
    axis = vecs[:, -1]    # principal direction (largest eigenvalue)

    # Project onto axis
    proj = xz_centered @ axis
    p_min = xz.mean(axis=0) + axis * proj.min()
    p_max = xz.mean(axis=0) + axis * proj.max()
    length = float(np.linalg.norm(p_max - p_min))

    # Height from Y extent of inliers
    height = float(np.clip(pts[:, 1].max() - pts[:, 1].min(), 0.5, 5.0))

    # 2D normal (perpendicular to axis, XZ plane)
    normal_2d = np.array([-axis[1], axis[0]])
    # Orient to face room center
    center_2d = np.mean(np.asarray(room_cloud.pcd.points)[:, [0, 2]], axis=0)
    mid_2d = (p_min + p_max) / 2
    if np.dot(normal_2d, center_2d - mid_2d) < 0:
        normal_2d = -normal_2d

    return WallSegment(
        p1=p_min, p2=p_max,
        length_m=length,
        height_m=height,
        wall_id=wall_id,
        normal_2d=normal_2d,
    )


def _compute_floor_polygon(room_cloud: RoomCloud) -> tuple[np.ndarray | None, float]:
    """Compute convex hull of floor-level points as the room footprint."""
    pts = np.asarray(room_cloud.pcd.points)
    if len(pts) < 3:
        return None, 0.0

    # Use floor plane to define "floor level"
    if room_cloud.floor:
        floor_y = room_cloud.floor.center[1]
        margin = 0.5   # include points within 50cm above floor
        mask = (pts[:, 1] >= floor_y - 0.05) & (pts[:, 1] <= floor_y + margin)
        floor_pts = pts[mask]
    else:
        # Fall back to bottom 20% of points
        y_min, y_max = pts[:, 1].min(), pts[:, 1].max()
        cutoff = y_min + 0.2 * (y_max - y_min)
        floor_pts = pts[pts[:, 1] <= cutoff]

    xz = floor_pts[:, [0, 2]]
    if len(xz) < 4:
        return xz, 0.0

    try:
        hull = ConvexHull(xz)
        vertices = xz[hull.vertices]
        area = float(hull.area)  # ConvexHull.area in 2D = perimeter; volume = area
        # In scipy, for 2D: hull.volume = area of polygon
        area = float(hull.volume)
        return vertices, area
    except Exception:
        # Degenerate hull; return bounding box area
        x_range = xz[:, 0].max() - xz[:, 0].min()
        z_range = xz[:, 1].max() - xz[:, 1].min()
        return xz, float(x_range * z_range)


def _detect_openings(wall_seg: WallSegment, room_cloud: RoomCloud) -> list[Opening]:
    """Detect gaps in wall point projections as candidate openings."""
    openings = []
    pts = np.asarray(room_cloud.pcd.points)
    if len(pts) == 0:
        return openings

    xz = pts[:, [0, 2]]
    axis = (wall_seg.p2 - wall_seg.p1)
    length = np.linalg.norm(axis)
    if length < 1e-6:
        return openings
    axis_unit = axis / length
    perp = np.array([-axis_unit[1], axis_unit[0]])

    # Find points near this wall (within ±0.2m of wall plane)
    origin = wall_seg.p1
    along = (xz - origin) @ axis_unit
    across = (xz - origin) @ perp

    near_wall = (along >= -0.1) & (along <= length + 0.1) & (np.abs(across) <= 0.2)
    near_pts_along = along[near_wall]

    if len(near_pts_along) < 10:
        return openings

    # Sort and look for gaps
    near_pts_along = np.sort(near_pts_along)
    gaps = np.diff(near_pts_along)
    gap_starts = near_pts_along[:-1][gaps >= OPENING_GAP_M]
    gap_ends = near_pts_along[1:][gaps >= OPENING_GAP_M]

    for gs, ge in zip(gap_starts, gap_ends):
        width = ge - gs
        if OPENING_MIN_W <= width <= OPENING_MAX_W:
            center_along = (gs + ge) / 2
            center_2d = origin + axis_unit * center_along
            op_type = "door" if width <= 1.2 else "window"
            openings.append(Opening(
                wall_id=wall_seg.wall_id,
                p_center=center_2d,
                width_m=float(width),
                type=op_type,
            ))

    return openings
