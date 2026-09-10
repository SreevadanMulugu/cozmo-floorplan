"""
Multi-room stitcher with drift correction.

Algorithm:
  1. For each pair of adjacent rooms, compute ICP alignment transform
  2. Build a GTSAM pose graph over all rooms
  3. Optimize to reduce global drift
  4. Apply transforms and merge into single global coordinate frame
  5. Detect room adjacency (shared openings)

Ablation mode: set drift_correction=False to skip optimization (shows impact).
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field

import numpy as np
import open3d as o3d

from pipeline.geometry import RoomGeometry, WallSegment, Opening

log = logging.getLogger(__name__)

ICP_MAX_DIST = 0.05          # 5cm ICP correspondence threshold
ICP_MAX_ITER = 50
ADJACENCY_DIST_M = 0.30      # rooms whose wall endpoints are within this are adjacent


@dataclass
class StitchedPlan:
    rooms: dict[str, RoomGeometry]          # room_id → geometry in GLOBAL frame
    global_polygon: np.ndarray | None       # convex hull of all rooms (N, 2)
    adjacency: list[tuple[str, str, str]]   # (room_a, room_b, opening_type)
    transforms: dict[str, np.ndarray]       # room_id → 4×4 transform applied
    footprint_m2: float = 0.0
    drift_correction_used: bool = True
    # For ablation
    footprint_uncorrected_m2: float = 0.0
    footprint_corrected_m2: float = 0.0


def stitch(
    room_geometries: dict[str, RoomGeometry],
    room_clouds: dict,    # room_id → RoomCloud (for ICP)
    drift_correction: bool = True,
) -> StitchedPlan:
    if len(room_geometries) == 1:
        room_id = list(room_geometries.keys())[0]
        geo = list(room_geometries.values())[0]
        plan = StitchedPlan(
            rooms=room_geometries,
            global_polygon=geo.floor_polygon_2d,
            adjacency=[],
            transforms={room_id: np.eye(4)},
            footprint_m2=geo.floor_area_m2,
            drift_correction_used=False,
        )
        plan.footprint_corrected_m2 = geo.floor_area_m2
        plan.footprint_uncorrected_m2 = geo.floor_area_m2
        return plan

    room_ids = list(room_geometries.keys())

    # Compute pairwise ICP transforms
    pairwise = _compute_pairwise_icp(room_ids, room_clouds)

    # Compute uncorrected footprint (identity transforms)
    uncorrected_fp = _compute_global_footprint(
        room_geometries, {rid: np.eye(4) for rid in room_ids}
    )

    if drift_correction and len(room_ids) > 1:
        transforms = _pose_graph_optimize(room_ids, pairwise)
    else:
        transforms = {rid: np.eye(4) for rid in room_ids}
        if not drift_correction:
            log.info("Drift correction DISABLED (ablation mode)")

    # Apply transforms to geometry
    aligned_geos = _apply_transforms(room_geometries, transforms)

    # Global footprint
    corrected_fp = _compute_global_footprint(aligned_geos, transforms)
    global_poly = _global_polygon(aligned_geos)

    # Adjacency detection
    adjacency = _detect_adjacency(aligned_geos, room_clouds)

    plan = StitchedPlan(
        rooms=aligned_geos,
        global_polygon=global_poly,
        adjacency=adjacency,
        transforms=transforms,
        footprint_m2=corrected_fp,
        drift_correction_used=drift_correction,
        footprint_uncorrected_m2=uncorrected_fp,
        footprint_corrected_m2=corrected_fp,
    )
    log.info(
        f"Stitched {len(room_ids)} rooms. Footprint: "
        f"uncorrected={uncorrected_fp:.2f}m², corrected={corrected_fp:.2f}m²  "
        f"(drift correction {'ON' if drift_correction else 'OFF'})"
    )
    return plan


def _compute_pairwise_icp(
    room_ids: list[str],
    room_clouds: dict
) -> dict[tuple[str, str], np.ndarray]:
    pairwise: dict[tuple[str, str], np.ndarray] = {}
    for i, id_a in enumerate(room_ids):
        for id_b in room_ids[i + 1:]:
            pcd_a = _get_pcd(room_clouds, id_a)
            pcd_b = _get_pcd(room_clouds, id_b)
            if pcd_a is None or pcd_b is None:
                continue
            T = _icp(pcd_a, pcd_b)
            pairwise[(id_a, id_b)] = T
    return pairwise


def _get_pcd(room_clouds: dict, room_id: str) -> o3d.geometry.PointCloud | None:
    cloud = room_clouds.get(room_id)
    if cloud is None:
        return None
    if isinstance(cloud, dict):
        return None
    if hasattr(cloud, "pcd"):
        return cloud.pcd
    return None


def _icp(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud
) -> np.ndarray:
    """Point-to-plane ICP between two room clouds."""
    if not source.has_normals():
        source.estimate_normals()
    if not target.has_normals():
        target.estimate_normals()

    result = o3d.pipelines.registration.registration_icp(
        source, target,
        max_correspondence_distance=ICP_MAX_DIST,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=ICP_MAX_ITER
        )
    )
    return result.transformation


def _pose_graph_optimize(
    room_ids: list[str],
    pairwise: dict[tuple[str, str], np.ndarray]
) -> dict[str, np.ndarray]:
    try:
        import gtsam
        return _gtsam_optimize(room_ids, pairwise)
    except ImportError:
        log.warning("GTSAM not available; using Open3D pose graph optimization")
        return _open3d_pose_graph(room_ids, pairwise)


def _gtsam_optimize(
    room_ids: list[str],
    pairwise: dict[tuple[str, str], np.ndarray]
) -> dict[str, np.ndarray]:
    import gtsam
    from gtsam import Pose3, Rot3

    id_to_idx = {rid: i for i, rid in enumerate(room_ids)}
    graph = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()

    # Prior on first room (anchored)
    noise_prior = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-6]*6))
    graph.add(gtsam.PriorFactorPose3(0, Pose3(), noise_prior))
    for i in range(len(room_ids)):
        initial.insert(i, Pose3())

    noise_between = gtsam.noiseModel.Diagonal.Sigmas(
        np.array([0.01, 0.01, 0.01, 0.05, 0.05, 0.05])
    )

    for (id_a, id_b), T in pairwise.items():
        idx_a = id_to_idx[id_a]
        idx_b = id_to_idx[id_b]
        try:
            pose = Pose3(T)
            graph.add(gtsam.BetweenFactorPose3(idx_a, idx_b, pose, noise_between))
        except Exception as e:
            log.warning(f"GTSAM edge {id_a}-{id_b} failed: {e}")

    params = gtsam.LevenbergMarquardtParams()
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
    result = optimizer.optimize()

    transforms = {}
    for i, rid in enumerate(room_ids):
        pose = result.atPose3(i)
        transforms[rid] = pose.matrix()

    return transforms


def _open3d_pose_graph(
    room_ids: list[str],
    pairwise: dict[tuple[str, str], np.ndarray]
) -> dict[str, np.ndarray]:
    """Fallback pose graph using Open3D."""
    id_to_idx = {rid: i for i, rid in enumerate(room_ids)}
    pg = o3d.pipelines.registration.PoseGraph()

    for i in range(len(room_ids)):
        pg.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.eye(4)))

    for (id_a, id_b), T in pairwise.items():
        idx_a = id_to_idx[id_a]
        idx_b = id_to_idx[id_b]
        info = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
            o3d.geometry.PointCloud(), o3d.geometry.PointCloud(),
            0.05, T
        ) if False else np.eye(6)   # skip expensive info matrix computation
        pg.edges.append(o3d.pipelines.registration.PoseGraphEdge(
            idx_a, idx_b, T, info, uncertain=True
        ))

    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=0.05,
        edge_prune_threshold=0.25,
        preference_loop_closure=0.1,
    )
    o3d.pipelines.registration.global_optimization(
        pg,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        option,
    )

    return {rid: pg.nodes[i].pose for i, rid in enumerate(room_ids)}


def _apply_transforms(
    room_geometries: dict[str, RoomGeometry],
    transforms: dict[str, np.ndarray],
) -> dict[str, RoomGeometry]:
    """Apply 4×4 transforms to 2D room geometries (X, Z only)."""
    from copy import deepcopy
    aligned = {}
    for rid, geo in room_geometries.items():
        T = transforms.get(rid, np.eye(4))
        R2d = T[[0, 2]][:, [0, 2]]   # 2×2 rotation in XZ
        t2d = T[[0, 2], 3]            # 2D translation

        new_geo = deepcopy(geo)
        if new_geo.floor_polygon_2d is not None:
            new_geo.floor_polygon_2d = new_geo.floor_polygon_2d @ R2d.T + t2d
        if new_geo.room_center_2d is not None:
            new_geo.room_center_2d = R2d @ new_geo.room_center_2d + t2d
        for wall in new_geo.walls:
            wall.p1 = R2d @ wall.p1 + t2d
            wall.p2 = R2d @ wall.p2 + t2d
        for op in new_geo.openings:
            op.p_center = R2d @ op.p_center + t2d
        aligned[rid] = new_geo
    return aligned


def _compute_global_footprint(
    room_geometries: dict[str, RoomGeometry],
    transforms: dict[str, np.ndarray],
) -> float:
    from scipy.spatial import ConvexHull
    all_pts = []
    for rid, geo in room_geometries.items():
        if geo.floor_polygon_2d is not None:
            all_pts.append(geo.floor_polygon_2d)
    if not all_pts:
        return sum(g.floor_area_m2 for g in room_geometries.values())
    pts = np.vstack(all_pts)
    try:
        hull = ConvexHull(pts)
        return float(hull.volume)  # In 2D, hull.volume = area
    except Exception:
        return float(sum(g.floor_area_m2 for g in room_geometries.values()))


def _global_polygon(room_geometries: dict[str, RoomGeometry]) -> np.ndarray | None:
    from scipy.spatial import ConvexHull
    all_pts = []
    for geo in room_geometries.values():
        if geo.floor_polygon_2d is not None:
            all_pts.append(geo.floor_polygon_2d)
    if not all_pts:
        return None
    pts = np.vstack(all_pts)
    try:
        hull = ConvexHull(pts)
        return pts[hull.vertices]
    except Exception:
        return pts


def _detect_adjacency(
    room_geometries: dict[str, RoomGeometry],
    room_clouds: dict,
) -> list[tuple[str, str, str]]:
    """Detect adjacent rooms by checking if their openings are near each other."""
    adjacency = []
    room_ids = list(room_geometries.keys())

    for i, id_a in enumerate(room_ids):
        for id_b in room_ids[i + 1:]:
            geo_a = room_geometries[id_a]
            geo_b = room_geometries[id_b]

            # Check if any opening in room A is close to any opening in room B
            connected = False
            op_type = "opening"
            for op_a in geo_a.openings:
                for op_b in geo_b.openings:
                    dist = np.linalg.norm(op_a.p_center - op_b.p_center)
                    if dist <= ADJACENCY_DIST_M:
                        connected = True
                        op_type = op_a.type
                        break
                if connected:
                    break

            # Fallback: check if room polygons overlap or are very close
            if not connected:
                if geo_a.room_center_2d is not None and geo_b.room_center_2d is not None:
                    dist = np.linalg.norm(geo_a.room_center_2d - geo_b.room_center_2d)
                    if dist < 5.0:   # within 5m
                        connected = True
                        op_type = "unknown"

            if connected:
                adjacency.append((id_a, id_b, op_type))

    return adjacency
