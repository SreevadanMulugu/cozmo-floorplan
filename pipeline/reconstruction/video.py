"""
Video tier reconstruction.

Pipeline:
  1. Extract frames from video (ffmpeg)
  2. Estimate camera poses with COLMAP (pycolmap)
  3. Estimate metric depth per frame with Depth Anything V2
  4. Fuse RGB-D into a TSDF volume with Open3D
  5. Return a RoomCloud

Drift correction is applied via GTSAM pose graph (see stitcher.py for multi-room).
"""

from __future__ import annotations
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

from pipeline.input_parser import ParsedInput
from pipeline.reconstruction.lidar import RoomCloud, _preprocess, _segment_planes

log = logging.getLogger(__name__)

FRAME_STRIDE = 10        # sample 1 frame every N for depth inference (speed)
DEPTH_TRUNC = 4.0        # metres — clip depth beyond this (indoor rooms)
TSDF_VOXEL = 0.02        # 2 cm TSDF voxel


def reconstruct(parsed: ParsedInput, output_dir: Path | None = None) -> RoomCloud:
    """Main entry point for video tier."""
    work_dir = output_dir or (parsed.root / "_video_work")
    work_dir.mkdir(parents=True, exist_ok=True)

    frames_dir = work_dir / "frames"
    if parsed.extracted_frames_dir:
        frames_dir = parsed.extracted_frames_dir
    elif parsed.video_file:
        frames_dir = _extract_frames(parsed.video_file, work_dir)

    if not frames_dir.exists() or not list(frames_dir.glob("*.jpg")):
        raise RuntimeError(f"No frames found in {frames_dir}")

    frame_paths = sorted(frames_dir.glob("*.jpg"))
    log.info(f"Total frames: {len(frame_paths)}")

    # 1. Camera poses
    # If the video file is inside an ARKit capture directory (has odometry.csv),
    # use the ARKit poses directly instead of running COLMAP on compressed RGB.
    arkit_dir = None
    if parsed.video_file:
        parent = parsed.video_file.parent
        if (parent / "odometry.csv").exists():
            arkit_dir = parent
    if arkit_dir:
        log.info("ARKit poses found — using odometry.csv instead of COLMAP")
        poses, intrinsic = _arkit_poses(arkit_dir, frame_paths)
    else:
        poses, intrinsic = _estimate_poses_colmap(frames_dir, work_dir, frame_paths)

    # 2. Metric depth per frame via Depth Anything V2
    depth_dir = work_dir / "depths"
    depth_dir.mkdir(exist_ok=True)
    _estimate_depth(frame_paths[::FRAME_STRIDE], depth_dir)

    # 3. TSDF fusion
    pcd = _fuse_tsdf(frame_paths[::FRAME_STRIDE], depth_dir, poses, intrinsic)

    log.info(f"Video point cloud: {len(pcd.points):,} points")
    pcd = _preprocess(pcd)

    room_cloud = RoomCloud(pcd=pcd)
    _segment_planes(room_cloud)
    return room_cloud


def _extract_frames(video_path: Path, work_dir: Path) -> Path:
    frames_dir = work_dir / "frames"
    frames_dir.mkdir(exist_ok=True)

    # Try ffmpeg first; fall back to cv2 (always available when OpenCV is installed)
    import shutil
    if shutil.which("ffmpeg"):
        cmd = [
            "ffmpeg", "-i", str(video_path),
            "-vf", "fps=3",
            "-q:v", "2",
            str(frames_dir / "%06d.jpg"),
            "-y", "-loglevel", "error"
        ]
        log.info("Extracting frames from video (ffmpeg)...")
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode == 0:
            extracted = list(frames_dir.glob("*.jpg"))
            log.info(f"Extracted {len(extracted)} frames via ffmpeg")
            return frames_dir
        log.warning("ffmpeg failed; falling back to cv2")

    # cv2 fallback: sample at ~3 fps
    import cv2
    log.info("Extracting frames from video (cv2 fallback)...")
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, int(fps / 3))
    idx = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % stride == 0:
            out_path = frames_dir / f"{saved:06d}.jpg"
            cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            saved += 1
        idx += 1
    cap.release()
    log.info(f"Extracted {saved} frames via cv2")
    return frames_dir


def _arkit_poses(
    arkit_dir: Path,
    frame_paths: list[Path],
) -> tuple[dict[str, np.ndarray], o3d.camera.PinholeCameraIntrinsic]:
    """Load poses from ARKit odometry.csv. Returns poses keyed by extracted frame filename."""
    import csv as csv_mod

    # Read odometry CSV: columns qx,qy,qz,qw,x,y,z,fx,fy,cx,cy (per-frame)
    odom_rows = []
    with open(arkit_dir / "odometry.csv") as f:
        reader = csv_mod.DictReader(f)
        for row in reader:
            odom_rows.append(row)

    if not odom_rows:
        log.warning("Empty odometry.csv; falling back to identity poses")
        return _identity_poses(frame_paths), _default_intrinsic(arkit_dir)

    # Sample the same frame indices that cv2 extracted (frame_paths are in order)
    total_frames = len(odom_rows)
    n_extracted = len(frame_paths)
    # Uniformly distribute extracted frames across odometry rows
    indices = np.linspace(0, total_frames - 1, n_extracted, dtype=int)

    # Intrinsics from first row
    row0 = odom_rows[0]
    try:
        fx = float(row0["fx"]); fy = float(row0["fy"])
        cx = float(row0["cx"]); cy = float(row0["cy"])
    except (KeyError, ValueError):
        fx = fy = 1440.0; cx = cy = 960.0
    intrinsic = o3d.camera.PinholeCameraIntrinsic(1920, 1440, fx, fy, cx, cy)

    poses: dict[str, np.ndarray] = {}
    for fp, idx in zip(frame_paths, indices):
        row = odom_rows[int(idx)]
        try:
            qx, qy, qz, qw = float(row["qx"]), float(row["qy"]), float(row["qz"]), float(row["qw"])
            tx, ty, tz = float(row["x"]), float(row["y"]), float(row["z"])
        except (KeyError, ValueError):
            T_wc = np.eye(4)
            poses[fp.name] = T_wc
            continue
        # Quaternion → rotation matrix
        R = _quat_to_mat(qx, qy, qz, qw)
        T_wc = np.eye(4)
        T_wc[:3, :3] = R
        T_wc[:3, 3] = [tx, ty, tz]
        poses[fp.name] = T_wc

    log.info(f"ARKit poses: {len(poses)} frames mapped from {total_frames} odometry entries")
    return poses, intrinsic


def _quat_to_mat(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Quaternion → 3×3 rotation matrix."""
    x, y, z, w = qx, qy, qz, qw
    return np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - z*w),   2*(x*z + y*w)],
        [  2*(x*y + z*w), 1 - 2*(x*x + z*z),   2*(y*z - x*w)],
        [  2*(x*z - y*w),   2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])


def _estimate_poses_colmap(
    frames_dir: Path,
    work_dir: Path,
    frame_paths: list[Path]
) -> tuple[dict[str, np.ndarray], o3d.camera.PinholeCameraIntrinsic]:
    """Run COLMAP SfM and return {filename: 4×4 pose} and intrinsics."""
    try:
        import pycolmap
    except ImportError:
        log.warning("pycolmap not installed; using identity poses (degraded accuracy)")
        return _identity_poses(frame_paths), _default_intrinsic(frames_dir)

    db_path = work_dir / "colmap.db"
    sparse_dir = work_dir / "sparse"
    sparse_dir.mkdir(exist_ok=True)

    log.info("Running COLMAP feature extraction...")
    pycolmap.extract_features(db_path, frames_dir)

    log.info("Running COLMAP feature matching...")
    pycolmap.match_exhaustive(db_path)

    log.info("Running COLMAP incremental mapping...")
    maps = pycolmap.incremental_mapping(db_path, frames_dir, sparse_dir)

    if not maps:
        log.warning("COLMAP mapping produced no reconstruction; using identity poses")
        return _identity_poses(frame_paths), _default_intrinsic(frames_dir)

    reconstruction = maps[0]

    # Extract camera intrinsics
    cameras = list(reconstruction.cameras.values())
    if not cameras:
        log.warning("COLMAP reconstruction has no cameras; using identity poses")
        return _identity_poses(frame_paths), _default_intrinsic(frames_dir)
    cam = cameras[0]
    if hasattr(cam, "focal_length"):
        fx = fy = cam.focal_length
        cx = cam.principal_point_x
        cy = cam.principal_point_y
        w, h = int(cam.width), int(cam.height)
    else:
        # Fallback
        w, h = 1920, 1080
        fx = fy = 1200.0
        cx, cy = w / 2, h / 2

    intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)

    # Build pose dict: image_name → camera-to-world 4×4
    # pycolmap 3.13: cam_from_world() is a method returning Rigid3d
    poses: dict[str, np.ndarray] = {}
    for img_id, image in reconstruction.images.items():
        if not image.has_pose:
            continue
        try:
            cfw = image.cam_from_world()
            R = cfw.rotation.matrix()
            t = cfw.translation
        except Exception:
            continue
        # Convert world-to-camera to camera-to-world
        T_wc = np.eye(4)
        T_wc[:3, :3] = R.T
        T_wc[:3, 3] = -R.T @ t
        poses[image.name] = T_wc

    log.info(f"COLMAP registered {len(poses)} images")
    # If reconstruction is too sparse (<5 images), fall back to identity
    if len(poses) < 5:
        log.warning(f"COLMAP registered only {len(poses)} images; falling back to identity poses")
        return _identity_poses(frame_paths), intrinsic
    return poses, intrinsic


def _identity_poses(frame_paths: list[Path]) -> dict[str, np.ndarray]:
    return {f.name: np.eye(4) for f in frame_paths}


def _default_intrinsic(frames_dir: Path) -> o3d.camera.PinholeCameraIntrinsic:
    # Try to read first image dimensions
    try:
        from PIL import Image
        imgs = sorted(frames_dir.glob("*.jpg"))
        if imgs:
            w, h = Image.open(imgs[0]).size
            fx = fy = max(w, h) * 0.8   # rough iPhone focal length estimate
            return o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, w / 2, h / 2)
    except Exception:
        pass
    return o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault
    )


def _estimate_depth(frame_paths: list[Path], depth_dir: Path) -> None:
    """Run Depth Anything V2 metric indoor model on all frames."""
    import cv2

    model = _load_depth_model()
    if model is None:
        log.warning("Depth Anything V2 not available; using constant depth (degraded)")
        for fp in frame_paths:
            np.save(str(depth_dir / (fp.stem + ".npy")), np.full((480, 640), 2.5))
        return

    log.info(f"Running metric depth on {len(frame_paths)} frames...")
    for fp in frame_paths:
        out_path = depth_dir / (fp.stem + ".npy")
        if out_path.exists():
            continue
        img = cv2.imread(str(fp))
        if img is None:
            continue
        depth = model.infer_image(img)  # returns H×W float32 in metres
        np.save(str(out_path), depth.astype(np.float32))

    log.info("Depth estimation complete")


def _load_depth_model():
    """Load Depth Anything V2 metric indoor model (ViT-L or ViT-S fallback)."""
    try:
        import torch
        # Try the cloned repo path
        repo_candidates = [
            Path("../Depth-Anything-V2"),
            Path("/Users/apple/Depth-Anything-V2"),
            Path.home() / "Depth-Anything-V2",
        ]
        for repo in repo_candidates:
            if (repo / "depth_anything_v2" / "dpt.py").exists():
                sys.path.insert(0, str(repo))
                break

        from depth_anything_v2.dpt import DepthAnythingV2

        model_path = Path("models/depth_anything_v2/depth_anything_v2_metric_hypersim_vitl.pth")
        if not model_path.exists():
            # Try ViT-S fallback
            model_path = Path("models/depth_anything_v2/depth_anything_v2_metric_hypersim_vits.pth")

        device = "mps" if torch.backends.mps.is_available() else \
                 "cuda" if torch.cuda.is_available() else "cpu"
        log.info(f"Loading Depth Anything V2 on {device}...")

        model = DepthAnythingV2(
            encoder="vitl" if "vitl" in str(model_path) else "vits",
            features=256 if "vitl" in str(model_path) else 64,
            out_channels=[256, 512, 1024, 1024] if "vitl" in str(model_path) else [48, 96, 192, 384],
            use_bn=False, use_clstoken=False,
            max_depth=10.0
        )
        state = torch.load(str(model_path), map_location="cpu")
        model.load_state_dict(state)
        model = model.to(device).eval()
        return model

    except Exception as e:
        log.warning(f"Could not load Depth Anything V2: {e}")
        return None


def _fuse_tsdf(
    frame_paths: list[Path],
    depth_dir: Path,
    poses: dict[str, np.ndarray],
    intrinsic: o3d.camera.PinholeCameraIntrinsic
) -> o3d.geometry.PointCloud:
    """Fuse RGB-D frames into a TSDF volume and extract a point cloud."""
    from PIL import Image

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=TSDF_VOXEL,
        sdf_trunc=0.04,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )

    integrated = 0
    for fp in frame_paths:
        depth_path = depth_dir / (fp.stem + ".npy")
        if not depth_path.exists():
            continue

        color = np.array(Image.open(fp).convert("RGB"), dtype=np.uint8)
        depth = np.load(str(depth_path)).astype(np.float32)

        # Resize to match if needed
        if color.shape[:2] != depth.shape[:2]:
            from PIL import Image as PILImage
            depth = np.array(
                PILImage.fromarray(depth).resize(
                    (color.shape[1], color.shape[0]), PILImage.BILINEAR
                )
            )

        color_o3d = o3d.geometry.Image(color)
        depth_o3d = o3d.geometry.Image(depth)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d, depth_o3d,
            depth_scale=1.0,
            depth_trunc=DEPTH_TRUNC,
            convert_rgb_to_intensity=False
        )

        # Get pose for this frame (fall back to identity)
        T = poses.get(fp.name, poses.get(fp.stem, np.eye(4)))
        extrinsic = np.linalg.inv(T)   # TSDF integrate expects world-to-camera

        volume.integrate(rgbd, intrinsic, extrinsic)
        integrated += 1

    log.info(f"Integrated {integrated} RGB-D frames into TSDF")
    if integrated == 0:
        return o3d.geometry.PointCloud()

    pcd = volume.extract_point_cloud()
    return pcd
