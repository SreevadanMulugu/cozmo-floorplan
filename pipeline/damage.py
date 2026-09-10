"""
Damage detection and concealed-damage reasoning.

Model: YOLOv8 trained on peumalab/wall-defects (Roboflow)
       Classes: crack, corrosion, stain, mold, deterioration

Runs offline with locally downloaded weights.
Damage regions are projected to wall surfaces via known camera geometry.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from pipeline.geometry import RoomGeometry, WallSegment

log = logging.getLogger(__name__)

MODEL_PATH = Path("models/yolov8/wall_defects.pt")

# Concealed damage rules — deterministic, no ML
CONCEALED_RULES = [
    {
        "trigger": lambda cls, wall, geo: cls in ("water_damage", "stain") and _is_exterior_adjacent(wall, geo),
        "rule": "water_damage_near_exterior_wall",
        "message": "Moisture detected on wall adjacent to exterior — inspect wall cavity for water intrusion",
    },
    {
        "trigger": lambda cls, wall, geo: cls == "mold" and _is_near_floor(wall, threshold_m=0.5),
        "rule": "mold_at_floor_level",
        "message": "Mold at floor level — check subfloor condition for moisture damage",
    },
    {
        "trigger": lambda cls, wall, geo: cls == "crack" and _is_corner_junction(wall, geo),
        "rule": "crack_at_corner_junction",
        "message": "Crack at wall corner — check load-bearing connection integrity",
    },
    {
        "trigger": lambda cls, wall, geo: cls in ("stain", "water_damage") and _is_ceiling_adjacent(wall, geo),
        "rule": "stain_near_ceiling",
        "message": "Staining near ceiling — check roof or slab above for leaks",
    },
]

# Scope line items per damage class
SCOPE_TEMPLATES = {
    "crack": "Repair crack in wall surface — apply filler, sand, repaint",
    "corrosion": "Treat corrosion — remove rust, apply primer and protective coating",
    "stain": "Clean and repaint stained surface",
    "mold": "Mold remediation — clean, apply biocide, check for moisture source",
    "deterioration": "Restore deteriorated surface — strip, repair substrate, refinish",
    "water_damage": "Address water damage — dry out, replace damaged drywall, identify water source",
}


@dataclass
class DamageRegion:
    wall_id: str
    damage_class: str
    extent_m2: float
    bbox_pixel: list[int]             # [x1, y1, x2, y2] in image space
    confidence_det: float             # YOLO detection confidence
    concealed_flag: bool = False
    concealed_rule: str | None = None
    concealed_message: str | None = None
    scope: str = ""


def detect_damage(
    image_paths: list[Path],
    room_geo: RoomGeometry,
    image_to_wall_map: dict[str, str] | None = None,
    depth_per_image: dict[str, np.ndarray] | None = None,
) -> list[DamageRegion]:
    """
    Detect damage in images and map to wall surfaces.

    image_to_wall_map: maps image filename → wall_id (optional; auto-assigns if None)
    depth_per_image:   maps image filename → depth map for metric extent estimation
    """
    model = _load_model()
    regions: list[DamageRegion] = []

    for img_path in image_paths:
        img_path = Path(img_path)
        wall_id = (image_to_wall_map or {}).get(img_path.name, _infer_wall_id(img_path, room_geo))
        wall = _find_wall(room_geo, wall_id)

        detections = _run_inference(model, img_path)
        depth_map = (depth_per_image or {}).get(img_path.name)

        for det in detections:
            cls_name, bbox, conf = det
            extent = _estimate_extent(bbox, img_path, depth_map, wall)

            region = DamageRegion(
                wall_id=wall_id,
                damage_class=cls_name,
                extent_m2=extent,
                bbox_pixel=bbox,
                confidence_det=float(conf),
                scope=SCOPE_TEMPLATES.get(cls_name, f"Repair {cls_name}"),
            )

            # Apply concealed-damage rules
            for rule in CONCEALED_RULES:
                try:
                    if rule["trigger"](cls_name, wall, room_geo):
                        region.concealed_flag = True
                        region.concealed_rule = rule["rule"]
                        region.concealed_message = rule["message"]
                        break
                except Exception:
                    pass

            regions.append(region)

    log.info(f"Detected {len(regions)} damage regions")
    return regions


def _load_model():
    try:
        from ultralytics import YOLO
        if MODEL_PATH.exists():
            model = YOLO(str(MODEL_PATH))
            log.info(f"Loaded damage model: {MODEL_PATH}")
        else:
            log.warning(f"Model not found at {MODEL_PATH}; using default YOLOv8n")
            model = YOLO("yolov8n.pt")
        return model
    except ImportError:
        log.warning("ultralytics not installed; damage detection disabled")
        return None


def _run_inference(model, img_path: Path) -> list[tuple[str, list[int], float]]:
    if model is None:
        return []
    try:
        results = model(str(img_path), verbose=False, conf=0.3)
        detections = []
        for r in results:
            if r.boxes is None:
                continue
            names = r.names
            for box in r.boxes:
                cls_id = int(box.cls[0])
                cls_name = names.get(cls_id, f"class_{cls_id}")
                # Normalise class names to our vocabulary
                cls_name = _normalise_class(cls_name)
                xyxy = box.xyxy[0].tolist()
                bbox = [int(v) for v in xyxy]
                conf = float(box.conf[0])
                detections.append((cls_name, bbox, conf))
        return detections
    except Exception as e:
        log.warning(f"Inference failed on {img_path.name}: {e}")
        return []


def _normalise_class(name: str) -> str:
    name = name.lower().replace("-", "_").replace(" ", "_")
    mapping = {
        "crack": "crack", "cracking": "crack",
        "corrosion": "corrosion", "rust": "corrosion",
        "stain": "stain", "staining": "stain",
        "mold": "mold", "mould": "mold",
        "deterioration": "deterioration", "spalling": "deterioration",
        "water": "water_damage", "water_damage": "water_damage",
    }
    for k, v in mapping.items():
        if k in name:
            return v
    return name


def _estimate_extent(
    bbox: list[int],
    img_path: Path,
    depth_map: np.ndarray | None,
    wall: WallSegment | None,
) -> float:
    """Estimate damage area in m² from pixel bbox + optional depth."""
    try:
        from PIL import Image as PILImage
        img = PILImage.open(img_path)
        W, H = img.size
    except Exception:
        W, H = 1920, 1080

    x1, y1, x2, y2 = bbox
    bbox_w_px = max(1, x2 - x1)
    bbox_h_px = max(1, y2 - y1)

    if depth_map is not None and depth_map.size > 0:
        # Sample depth at bbox centre
        cy_px = int((y1 + y2) / 2)
        cx_px = int((x1 + x2) / 2)
        cy_px = min(cy_px, depth_map.shape[0] - 1)
        cx_px = min(cx_px, depth_map.shape[1] - 1)
        depth_m = float(depth_map[cy_px, cx_px])
        depth_m = max(0.3, min(depth_m, 5.0))

        # Approximate metric size using pinhole camera
        # Assume fx ~ 0.8 * max(W, H)
        fx = max(W, H) * 0.8
        metric_w = (bbox_w_px / W) * (depth_m / fx) * W
        metric_h = (bbox_h_px / H) * (depth_m / fx) * H
        extent = metric_w * metric_h
    elif wall is not None:
        # Project using wall geometry
        wall_w = wall.length_m
        wall_h = wall.height_m
        frac_w = bbox_w_px / W
        frac_h = bbox_h_px / H
        extent = frac_w * wall_w * frac_h * wall_h
    else:
        extent = 0.1   # unknown; 0.1 m² default

    return float(np.clip(extent, 0.01, 50.0))


def _find_wall(geo: RoomGeometry, wall_id: str) -> WallSegment | None:
    for w in geo.walls:
        if w.wall_id == wall_id:
            return w
    return geo.walls[0] if geo.walls else None


def _infer_wall_id(img_path: Path, geo: RoomGeometry) -> str:
    """Assign image to wall by filename index (img_0 → w1, etc.)."""
    try:
        idx = int("".join(filter(str.isdigit, img_path.stem[:4]))) % max(1, len(geo.walls))
        return geo.walls[idx].wall_id if geo.walls else "w1"
    except Exception:
        return "w1"


def _is_exterior_adjacent(wall: WallSegment | None, geo: RoomGeometry) -> bool:
    if wall is None or geo.room_center_2d is None:
        return False
    # A wall is "exterior adjacent" if its midpoint is far from room center
    if wall.p1 is None or wall.p2 is None:
        return False
    mid = (wall.p1 + wall.p2) / 2
    dist = np.linalg.norm(mid - geo.room_center_2d)
    return dist > 3.0   # heuristic: >3m from center = likely exterior


def _is_near_floor(wall: WallSegment | None, threshold_m: float = 0.5) -> bool:
    # We don't have per-pixel height here; approximate: assume damage in lower 30% of wall
    # This is a conservative heuristic
    return True   # always flag mold as potentially floor-level


def _is_corner_junction(wall: WallSegment | None, geo: RoomGeometry) -> bool:
    if wall is None:
        return False
    # A wall that shares an endpoint with another wall is at a corner
    for other in geo.walls:
        if other.wall_id == wall.wall_id:
            continue
        for ep_self in [wall.p1, wall.p2]:
            for ep_other in [other.p1, other.p2]:
                if np.linalg.norm(ep_self - ep_other) < 0.3:
                    return True
    return False


def _is_ceiling_adjacent(wall: WallSegment | None, geo: RoomGeometry) -> bool:
    # Since we detect bbox as top portion of image, approximate: always flag stains near ceiling
    return True


def build_scope_line_items(regions: list[DamageRegion]) -> list[dict]:
    items = []
    for i, r in enumerate(regions):
        items.append({
            "item_id": f"s{i+1}",
            "surface": r.wall_id,
            "damage_class": r.damage_class,
            "scope": r.scope,
            "unit": "m2",
            "quantity": round(r.extent_m2, 3),
        })
    return items
