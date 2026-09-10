"""
Output generator: produces the full JSON output contract + SVG floor plan.

JSON schema per capture:
  capture_id, tier, rooms (walls, openings, damage), multi_room_plan,
  scope_line_items, processing metadata

SVG: dimensioned floor plan with room labels and opening markers.
"""

from __future__ import annotations
import json
import logging
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pipeline.geometry import RoomGeometry
from pipeline.stitcher import StitchedPlan
from pipeline.damage import DamageRegion, build_scope_line_items
from pipeline.confidence import (
    wall_length_ci, ceiling_height_ci, floor_area_ci,
    opening_width_ci, damage_extent_ci, footprint_ci,
    Tier,
)

log = logging.getLogger(__name__)


def generate_output(
    plan: StitchedPlan,
    tier: Tier,
    damage_regions: list[DamageRegion],
    capture_id: str | None = None,
    n_images: int = 4,
    timing_ms: int = 0,
    output_dir: Path | None = None,
) -> dict:
    """Build the full JSON output contract and write to disk."""
    if capture_id is None:
        capture_id = str(uuid.uuid4())[:8]

    out = {
        "capture_id": capture_id,
        "tier": tier,
        "rooms": [],
        "multi_room_plan": {},
        "scope_line_items": [],
        "processing": {},
    }

    # --- Rooms ---
    for room_id, geo in plan.rooms.items():
        room_dict = _room_to_dict(room_id, geo, tier, n_images, damage_regions)
        out["rooms"].append(room_dict)

    # --- Multi-room plan ---
    global_poly = plan.global_polygon
    out["multi_room_plan"] = {
        "footprint_m2": round(plan.footprint_m2, 3),
        "footprint_ci": footprint_ci(plan.footprint_m2, tier).to_list(),
        "adjacency": [
            {"room_a": a, "room_b": b, "connection_type": c}
            for a, b, c in plan.adjacency
        ],
        "global_polygon": (
            [[round(x, 3), round(z, 3)] for x, z in global_poly.tolist()]
            if global_poly is not None else []
        ),
    }

    # --- Scope line items ---
    out["scope_line_items"] = build_scope_line_items(damage_regions)

    # --- Processing metadata ---
    out["processing"] = {
        "drift_correction_method": (
            "gtsam_pose_graph_floor_ceiling_constrained"
            if plan.drift_correction_used else "none"
        ),
        "drift_ablation_footprint_error_m2": {
            "without_correction": round(plan.footprint_uncorrected_m2, 3),
            "with_correction": round(plan.footprint_corrected_m2, 3),
        },
        "timing_ms": timing_ms,
    }

    # --- Write files ---
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / f"{capture_id}.json"
        with open(json_path, "w") as f:
            json.dump(out, f, indent=2)
        log.info(f"JSON written to {json_path}")

        svg_path = output_dir / f"{capture_id}.svg"
        render_svg(plan, svg_path)
        log.info(f"SVG written to {svg_path}")

    return out


def _room_to_dict(
    room_id: str,
    geo: RoomGeometry,
    tier: Tier,
    n_images: int,
    damage_regions: list[DamageRegion],
) -> dict:
    walls = []
    for w in geo.walls:
        wci = wall_length_ci(w.length_m, tier, n_images)
        walls.append({
            "wall_id": w.wall_id,
            "length_m": round(w.length_m, 4),
            "length_ci": wci.to_list(),
            "height_m": round(w.height_m, 4),
            "p1_xz": [round(w.p1[0], 4), round(w.p1[1], 4)],
            "p2_xz": [round(w.p2[0], 4), round(w.p2[1], 4)],
        })

    openings = []
    for op in geo.openings:
        oci = opening_width_ci(op.width_m, tier)
        openings.append({
            "type": op.type,
            "wall_id": op.wall_id,
            "center_xz": [round(op.p_center[0], 4), round(op.p_center[1], 4)],
            "width_m": round(op.width_m, 4),
            "width_ci": oci.to_list(),
        })

    hci = ceiling_height_ci(geo.ceiling_height_m, tier)
    aci = floor_area_ci(geo.floor_area_m2, tier, n_images)

    damage_for_room = [r for r in damage_regions if r.wall_id.startswith("w")]
    damage_list = []
    for dr in damage_for_room:
        dci = damage_extent_ci(dr.extent_m2)
        damage_list.append({
            "wall_id": dr.wall_id,
            "class": dr.damage_class,
            "extent_m2": round(dr.extent_m2, 4),
            "extent_ci": dci.to_list(),
            "detection_confidence": round(dr.confidence_det, 3),
            "concealed_flag": dr.concealed_flag,
            "concealed_rule": dr.concealed_rule,
            "concealed_message": dr.concealed_message,
        })

    poly = None
    if geo.floor_polygon_2d is not None:
        poly = [[round(x, 4), round(z, 4)] for x, z in geo.floor_polygon_2d.tolist()]

    return {
        "room_id": room_id,
        "floor_area_m2": round(geo.floor_area_m2, 4),
        "floor_area_ci": aci.to_list(),
        "floor_polygon_xz": poly or [],
        "ceiling_height_m": round(geo.ceiling_height_m, 4),
        "ceiling_height_ci": hci.to_list(),
        "walls": walls,
        "openings": openings,
        "damage_regions": damage_list,
    }


def render_svg(plan: StitchedPlan, output_path: Path) -> None:
    """Render a dimensioned SVG floor plan from the stitched plan."""
    try:
        import svgwrite
    except ImportError:
        log.warning("svgwrite not installed; skipping SVG output")
        _render_matplotlib(plan, Path(str(output_path).replace(".svg", ".png")))
        return

    # Collect all wall endpoints to determine bounding box
    all_pts = []
    for geo in plan.rooms.values():
        for w in geo.walls:
            all_pts.extend([w.p1, w.p2])
        if geo.floor_polygon_2d is not None:
            all_pts.append(geo.floor_polygon_2d.mean(axis=0))

    if not all_pts:
        log.warning("No geometry to render")
        return

    pts = np.array(all_pts)
    xmin, zmin = pts.min(axis=0) - 0.5
    xmax, zmax = pts.max(axis=0) + 0.5
    scale = 100   # pixels per metre

    W = int((xmax - xmin) * scale) + 40
    H = int((zmax - zmin) * scale) + 40
    margin = 20

    dwg = svgwrite.Drawing(str(output_path), size=(W, H))
    dwg.add(dwg.rect(insert=(0, 0), size=(W, H), fill="white"))

    def to_svg(x, z):
        sx = (x - xmin) * scale + margin
        sz = (z - zmin) * scale + margin
        return (sx, sz)

    # Draw floor polygons (filled)
    colors = ["#e8f4f8", "#f0ede8", "#f8f4e8", "#e8f8ec", "#f4e8f8"]
    for i, (room_id, geo) in enumerate(plan.rooms.items()):
        color = colors[i % len(colors)]
        if geo.floor_polygon_2d is not None and len(geo.floor_polygon_2d) >= 3:
            pts_svg = [to_svg(x, z) for x, z in geo.floor_polygon_2d]
            dwg.add(dwg.polygon(points=pts_svg, fill=color, stroke="#aaa", stroke_width=1))

    # Draw walls
    for geo in plan.rooms.values():
        for wall in geo.walls:
            p1 = to_svg(*wall.p1)
            p2 = to_svg(*wall.p2)
            dwg.add(dwg.line(start=p1, end=p2, stroke="#333", stroke_width=4, stroke_linecap="round"))

            # Dimension label
            mid = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
            dwg.add(dwg.text(
                f"{wall.length_m:.2f}m",
                insert=(mid[0] + 4, mid[1] - 4),
                font_size=10, fill="#555",
                font_family="monospace",
            ))

    # Draw openings as gaps in walls
    for geo in plan.rooms.values():
        for op in geo.openings:
            c = to_svg(*op.p_center)
            half_w = op.width_m * scale / 2
            color = "#4CAF50" if op.type == "door" else "#2196F3"
            dwg.add(dwg.rect(
                insert=(c[0] - half_w, c[1] - 3),
                size=(half_w * 2, 6),
                fill=color, opacity=0.7,
            ))

    # Room labels
    for i, (room_id, geo) in enumerate(plan.rooms.items()):
        if geo.room_center_2d is not None:
            c = to_svg(*geo.room_center_2d)
            dwg.add(dwg.text(
                room_id,
                insert=c,
                font_size=12, fill="#333",
                font_family="sans-serif",
                text_anchor="middle",
            ))
            dwg.add(dwg.text(
                f"{geo.floor_area_m2:.1f}m²  h={geo.ceiling_height_m:.2f}m",
                insert=(c[0], c[1] + 14),
                font_size=9, fill="#666",
                font_family="monospace",
                text_anchor="middle",
            ))

    dwg.save()


def _render_matplotlib(plan: StitchedPlan, output_path: Path) -> None:
    """Fallback renderer using matplotlib."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.patches import Polygon
        from matplotlib.collections import PatchCollection

        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        ax.set_aspect("equal")

        colors = ["#e8f4f8", "#f0ede8", "#f8f4e8", "#e8f8ec"]
        for i, (room_id, geo) in enumerate(plan.rooms.items()):
            c = colors[i % len(colors)]
            if geo.floor_polygon_2d is not None and len(geo.floor_polygon_2d) >= 3:
                poly = Polygon(geo.floor_polygon_2d, closed=True, facecolor=c, edgecolor="#aaa", lw=1)
                ax.add_patch(poly)

        for geo in plan.rooms.values():
            for wall in geo.walls:
                ax.plot([wall.p1[0], wall.p2[0]], [wall.p1[1], wall.p2[1]],
                        "k-", lw=3)
                mid = (wall.p1 + wall.p2) / 2
                ax.text(mid[0], mid[1], f"{wall.length_m:.2f}m", fontsize=7, color="#555")

        ax.autoscale_view()
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Z (m)")
        ax.set_title("Floor Plan")
        plt.tight_layout()
        plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
        plt.close()
        log.info(f"PNG plan saved to {output_path}")
    except Exception as e:
        log.warning(f"Matplotlib render failed: {e}")
