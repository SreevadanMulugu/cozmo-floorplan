#!/usr/bin/env python3
"""
cozmo-floorplan: Main pipeline entry point.

Usage:
    python run.py --input <path> --tier lidar|video|photo [options]

Options:
    --input PATH         Input file or directory (auto-detects tier if --tier omitted)
    --tier lidar|video|photo  Force tier (optional)
    --output DIR         Output directory (default: ./output/<capture_id>/)
    --capture-id ID      Custom capture ID (default: auto-generated)
    --no-drift           Disable drift correction (ablation mode)
    --no-damage          Skip damage detection
    --images PATH        Additional image folder for damage detection
    --quiet              Suppress progress output
"""

import argparse
import logging
import sys
import time
from pathlib import Path


def setup_logging(quiet: bool) -> None:
    level = logging.WARNING if quiet else logging.INFO
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        level=level,
    )
    # Suppress noisy third-party loggers
    for noisy in ("open3d", "PIL", "torch", "urllib3", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Floor plan pipeline: phone captures → dimensioned floor plan JSON + SVG"
    )
    parser.add_argument("--input", required=True, help="Input file or directory")
    parser.add_argument("--tier", choices=["lidar", "video", "photo"], default=None)
    parser.add_argument("--output", default=None, help="Output directory")
    parser.add_argument("--capture-id", default=None)
    parser.add_argument("--no-drift", action="store_true", help="Disable drift correction")
    parser.add_argument("--no-damage", action="store_true", help="Skip damage detection")
    parser.add_argument("--images", default=None, help="Images folder for damage detection")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    setup_logging(args.quiet)
    log = logging.getLogger("run")

    # --- Imports (after logging setup) ---
    from pipeline.input_parser import parse_input, detect_tier
    from pipeline.geometry import extract_geometry
    from pipeline.stitcher import stitch
    from pipeline.damage import detect_damage
    from pipeline.output import generate_output

    t_start = time.time()

    # --- Detect tier ---
    tier = args.tier or detect_tier(args.input)
    log.info(f"Tier: {tier}  Input: {args.input}")

    parsed = parse_input(args.input, tier)

    # --- Output directory ---
    import uuid
    capture_id = args.capture_id or str(uuid.uuid4())[:8]
    output_dir = Path(args.output) if args.output else Path("output") / capture_id
    work_dir = output_dir / "work"
    work_dir.mkdir(parents=True, exist_ok=True)

    # --- Reconstruction ---
    if tier == "lidar":
        from pipeline.reconstruction.lidar import reconstruct
        room_cloud = reconstruct(parsed)
        room_clouds = {"room_0": room_cloud}
        room_geos = {"room_0": extract_geometry(room_cloud)}

    elif tier == "video":
        from pipeline.reconstruction.video import reconstruct
        room_cloud = reconstruct(parsed, output_dir=work_dir)
        room_clouds = {"room_0": room_cloud}
        room_geos = {"room_0": extract_geometry(room_cloud)}

    elif tier == "photo":
        from pipeline.reconstruction.photo import reconstruct
        room_clouds = reconstruct(parsed, output_dir=work_dir)
        room_geos = {}
        for room_id, cloud in room_clouds.items():
            room_geos[room_id] = extract_geometry(cloud)

    # --- Stitch ---
    use_drift = not args.no_drift
    plan = stitch(room_geos, room_clouds, drift_correction=use_drift)

    # --- Damage detection ---
    damage_regions = []
    if not args.no_damage:
        image_dir = args.images or args.input
        img_path = Path(image_dir)
        image_paths = []
        if img_path.is_file() and img_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
            image_paths = [img_path]
        elif img_path.is_dir():
            image_paths = sorted(img_path.rglob("*.jpg")) + sorted(img_path.rglob("*.png"))

        if image_paths:
            # Use first room's geometry for damage mapping
            first_geo = list(plan.rooms.values())[0]
            damage_regions = detect_damage(image_paths[:20], first_geo)

    timing_ms = int((time.time() - t_start) * 1000)

    # --- Output ---
    n_images = sum(len(v) for v in parsed.photo_rooms.values()) if tier == "photo" else 4
    result = generate_output(
        plan=plan,
        tier=tier,
        damage_regions=damage_regions,
        capture_id=capture_id,
        n_images=n_images,
        timing_ms=timing_ms,
        output_dir=output_dir,
    )

    import json
    print(json.dumps(result, indent=2))
    log.info(f"Done in {timing_ms}ms  →  {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
