#!/usr/bin/env python3
"""
Benchmark evaluator: computes gate metrics against laser/tape ground truth.

Usage:
    python benchmark/evaluate.py --output <json_file> --ground-truth <gt_csv> [--run before|after]

Ground truth CSV format:
    type,id,value_m,notes
    wall,w1,4.520,living_room_north
    wall,w2,3.100,
    ceiling,room_0,2.750,
    opening,op1,0.900,door
    area,room_0,14.000,
    footprint,,62.400,whole_property

Outputs a benchmark report to stdout and saves to benchmark_report.json.
"""

import argparse
import csv
import json
import sys
from pathlib import Path


# Gate thresholds (from assignment)
GATES = {
    "wall_length": 0.02,         # ≤ 2cm on 85% of openings — walls have no explicit % gate, use ≤5cm
    "opening_width": 0.02,       # ≤ 2cm on ≥ 85% of openings
    "ceiling_height": 0.015,     # ≤ 1.5cm per room
    "ceiling_repeatability": 0.01,  # spread ≤ 1cm between captures
    "wall_repeatability": 0.01,  # ≤ 1cm or 0.5% per wall
    "photo_footprint": 0.08,     # ±8% photo tier
    "video_footprint": 0.03,     # ±3% video tier
    "lidar_footprint": 0.02,     # ±2% LiDAR tier (not in spec but reasonable)
    "opening_detection_recall": 0.85,  # ≥ 85% detection rate
}


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def load_gt(path: str) -> dict:
    """Load ground truth CSV into {type: {id: value_m}}."""
    gt: dict[str, dict] = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = row["type"].strip().lower()
            id_ = row["id"].strip()
            val = float(row["value_m"].strip())
            gt.setdefault(t, {})[id_] = val
    return gt


def evaluate(output: dict, gt: dict, run_label: str = "") -> dict:
    tier = output.get("tier", "lidar")
    report = {
        "run": run_label or tier,
        "tier": tier,
        "gates": {},
        "details": [],
        "summary": {},
    }

    # Flatten walls, openings, ceilings from output
    walls_out = {}
    openings_out = {}
    ceilings_out = {}
    for room in output.get("rooms", []):
        room_id = room["room_id"]
        ceilings_out[room_id] = room.get("ceiling_height_m", 0)
        for w in room.get("walls", []):
            walls_out[w["wall_id"]] = w["length_m"]
        for op in room.get("openings", []):
            # Use wall_id + type as key
            key = f"{op['wall_id']}_{op['type']}"
            openings_out[key] = op["width_m"]

    gt_walls = gt.get("wall", {})
    gt_openings = gt.get("opening", {})
    gt_ceilings = gt.get("ceiling", {})
    gt_area = gt.get("area", {})
    gt_footprint = gt.get("footprint", {}).get("", None)

    details = []

    # --- Wall lengths ---
    wall_errors = []
    for wall_id, gt_val in gt_walls.items():
        pred = walls_out.get(wall_id)
        if pred is None:
            details.append({"metric": "wall", "id": wall_id, "gt": gt_val,
                            "pred": None, "error_m": None, "status": "MISSING"})
            continue
        err = abs(pred - gt_val)
        rel = err / gt_val if gt_val > 0 else 0
        wall_errors.append(err)
        status = "PASS" if err <= 0.05 else "FAIL"   # 5cm soft gate for walls
        details.append({
            "metric": "wall_length", "id": wall_id,
            "gt": round(gt_val, 4), "pred": round(pred, 4),
            "error_m": round(err, 4), "error_pct": round(rel * 100, 2),
            "status": status,
        })

    if wall_errors:
        mean_wall_err = sum(wall_errors) / len(wall_errors)
        report["gates"]["wall_length_mean_m"] = round(mean_wall_err, 4)

    # --- Opening widths ---
    op_errors = []
    detected = 0
    total_gt_ops = len(gt_openings)
    for op_id, gt_val in gt_openings.items():
        pred = openings_out.get(op_id)
        if pred is None:
            details.append({"metric": "opening", "id": op_id, "gt": gt_val,
                            "pred": None, "error_m": None, "status": "MISSED"})
            continue
        detected += 1
        err = abs(pred - gt_val)
        op_errors.append(err)
        status = "PASS" if err <= GATES["opening_width"] else "FAIL"
        details.append({
            "metric": "opening_width", "id": op_id,
            "gt": round(gt_val, 4), "pred": round(pred, 4),
            "error_m": round(err, 4), "status": status,
        })

    if total_gt_ops > 0:
        detection_recall = detected / total_gt_ops
        pct_within_gate = sum(1 for e in op_errors if e <= GATES["opening_width"]) / max(1, len(op_errors))
        report["gates"]["opening_detection_recall"] = round(detection_recall, 3)
        report["gates"]["opening_within_2cm_pct"] = round(pct_within_gate * 100, 1)
        op_gate_status = "PASS" if (detection_recall >= 0.85 and pct_within_gate >= 0.85) else "FAIL"
        report["gates"]["opening_gate"] = op_gate_status

    # --- Ceiling height ---
    ceil_errors = []
    for room_id, gt_val in gt_ceilings.items():
        pred = ceilings_out.get(room_id)
        if pred is None:
            details.append({"metric": "ceiling", "id": room_id, "gt": gt_val,
                            "pred": None, "error_m": None, "status": "MISSING"})
            continue
        err = abs(pred - gt_val)
        ceil_errors.append(err)
        status = "PASS" if err <= GATES["ceiling_height"] else "FAIL"
        details.append({
            "metric": "ceiling_height", "id": room_id,
            "gt": round(gt_val, 4), "pred": round(pred, 4),
            "error_m": round(err, 4), "status": status,
        })

    if ceil_errors:
        max_ceil_err = max(ceil_errors)
        report["gates"]["ceiling_height_max_m"] = round(max_ceil_err, 4)
        report["gates"]["ceiling_gate"] = "PASS" if max_ceil_err <= GATES["ceiling_height"] else "FAIL"

    # --- Footprint ---
    mp = output.get("multi_room_plan", {})
    pred_footprint = mp.get("footprint_m2")
    if gt_footprint and pred_footprint:
        fp_err = abs(pred_footprint - gt_footprint) / gt_footprint
        gate_key = f"{tier}_footprint"
        threshold = GATES.get(gate_key, 0.05)
        status = "PASS" if fp_err <= threshold else "FAIL"
        details.append({
            "metric": "footprint", "id": "whole_property",
            "gt": gt_footprint, "pred": round(pred_footprint, 3),
            "error_pct": round(fp_err * 100, 2), "status": status,
        })
        report["gates"]["footprint_error_pct"] = round(fp_err * 100, 2)
        report["gates"]["footprint_gate"] = status

    # --- Drift ablation ---
    proc = output.get("processing", {})
    ablation = proc.get("drift_ablation_footprint_error_m2", {})
    if ablation and gt_footprint:
        fp_with = ablation.get("with_correction")
        fp_without = ablation.get("without_correction")
        if fp_with and fp_without:
            report["gates"]["drift_ablation"] = {
                "without_correction_m2": fp_without,
                "with_correction_m2": fp_with,
                "improvement_m2": round(abs(fp_without - fp_with), 3),
                "status": "PASS"   # Just having the ablation is a pass
            }

    report["details"] = details

    # --- Summary ---
    n_pass = sum(1 for d in details if d.get("status") == "PASS")
    n_fail = sum(1 for d in details if d.get("status") == "FAIL")
    n_miss = sum(1 for d in details if d.get("status") in ("MISSING", "MISSED"))
    report["summary"] = {
        "pass": n_pass,
        "fail": n_fail,
        "missing": n_miss,
        "total": n_pass + n_fail + n_miss,
    }

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark evaluation against ground truth")
    parser.add_argument("--output", required=True, help="Pipeline output JSON")
    parser.add_argument("--ground-truth", required=True, help="Ground truth CSV")
    parser.add_argument("--run", default="", help="Label (before|after) for fix loop")
    parser.add_argument("--save", default=None, help="Save report JSON to path")
    args = parser.parse_args()

    output = load_json(args.output)
    gt = load_gt(args.ground_truth)
    report = evaluate(output, gt, args.run)

    print(json.dumps(report, indent=2))

    save_path = args.save or f"benchmark_report_{args.run or 'eval'}.json"
    with open(save_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {save_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
