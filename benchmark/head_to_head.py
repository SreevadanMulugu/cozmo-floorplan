#!/usr/bin/env python3
"""
Head-to-head comparison: our pipeline vs MagicPlan (or any incumbent).

Usage:
    python benchmark/head_to_head.py \
        --ours output/run1/capture.json \
        --incumbent incumbent/magicplan_measurements.csv \
        --ground-truth benchmark/ground_truth.csv

Incumbent CSV format (manually read from MagicPlan PDF export):
    type,id,value_m
    wall,w1,4.55
    wall,w2,3.08
    ceiling,room_0,2.76
    opening,op1,0.92

Gate: beat or tie on ≥ 70% of shared dimensions.
"""

import argparse
import csv
import json
import sys


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def load_measurements(path: str) -> dict[str, dict[str, float]]:
    m: dict[str, dict] = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = row["type"].strip().lower()
            id_ = row["id"].strip()
            val = float(row["value_m"].strip())
            m.setdefault(t, {})[id_] = val
    return m


def extract_our_measurements(output: dict) -> dict[str, dict[str, float]]:
    m: dict[str, dict] = {}
    for room in output.get("rooms", []):
        room_id = room["room_id"]
        m.setdefault("ceiling", {})[room_id] = room.get("ceiling_height_m", 0)
        m.setdefault("area", {})[room_id] = room.get("floor_area_m2", 0)
        for w in room.get("walls", []):
            m.setdefault("wall", {})[w["wall_id"]] = w["length_m"]
        for op in room.get("openings", []):
            key = f"{op['wall_id']}_{op['type']}"
            m.setdefault("opening", {})[key] = op["width_m"]
    mp = output.get("multi_room_plan", {})
    if "footprint_m2" in mp:
        m.setdefault("footprint", {})[""] = mp["footprint_m2"]
    return m


def compare(ours: dict, incumbent: dict, gt: dict) -> dict:
    report = {
        "dimensions": [],
        "our_wins": 0,
        "ties": 0,
        "incumbent_wins": 0,
        "total_shared": 0,
        "our_win_pct": 0.0,
        "gate_status": "FAIL",
    }

    for dim_type in set(gt) | set(ours) | set(incumbent):
        gt_vals = gt.get(dim_type, {})
        our_vals = ours.get(dim_type, {})
        inc_vals = incumbent.get(dim_type, {})

        shared_ids = set(gt_vals) & set(our_vals) & set(inc_vals)
        for id_ in sorted(shared_ids):
            gt_v = gt_vals[id_]
            our_v = our_vals[id_]
            inc_v = inc_vals[id_]

            our_err = abs(our_v - gt_v)
            inc_err = abs(inc_v - gt_v)

            if our_err < inc_err - 1e-4:
                winner = "ours"
                report["our_wins"] += 1
            elif inc_err < our_err - 1e-4:
                winner = "incumbent"
                report["incumbent_wins"] += 1
            else:
                winner = "tie"
                report["ties"] += 1

            report["dimensions"].append({
                "type": dim_type,
                "id": id_,
                "gt": round(gt_v, 4),
                "ours": round(our_v, 4),
                "incumbent": round(inc_v, 4),
                "our_error_m": round(our_err, 4),
                "incumbent_error_m": round(inc_err, 4),
                "winner": winner,
            })
            report["total_shared"] += 1

    total = report["total_shared"]
    if total > 0:
        win_or_tie = report["our_wins"] + report["ties"]
        win_pct = win_or_tie / total
        report["our_win_pct"] = round(win_pct * 100, 1)
        report["gate_status"] = "PASS" if win_pct >= 0.70 else "FAIL"

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Head-to-head comparison vs incumbent")
    parser.add_argument("--ours", required=True)
    parser.add_argument("--incumbent", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--save", default="head_to_head_report.json")
    args = parser.parse_args()

    ours_json = load_json(args.ours)
    our_measurements = extract_our_measurements(ours_json)
    incumbent = load_measurements(args.incumbent)
    gt = load_measurements(args.ground_truth)

    report = compare(our_measurements, incumbent, gt)

    print(json.dumps(report, indent=2))
    with open(args.save, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nOur win rate: {report['our_win_pct']}%  Gate: {report['gate_status']}", file=sys.stderr)
    print(f"Ours: {report['our_wins']} wins, {report['ties']} ties, {report['incumbent_wins']} losses", file=sys.stderr)
    return 0 if report["gate_status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
