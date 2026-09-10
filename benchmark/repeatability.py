#!/usr/bin/env python3
"""
Repeatability tester: compares two pipeline runs on the same room.

Usage:
    python benchmark/repeatability.py --run1 output/run1/capture.json --run2 output/run2/capture.json

Checks:
  - Wall length agreement: ≤ 1cm or 0.5% per wall (gate from assignment)
  - Ceiling height spread: ≤ 1cm between captures
  - Floor area spread

Outputs a repeatability table and PASS/FAIL verdict.
"""

import argparse
import json
import sys


WALL_GATE_ABS_M = 0.01      # 1cm
WALL_GATE_REL = 0.005       # 0.5%
CEILING_GATE_M = 0.010      # 1cm
AREA_GATE_REL = 0.005       # 0.5% floor area


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def repeatability_check(run1: dict, run2: dict) -> dict:
    report = {
        "tier": run1.get("tier", "unknown"),
        "area_repeatability": [],
        "ceiling_repeatability": [],
        "wall_repeatability": [],
        "verdict": "PASS",
    }

    # Floor area repeatability (most stable metric)
    for room in run1.get("rooms", []):
        rid = room["room_id"]
        a1 = room.get("floor_area_m2", 0)
        # Find matching room in run2
        a2 = next(
            (r["floor_area_m2"] for r in run2.get("rooms", []) if r["room_id"] == rid),
            None
        )
        if a2 is None:
            continue
        diff = abs(a1 - a2)
        diff_rel = diff / max(a1, 0.001)
        gate_ok = diff_rel <= AREA_GATE_REL
        status = "PASS" if gate_ok else "FAIL"
        if status == "FAIL":
            report["verdict"] = "FAIL"
        report["area_repeatability"].append({
            "room_id": rid,
            "run1_m2": round(a1, 4),
            "run2_m2": round(a2, 4),
            "diff_m2": round(diff, 4),
            "diff_pct": round(diff_rel * 100, 3),
            "gate": f"≤{AREA_GATE_REL*100:.1f}%",
            "status": status,
        })

    # Index walls and ceilings by room+wall_id
    def index_walls(output):
        idx = {}
        for room in output.get("rooms", []):
            for w in room.get("walls", []):
                idx[w["wall_id"]] = w["length_m"]
        return idx

    def index_ceilings(output):
        idx = {}
        for room in output.get("rooms", []):
            idx[room["room_id"]] = room.get("ceiling_height_m", 0)
        return idx

    walls1 = index_walls(run1)
    walls2 = index_walls(run2)
    ceil1 = index_ceilings(run1)
    ceil2 = index_ceilings(run2)

    # Wall repeatability — match by nearest length (RANSAC IDs are non-deterministic)
    # Sort both lists by length and match 1-to-1 in order
    sorted1 = sorted(walls1.values())
    sorted2 = sorted(walls2.values())
    n_match = min(len(sorted1), len(sorted2))
    for i in range(n_match):
        l1 = sorted1[i]
        l2 = sorted2[i]
        diff_abs = abs(l1 - l2)
        diff_rel = diff_abs / max(l1, 0.001)
        gate_ok = diff_abs <= WALL_GATE_ABS_M or diff_rel <= WALL_GATE_REL
        status = "PASS" if gate_ok else "FAIL"
        if status == "FAIL":
            report["verdict"] = "FAIL"
        report["wall_repeatability"].append({
            "wall_rank": i + 1,
            "run1_m": round(l1, 4),
            "run2_m": round(l2, 4),
            "diff_m": round(diff_abs, 4),
            "diff_pct": round(diff_rel * 100, 2),
            "gate": f"≤{WALL_GATE_ABS_M*100:.0f}cm or ≤{WALL_GATE_REL*100:.1f}%",
            "status": status,
            "note": "matched by length rank (RANSAC IDs non-deterministic)",
        })

    # Ceiling repeatability
    common_rooms = set(ceil1) & set(ceil2)
    for rid in sorted(common_rooms):
        h1 = ceil1[rid]
        h2 = ceil2[rid]
        diff = abs(h1 - h2)
        status = "PASS" if diff <= CEILING_GATE_M else "FAIL"
        if status == "FAIL":
            report["verdict"] = "FAIL"
        report["ceiling_repeatability"].append({
            "room_id": rid,
            "run1_m": round(h1, 4),
            "run2_m": round(h2, 4),
            "diff_m": round(diff, 4),
            "gate": f"≤{CEILING_GATE_M*100:.0f}cm",
            "status": status,
        })

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Repeatability test for two pipeline runs")
    parser.add_argument("--run1", required=True)
    parser.add_argument("--run2", required=True)
    parser.add_argument("--save", default="repeatability_report.json")
    args = parser.parse_args()

    run1 = load_json(args.run1)
    run2 = load_json(args.run2)
    report = repeatability_check(run1, run2)

    print(json.dumps(report, indent=2))
    with open(args.save, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {args.save}", file=sys.stderr)
    print(f"Verdict: {report['verdict']}", file=sys.stderr)
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
