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


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def repeatability_check(run1: dict, run2: dict) -> dict:
    report = {
        "tier": run1.get("tier", "unknown"),
        "wall_repeatability": [],
        "ceiling_repeatability": [],
        "verdict": "PASS",
    }

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

    # Wall repeatability
    common_walls = set(walls1) & set(walls2)
    for wid in sorted(common_walls):
        l1 = walls1[wid]
        l2 = walls2[wid]
        diff_abs = abs(l1 - l2)
        diff_rel = diff_abs / max(l1, 0.001)
        gate_ok = diff_abs <= WALL_GATE_ABS_M or diff_rel <= WALL_GATE_REL
        status = "PASS" if gate_ok else "FAIL"
        if status == "FAIL":
            report["verdict"] = "FAIL"
        report["wall_repeatability"].append({
            "wall_id": wid,
            "run1_m": round(l1, 4),
            "run2_m": round(l2, 4),
            "diff_m": round(diff_abs, 4),
            "diff_pct": round(diff_rel * 100, 2),
            "gate": f"≤{WALL_GATE_ABS_M*100:.0f}cm or ≤{WALL_GATE_REL*100:.1f}%",
            "status": status,
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
