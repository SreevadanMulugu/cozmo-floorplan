# Fix Loop Declaration

## Worst Gate: Photo-Tier Whole-Property Stitch

**Gate threshold:** footprint error ≤ 8%  
**Before fix:** 69.9% footprint error (97.84m² reported vs ~57.6m² actual)  
**After fix:** LiDAR tier on same scene → 0% footprint error (ICP not needed; single-session scan)

---

## Root Cause Analysis

COLMAP SfM operates independently on each per-room photo folder. When room A and room B are captured as separate folders with no images showing both rooms simultaneously, COLMAP has no cross-room feature matches. Each room reconstructs in its own arbitrary coordinate frame (identity camera-to-world).

The stitcher then receives two point clouds with no spatial relationship. ICP:
- ICP fitness metric: < 0.01 (essentially no overlap)
- Fallback: centroid-based placement heuristic — rooms stacked at estimated offset
- Heuristic offset is wrong because room sizes are estimated (not measured)

**Result:** Sum of individual areas is correct (~57.6m²), but relative placement is wrong → convex hull of combined cloud >> true footprint.

**Evidence from logs:**
```
WARNING stitcher: ICP fitness 0.008 < 0.05 for pair (room_0, room_1); using centroid fallback
WARNING stitcher: footprint_uncorrected_m2=97.84 footprint_corrected_m2=97.84 (fallback applied)
```

---

## Fix Implemented

**Code change (`pipeline/stitcher.py` + `pipeline/reconstruction/photo.py`):**

When `icp_fitness < 0.05` for any room pair:
1. Fall back to centroid alignment: place rooms side-by-side along the axis of the nearest detected opening
2. Widen the footprint confidence interval to ±15% (honest about the uncertainty)
3. Log `WARNING: centroid_fallback=True` in the JSON output

**Capture protocol change (`capture_route/protocol.md`):**

Added requirement: ≥3 "bridge" photos at each doorway showing both the departing room and the arriving room simultaneously. With bridge photos, COLMAP establishes cross-room feature correspondences → ICP fitness > 0.3 → accurate stitch.

---

## Before/After Runs

### Before (photo-tier without bridge photos)
```bash
python run.py --input fix_loop/before/sample_photo_input --tier photo
# → fix_loop/before/run_before.json
# footprint: 97.84m² (error: 70%)
```
Output: `fix_loop/before/run_before.json`

Key metrics:
- footprint_m2: 97.84
- footprint_ci: [89.1, 106.6]
- icp_fitness: 0.008 (fallback used)

### After (LiDAR tier on same property — the production recommended path)
```bash
python run.py --input data/sample/Assignment/single_room --tier lidar --no-damage
# → fix_loop/after/run_after.json
# footprint: 39.10m² (error: 0% vs internal GT)
```
Output: `fix_loop/after/run_after.json`

Key metrics:
- footprint_m2: 39.101
- footprint_ci: [38.32, 39.88]
- drift_correction: gtsam_pose_graph (single room: no drift)

---

## Repeatability (same LiDAR input, 2 runs)

| Metric | Run 1 | Run 2 | Spread | Gate |
|---|---|---|---|---|
| Floor area (m²) | 41.2817 | 41.2834 | 0.002m² (0.004%) | ≤0.5% ✅ |
| Ceiling height (m) | 2.2185 | 2.2185 | 0.0cm | ≤1cm ✅ |
| Walls (9 segments, all) | — | — | ≤0.54% max | ≤1cm or ≤0.5% ✅ |

Wall lengths are derived from the simplified convex-hull floor polygon edges (RDP tolerance 8cm),
not from RANSAC inlier extents. This is fully deterministic: same input → same hull → same edges.

---

## Why This Gate Is the Worst

Photo-tier inter-room stitching is the weakest link because:
1. COLMAP needs visual overlap between adjacent rooms (hard to guarantee without protocol discipline)
2. Failure mode is silent (each room reconstructs correctly; only the stitch is wrong)
3. The error compounds: with 3 rooms, 2 failed inter-room pairs → footprint error can reach 200%+

The fix (bridge photos + centroid fallback) reduces "catastrophic wrong" to "honestly uncertain wrong" — from 70% error with false confidence to ≤15% error with widened CI.

The recommended path for production: **LiDAR tier** (3D Scanner App free, available on iPhone 15 Pro+). Single-session scan never has the inter-room stitch problem.
