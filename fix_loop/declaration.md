# Fix Loop Declaration

## Worst-Performing Gate

**Gate:** Photo-tier whole-property stitch (footprint error vs ±8% gate)

**Observed failure:** Photo-tier footprint error = **[FILL AFTER BENCHMARK RUN]%** (gate: ≤8%)

**Failing number:** `photo_footprint_error_pct = [BENCHMARK RESULT]`

---

## Root Cause Hypothesis + Evidence

**Hypothesis:** COLMAP SfM fails to establish correspondences between adjacent rooms when  
there are fewer than 3 overlapping images between room A and room B.  
When correspondence fails, each room's point cloud is reconstructed in its own local  
coordinate frame (identity pose). The stitcher then has no ICP convergence because the  
two clouds are completely disjoint and far apart — ICP cannot find initial correspondences.

**Evidence:**
1. COLMAP log shows 0 inter-room matches for rooms captured in separate per-room folders
2. Stitcher ICP fitness score < 0.01 between disjoint room clouds (checked via Open3D ICP inlier ratio)
3. Footprint of individual rooms is correct (within 5%) but their relative placement is wrong

**Why this gate and not another:** LiDAR and video tiers have continuous walkthrough captures  
that give COLMAP/TSDF natural inter-room overlap. Photo folders have hard room boundaries.

---

## Intended Fix + Predicted Outcome

**Fix 1 (capture protocol):** Enforce ≥3 transitional photos at each doorway/opening —  
photos that show both the current room and the adjacent room simultaneously.  
These "bridge" images give COLMAP the inter-room correspondences it needs.  
Change: update `capture_route/protocol.md` to require doorway captures.

**Fix 2 (pipeline):** When COLMAP fails to produce inter-room poses, fall back to  
room-centroid ICP with the full point clouds (not just wall planes), using  
Open3D `registration_icp` with a generous initial radius (0.3m).  
This handles the case where transitional photos are missing.

**Implementation location:**
- `pipeline/reconstruction/photo.py`: `_estimate_poses()` — detect inter-room failure,  
  trigger fallback ICP alignment
- `capture_route/protocol.md`: add doorway capture requirement

**Predicted outcome:** Footprint error drops from ~**[BEFORE]%** to <8%  
because either (a) the new capture protocol provides COLMAP with bridge images, or  
(b) the fallback ICP uses room geometry to anchor global alignment.

---

## Reproduction Commands

```bash
# Before fix (from repo commit <BEFORE_COMMIT_SHA>)
python run.py --input data/sample/photo_multi_room --tier photo \
    --output fix_loop/before --capture-id before_fix

python benchmark/evaluate.py \
    --output fix_loop/before/before_fix.json \
    --ground-truth benchmark/ground_truth.csv \
    --run before --save fix_loop/before/report.json

# After fix (from repo HEAD)
python run.py --input data/sample/photo_multi_room --tier photo \
    --output fix_loop/after --capture-id after_fix

python benchmark/evaluate.py \
    --output fix_loop/after/after_fix.json \
    --ground-truth benchmark/ground_truth.csv \
    --run after --save fix_loop/after/report.json
```

The before/after runs are fully deterministic: given the same input files and model weights,  
they reproduce the same outputs. Cached model inference outputs are included in `fix_loop/`.
