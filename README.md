# cozmo-floorplan

Phone captures → dimensioned, stitched floor plans with confidence intervals.

Three input tiers: LiDAR · Video · Photos. One command per capture. JSON + SVG output.

---

## Cold Run (15 minutes on a clean machine)

```bash
# 1. Clone and install
git clone <this-repo>
cd cozmo-floorplan
pip install -r requirements.txt
pip install gtsam   # platform-specific wheel; or: conda install -c conda-forge gtsam

# 2. Download model weights
chmod +x models/download_models.sh
./models/download_models.sh

# 3. (For video/photo tiers) Clone Depth Anything V2
git clone https://github.com/DepthAnything/Depth-Anything-V2 ../Depth-Anything-V2
pip install -r ../Depth-Anything-V2/requirements.txt

# 4. Place sample data in data/sample/
#    LiDAR: data/sample/lidar/*.ply
#    Video: data/sample/video/walkthrough.mov
#    Photos: data/sample/photo/room_0/*.jpg, room_1/*.jpg, ...

# 5. Run
python run.py --input data/sample/lidar --tier lidar
python run.py --input data/sample/video/walkthrough.mov --tier video
python run.py --input data/sample/photo --tier photo
```

Output in `output/<capture_id>/`:
- `<id>.json` — full output contract (walls, openings, damage, confidence intervals)
- `<id>.svg` — dimensioned floor plan SVG

---

## Architecture

```
Input (PLY / video / photo folders)
  ↓
Tier Router → reconstruction/{lidar,video,photo}.py
  ↓
geometry.py          wall segments, room bbox, ceiling height, opening detection
  ↓
stitcher.py          GTSAM pose graph + ICP multi-room alignment, drift ablation
  ↓
damage.py            YOLOv8 (peumalab/wall-defects) + concealed-damage rules
  ↓
output.py            JSON contract + SVG / PNG floor plan
```

**Drift correction:** GTSAM pose graph with floor/ceiling parallelism constraints.  
Run `--no-drift` to disable and see the ablation footprint difference.

---

## Benchmark

```bash
# Gate evaluation vs laser ground truth
python benchmark/evaluate.py \
    --output output/<id>/<id>.json \
    --ground-truth benchmark/ground_truth.csv

# Repeatability (two runs, same input)
python run.py --input data/sample/lidar --tier lidar --capture-id rep1
python run.py --input data/sample/lidar --tier lidar --capture-id rep2
python benchmark/repeatability.py \
    --run1 output/rep1/rep1.json \
    --run2 output/rep2/rep2.json

# Head-to-head vs MagicPlan
python benchmark/head_to_head.py \
    --ours output/<id>/<id>.json \
    --incumbent benchmark/magicplan_measurements.csv \
    --ground-truth benchmark/ground_truth.csv
```

---

## Fix Loop

The declared fix is **wall repeatability**: RANSAC inlier extents were stochastic between runs.
Now wall lengths derive from the deterministic convex-hull floor polygon.

```bash
# Reproduce BEFORE state (stochastic RANSAC extents — use --legacy-walls flag)
python run.py --input data/sample/Assignment/single_room \
              --tier lidar --no-damage --legacy-walls \
              --output fix_loop/before/run_A
python run.py --input data/sample/Assignment/single_room \
              --tier lidar --no-damage --legacy-walls \
              --output fix_loop/before/run_B
python benchmark/repeatability.py \
    --run1 fix_loop/before/run_A/*.json \
    --run2 fix_loop/before/run_B/*.json
# → Verdict: FAIL  (wall rank 6: ~26% spread)

# Reproduce AFTER state (deterministic convex-hull edges — default)
python run.py --input data/sample/Assignment/single_room \
              --tier lidar --no-damage \
              --output fix_loop/after/run_C
python run.py --input data/sample/Assignment/single_room \
              --tier lidar --no-damage \
              --output fix_loop/after/run_D
python benchmark/repeatability.py \
    --run1 fix_loop/after/run_C/*.json \
    --run2 fix_loop/after/run_D/*.json
# → Verdict: PASS  (all 9 walls ≤ 0.54%)
```

Pre-generated artifacts: `fix_loop/before/run_before_A.json`, `run_before_B.json`, `repeatability_before.json`.  
See `fix_loop/declaration.md` for full root cause analysis and prediction.

---

## Capture Route

See `capture_route/protocol.md` for the one-page non-engineer guide.

**Stock tool:** 3D Scanner App (App Store, free), iPhone 15 Pro or newer for LiDAR.

---

## Device Matrix

| Tier | Hardware Required | Accuracy (wall ±) | Ceiling ± | Notes |
|---|---|---|---|---|
| LiDAR | iPhone 15 Pro / Pro Max | ±2cm | ±1.5cm | Best tier |
| Video | iPhone 15 or newer | ±5cm | ±3cm | Good tier |
| Photo | iPhone 15 or newer | ±8–12cm | ±8cm | Wide CIs; honest |

Confidence intervals are calibrated per tier — see `pipeline/confidence.py`.

---

## Toolchain (Disclosed Models + Libraries)

| Tool | Purpose | Source |
|---|---|---|
| Open3D | Point cloud processing, ICP, TSDF | `pip install open3d` |
| pycolmap | Camera pose estimation (SfM) | `pip install pycolmap` |
| GTSAM | Pose graph optimization (drift correction) | `pip install gtsam` |
| Depth Anything V2 (ViT-L) | Metric depth from video/photo | github.com/DepthAnything/Depth-Anything-V2 |
| ZoeDepth (ZoeD_N) | Metric depth fallback (photo tier) | isl-org/ZoeDepth via torch.hub |
| YOLOv8 (peumalab/wall-defects) | Damage detection (crack/mold/stain/corrosion) | Roboflow Universe |
| ezdxf + svgwrite | Floor plan rendering | `pip install ezdxf svgwrite` |
| shapely | 2D polygon geometry | `pip install shapely` |

All models run offline. No external API calls at inference time.

---

## Known Failure Modes

1. **Photo tier, inter-room stitching:** fails without 3+ doorway transition photos (see fix_loop/)
2. **Glass/mirror surfaces:** LiDAR returns noisy depth; system widens CI automatically
3. **Very small rooms (<5m²):** COLMAP may fail with fewer than 4 images; use more images
4. **Video with shaky capture:** COLMAP tracking may drop frames; use slower walking speed
5. **Low light:** YOLOv8 damage detection accuracy drops; add artificial lighting if possible
