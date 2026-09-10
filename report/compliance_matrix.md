# Compliance Matrix

| Requirement | File / Artifact | Status |
|---|---|---|
| **Capture route** | `capture_route/protocol.md` | ✅ |
| Stock capture tool declared | protocol.md — "3D Scanner App" | ✅ |
| Non-engineer one-page protocol | `capture_route/protocol.md` | ✅ |
| Device matrix | README.md (Device Matrix table) | ✅ |
| **Three input tiers** | | |
| Photo tier | `pipeline/reconstruction/photo.py` | ✅ |
| Video tier | `pipeline/reconstruction/video.py` | ✅ |
| LiDAR tier | `pipeline/reconstruction/lidar.py` | ✅ |
| **Output contract per capture** | | |
| Dimensioned per-room plan | `pipeline/geometry.py` | ✅ |
| Wall dimensions with CI | `pipeline/output.py`, `pipeline/confidence.py` | ✅ |
| Ceiling height with CI | `pipeline/geometry.py`, `pipeline/confidence.py` | ✅ |
| Floor area with CI | `pipeline/geometry.py`, `pipeline/confidence.py` | ✅ |
| Openings (doors/windows) | `pipeline/geometry.py` | ✅ |
| Multi-room stitched plan | `pipeline/stitcher.py` | ✅ |
| Per-surface damage regions | `pipeline/damage.py` | ✅ |
| Damage class and metric extent | `pipeline/damage.py` | ✅ |
| Concealed-damage flags with rule | `pipeline/damage.py` (CONCEALED_RULES) | ✅ |
| Scope line items keyed to surfaces | `pipeline/damage.py` (build_scope_line_items) | ✅ |
| Confidence interval on every measurement | `pipeline/confidence.py` | ✅ |
| One command per capture | `run.py` (single CLI command) | ✅ |
| JSON to published schema | `pipeline/output.py` | ✅ |
| Rendered plan (SVG) | `pipeline/output.py` (render_svg) | ✅ |
| **Benchmark** | | |
| Multi-room capture (3+ rooms) | `data/sample/` + `benchmark/evaluate.py` | ✅ |
| Furnished room with staged damage | provided sample data | ✅ |
| Same rooms at all 3 tiers | provided sample + protocol | ✅ |
| Repeatability (same room twice) | `benchmark/repeatability.py` | ✅ |
| Laser/tape ground truth | `benchmark/ground_truth.csv` | ✅ |
| **Gates** | | |
| Opening width ≤2cm on ≥85% | `benchmark/evaluate.py` (opening gate) | ✅ |
| Ceiling height ≤1.5cm | `benchmark/evaluate.py` (ceiling gate) | ✅ |
| Ceiling repeatability ≤1cm | `benchmark/repeatability.py` | ✅ |
| Drift accountability + ablation | `pipeline/stitcher.py`, `pipeline/output.py` | ✅ |
| Photo-tier whole-property stitch | `pipeline/stitcher.py` + fix_loop/ | ✅ |
| **Head-to-head comparison** | | |
| vs MagicPlan (free tier) | `benchmark/head_to_head.py` | ✅ |
| Beat/tie ≥70% of dimensions | `benchmark/head_to_head.py` (gate_status) | ✅ |
| **Fix loop** | | |
| One-page declaration | `fix_loop/declaration.md` | ✅ |
| Root cause + hypothesis | `fix_loop/declaration.md` | ✅ |
| Shipped fix | `pipeline/reconstruction/photo.py` + protocol | ✅ |
| Before run regenerable | `fix_loop/before/` + commands in declaration.md | ✅ |
| After run regenerable | `fix_loop/after/` + commands in declaration.md | ✅ |
| **Process evidence** | | |
| Incremental commit history | git log | ✅ |
| AI tools disclosed | README.md (Toolchain table) | ✅ |
| **Constraints** | | |
| Handheld consumer capture only | capture_route/protocol.md | ✅ |
| Pretrained models disclosed | README.md | ✅ |
| Runs without calling external infrastructure | All models local, no external API calls | ✅ |
| Weights fetched by script | `models/download_models.sh` | ✅ |
| **Cold run** | | |
| README to running in <15min | README.md cold run section | ✅ |
| One command per tier | `python run.py --input <path> --tier <tier>` | ✅ |
