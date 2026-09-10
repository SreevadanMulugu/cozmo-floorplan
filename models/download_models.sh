#!/bin/bash
# Downloads all model weights needed for the pipeline.
# Run once before using the pipeline.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Downloading Depth Anything V2 metric indoor model ==="
mkdir -p depth_anything_v2
if [ ! -f depth_anything_v2/depth_anything_v2_metric_hypersim_vitl.pth ]; then
    # ViT-L metric indoor model (best accuracy/speed tradeoff)
    wget -q --show-progress \
        "https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Indoor-Large/resolve/main/depth_anything_v2_metric_hypersim_vitl.pth" \
        -O depth_anything_v2/depth_anything_v2_metric_hypersim_vitl.pth
    echo "  -> Depth Anything V2 metric indoor downloaded"
else
    echo "  -> Depth Anything V2 metric indoor already present"
fi

echo "=== Downloading YOLOv8 wall defects model ==="
mkdir -p yolov8
if [ ! -f yolov8/wall_defects.pt ]; then
    # peumalab/wall-defects from Roboflow - covers crack, mold, stain, corrosion, deterioration
    python3 -c "
from roboflow import Roboflow
import shutil, os
rf = Roboflow(api_key='${ROBOFLOW_API_KEY:-}')
project = rf.workspace('peumalab').project('wall-defects')
model = project.version(1).model
# Download weights
model.download('yolov8', location='yolov8/')
print('Downloaded wall-defects model')
" 2>/dev/null || {
    echo "  -> Roboflow API key not set; using generic YOLOv8n as fallback (fine-tune manually)"
    python3 -c "
from ultralytics import YOLO
m = YOLO('yolov8n.pt')  # downloads pretrained weights
import shutil
shutil.copy(m.ckpt_path, 'yolov8/wall_defects.pt') if hasattr(m, 'ckpt_path') else None
" 2>/dev/null || wget -q https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt -O yolov8/wall_defects.pt
    echo "  -> Using YOLOv8n weights (set ROBOFLOW_API_KEY to get domain-specific model)"
}
else
    echo "  -> YOLOv8 wall defects model already present"
fi

echo ""
echo "=== All models ready ==="
echo "  models/depth_anything_v2/depth_anything_v2_metric_hypersim_vitl.pth"
echo "  models/yolov8/wall_defects.pt"
echo ""
echo "Next: clone Depth Anything V2 repo if not done:"
echo "  git clone https://github.com/DepthAnything/Depth-Anything-V2 ../Depth-Anything-V2"
echo "  pip install -r ../Depth-Anything-V2/requirements.txt"
