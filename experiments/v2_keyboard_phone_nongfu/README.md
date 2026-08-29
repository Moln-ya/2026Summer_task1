# V2 Keyboard + Phone + Nongfu Spring Baseline
## Classes
0. keyboard
1. phone
2. nongfu_spring
## Dataset
Roboflow manually reviewed dataset.
Core imported batch: 596 images.
Laptop class was removed and replaced with Nongfu Spring water bottle.
## Model
YOLO11n
## Training
- image size: 640
- epochs: 100
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU
- model: yolo11n.pt
## Purpose
This model is the V2 baseline before robustness-oriented data augmentation
and external dataset expansion.
## Current Model Selection
Two fixed-label baseline models are retained:
- `best.pt`: YOLO11n fixed-label baseline
- `best11s.pt`: YOLO11s fixed-label baseline
The YOLO11s model is currently selected as the deployment candidate.
YOLO11s test results:
- Precision: 0.800
- Recall: 0.940
- mAP50: 0.940
- mAP50-95: 0.849
Per-class mAP50-95:
- keyboard: 0.857
- nongfu_spring: 0.858
- phone: 0.832
