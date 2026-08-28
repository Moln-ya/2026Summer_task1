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
