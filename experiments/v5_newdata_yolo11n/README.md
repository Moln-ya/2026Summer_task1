cat > experiments/v5_newdata_yolo11n/README.md <<'EOF'
# V5 Expanded Dataset YOLO11n
## Task
Object detection.
## Classes
0. keyboard
1. nongfu_spring
2. phone
## Model
YOLO11n
## Training
- image size: 640
- epochs: 100
- batch size: 16
- training device: NVIDIA GeForce RTX 4060 Laptop GPU
## Dataset
Expanded manually reviewed dataset based on the previous dataset with additional real images.
## Test Results
- Precision: 0.957
- Recall: 0.960
- mAP50: 0.973
- mAP50-95: 0.806
### Per-class mAP50-95
- keyboard: 0.835
- nongfu_spring: 0.813
- phone: 0.771
## Deployment Test
The model was tested on the Jetson platform with a live camera and showed good practical detection performance.
## Model File
`models/v2_keyboard_phone_nongfu/best11n_v5.pt`
This model is the current deployment candidate.
EOF
