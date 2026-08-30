cat > README.md <<'EOF'
# 2026Summer_task1
ROS 2 and YOLO‑based real‑time object detection project.
## Object Detection Classes
The current detector recognises three object classes:
1. keyboard
2. nongfu_spring
3. phone
## Current Deployment Model
The current deployment candidate is YOLO11n.
Model:
`models/v2_keyboard_phone_nongfu/best11n_v5.pt`
Input size:
`640 x 640`
Default confidence threshold:
`0.25`
## V5 Test Results
- Precision: 0.957
- Recall: 0.960
- mAP50: 0.973
- mAP50‑95: 0.806
Per‑class mAP50‑95:
- keyboard: 0.835
- nongfu_spring: 0.813
- phone: 0.771
The model has also been tested with a live camera on the Jetson platform.
## Training
Training is implemented using the Ultralytics YOLO Python API.
Example:
```bash
python scripts/train_yolo11n.py \
    --data /path/to/dataset/data.yaml
