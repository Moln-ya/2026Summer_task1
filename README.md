# 2026Summer_task1
## ROS 2 + YOLO11n Real-Time Object Detection on Jetson
This project implements a real-time object detection system for desktop objects using YOLO11n, NVIDIA Jetson, and ROS 2.
The complete pipeline includes:
- image collection
- manual annotation and dataset review
- YOLO11n training
- model evaluation
- Jetson deployment
- real-time camera inference
- bounding-box, class and confidence display
- FPS measurement
- ROS 2 detection-result publishing
- test-result and typical-error analysis
---
# 1. Project Task
The objective of this experiment is to build a real-time object detection system satisfying the following requirements:
- Detect at least two categories of desktop objects.
- Collect and annotate the dataset manually.
- Train an object detection model.
- Run the trained model on Jetson.
- Display object class, bounding box and confidence in real time.
- Publish detection results through ROS 2.
- Test at least 20 objects with an accuracy of at least 80%.
- Maintain a Jetson real-time detection speed of at least 5 FPS.
- Save model test results and representative error cases.
The final system detects three classes:
1. `keyboard`
2. `nongfu_spring`
3. `phone`
---
# 2. Final System Pipeline
```text
USB Camera
    |
    v
NVIDIA Jetson
    |
    v
YOLO11n
    |
    +-----------------------------+
    |                             |
    v                             v
Bounding Box                  ROS 2 Publisher
Class Name                        |
Confidence                        v
FPS                          /detections
