#!/usr/bin/env python3

import json
import os
import time
from datetime import datetime
from pathlib import Path

import cv2
import rclpy
import torch

from rclpy.node import Node
from std_msgs.msg import String
from ultralytics import YOLO


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = (
    Path.home()
    / "Desktop"
    / "ljx_task1"
    / "best11n_v5.pt"
)

CAMERA_ID = 2

CONF_THRESHOLD = 0.7

IMAGE_SIZE = 640

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480


class YoloDetectorNode(Node):

    def __init__(self):

        super().__init__("yolo_detector")

        # ----------------------------------------------------
        # Check model
        # ----------------------------------------------------

        if not MODEL_PATH.exists():

            raise FileNotFoundError(
                f"Model not found: {MODEL_PATH}"
            )

        self.get_logger().info(
            f"Loading model: {MODEL_PATH}"
        )

        # ----------------------------------------------------
        # Load YOLO
        # ----------------------------------------------------

        self.model = YOLO(
            str(MODEL_PATH)
        )

        self.get_logger().info(
            f"Classes: {self.model.names}"
        )

        # ----------------------------------------------------
        # Device
        # ----------------------------------------------------

        if torch.cuda.is_available():

            self.device = "0"

            self.get_logger().info(
                "Inference device: CUDA GPU"
            )

        else:

            self.device = "cpu"

            self.get_logger().warning(
                "CUDA unavailable, using CPU"
            )

        # ----------------------------------------------------
        # ROS2 publisher
        # ----------------------------------------------------

        self.publisher_ = self.create_publisher(
            String,
            "/detections",
            10,
        )

        # ----------------------------------------------------
        # Camera
        # ----------------------------------------------------

        self.cap = cv2.VideoCapture(
            CAMERA_ID
        )

        self.cap.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            CAMERA_WIDTH,
        )

        self.cap.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            CAMERA_HEIGHT,
        )

        if not self.cap.isOpened():

            raise RuntimeError(
                f"Cannot open camera {CAMERA_ID}"
            )

        self.get_logger().info(
            "Camera opened successfully."
        )

        # ----------------------------------------------------
        # GUI
        # ----------------------------------------------------

        self.show_gui = bool(
            os.environ.get("DISPLAY")
        )

        if self.show_gui:

            self.get_logger().info(
                "GUI enabled."
            )

        else:

            self.get_logger().info(
                "SSH/headless mode: GUI disabled."
            )

        # ----------------------------------------------------
        # FPS
        # ----------------------------------------------------

        self.last_time = time.perf_counter()

        self.fps = 0.0

        self.fps_values = []

        # ----------------------------------------------------
        # Timer
        # ----------------------------------------------------

        self.timer = self.create_timer(
            0.01,
            self.detect_callback,
        )

        self.get_logger().info(
            "YOLO11n ROS2 detector started."
        )

        self.get_logger().info(
            "Publishing: /detections"
        )

    def update_fps(self):

        now = time.perf_counter()

        delta = now - self.last_time

        self.last_time = now

        if delta <= 0:
            return

        current_fps = 1.0 / delta

        self.fps_values.append(
            current_fps
        )

        if len(self.fps_values) > 20:
            self.fps_values.pop(0)

        self.fps = (
            sum(self.fps_values)
            /
            len(self.fps_values)
        )

    def detect_callback(self):

        # ----------------------------------------------------
        # Read camera
        # ----------------------------------------------------

        ok, frame = self.cap.read()

        if not ok:

            self.get_logger().warning(
                "Failed to read camera frame."
            )

            return

        # ----------------------------------------------------
        # YOLO inference
        # ----------------------------------------------------

        results = self.model.predict(
            source=frame,
            imgsz=IMAGE_SIZE,
            conf=CONF_THRESHOLD,
            device=self.device,
            verbose=False,
        )

        result = results[0]

        # ----------------------------------------------------
        # FPS
        # ----------------------------------------------------

        self.update_fps()

        # ----------------------------------------------------
        # Parse detections
        # ----------------------------------------------------

        detections = []

        if result.boxes is not None:

            for box in result.boxes:

                class_id = int(
                    box.cls[0].item()
                )

                confidence = float(
                    box.conf[0].item()
                )

                x1, y1, x2, y2 = [
                    int(value)
                    for value
                    in box.xyxy[0].tolist()
                ]

                class_name = (
                    self.model.names[
                        class_id
                    ]
                )

                detections.append(
                    {
                        "class": class_name,
                        "confidence": round(
                            confidence,
                            3,
                        ),
                        "bbox": {
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                        },
                    }
                )

        # ----------------------------------------------------
        # Publish ROS2 message
        # ----------------------------------------------------

        output = {
            "timestamp": datetime.now().isoformat(
                timespec="milliseconds"
            ),
            "fps": round(
                self.fps,
                2,
            ),
            "count": len(
                detections
            ),
            "detections": detections,
        }

        message = String()

        message.data = json.dumps(
            output,
            ensure_ascii=False,
        )

        self.publisher_.publish(
            message
        )

        # ----------------------------------------------------
        # GUI
        # ----------------------------------------------------

        if self.show_gui:

            annotated = result.plot()

            cv2.putText(
                annotated,
                f"FPS: {self.fps:.1f}",
                (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                annotated,
                "ROS2: /detections",
                (15, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            cv2.imshow(
                "YOLO11n Object Detection",
                annotated,
            )

            key = cv2.waitKey(1) & 0xFF

            if (
                key == ord("q")
                or
                key == 27
            ):

                rclpy.shutdown()

    def destroy_node(self):

        if self.cap is not None:
            self.cap.release()

        if self.show_gui:
            cv2.destroyAllWindows()

        super().destroy_node()


def main(args=None):

    rclpy.init(
        args=args
    )

    node = None

    try:

        node = YoloDetectorNode()

        rclpy.spin(
            node
        )

    except KeyboardInterrupt:

        print(
            "\nStopped by user."
        )

    finally:

        if node is not None:
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
