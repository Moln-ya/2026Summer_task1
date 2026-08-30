cat > scripts/jetson_camera_test.py <<'EOF'
#!/usr/bin/env python3
"""
Real-time YOLO11n camera inference for Jetson.
This script does not require ROS 2.
It is used to verify:
camera -> YOLO11n -> detections
before integrating the model into a ROS 2 node.
"""
import argparse
import time
from pathlib import Path
import cv2
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(
        description="Jetson YOLO11n camera test."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=(
            "models/v2_keyboard_phone_nongfu/"
            "best11n_v5.pt"
        ),
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.7,
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )
    print("Loading model...")
    print(model_path)
    model = YOLO(str(model_path))
    print("Model classes:")
    print(model.names)
    cap = cv2.VideoCapture(
        args.camera
    )
    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open camera {args.camera}"
        )
    print()
    print("Camera started.")
    print("Press Q or ESC to exit.")
    previous_time = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read camera frame.")
            break
        results = model.predict(
            source=frame,
            imgsz=args.imgsz,
            conf=args.conf,
            device=args.device,
            verbose=False,
        )
        result = results[0]
        annotated = result.plot()
        # ---------------------------------------------
        # FPS
        # ---------------------------------------------
        current_time = time.perf_counter()
        delta = (
            current_time
            -
            previous_time
        )
        previous_time = current_time
        if delta > 0:
            fps = 1.0 / delta
        else:
            fps = 0.0
        cv2.putText(
            annotated,
            f"FPS: {fps:.1f}",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        # ---------------------------------------------
        # Print detected objects
        # ---------------------------------------------
        if result.boxes is not None:
            detections = []
            for box in result.boxes:
                cls_id = int(
                    box.cls[0].item()
                )
                confidence = float(
                    box.conf[0].item()
                )
                class_name = (
                    model.names[cls_id]
                )
                detections.append(
                    f"{class_name}"
                    f"({confidence:.2f})"
                )
            if detections:
                print(
                    "Detected:",
                    ", ".join(detections),
                )
        cv2.imshow(
            "Jetson YOLO11n Camera Test",
            annotated,
        )
        key = (
            cv2.waitKey(1)
            &
            0xFF
        )
        if (
            key == ord("q")
            or
            key == 27
        ):
            break
    cap.release()
    cv2.destroyAllWindows()
    print("Camera test finished.")

if __name__ == "__main__":
    main()
EOF
