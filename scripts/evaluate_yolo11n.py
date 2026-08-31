#!/usr/bin/env python3
"""
Evaluate the trained YOLO11n model on the test split.
Example:
    python scripts/evaluate_yolo11n.py \
        --model models/v2_keyboard_phone_nongfu/best11n_v5.pt \
        --data dataset_nongfu_new/data.yaml
"""
import argparse
from pathlib import Path
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate YOLO11n."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=(
            "models/v2_keyboard_phone_nongfu/"
            "best11n_v5.pt"
        ),
        help="Path to trained weights.",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="dataset_nongfu_new/data.yaml",
        help="Path to dataset data.yaml.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--name",
        type=str,
        default="v5_newdata_yolo11n_test",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )
    print("=== YOLO11n Evaluation ===")
    print(f"Model: {args.model}")
    print(f"Data : {args.data}")
    print(f"Split: {args.split}")
    model = YOLO(args.model)
    metrics = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        name=args.name,
    )
    print()
    print("=== Evaluation Complete ===")
    if hasattr(metrics, "box"):
        print(
            f"mAP50     : {metrics.box.map50:.4f}"
        )
        print(
            f"mAP50-95  : {metrics.box.map:.4f}"
        )

if __name__ == "__main__":
    main()
EOF
