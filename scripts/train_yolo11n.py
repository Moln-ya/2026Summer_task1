cat > scripts/train_yolo11n.py <<'EOF'
#!/usr/bin/env python3
"""
Train YOLO11n for the 3-class object detection task.
Example:
    python scripts/train_yolo11n.py \
        --data dataset_nongfu_new/data.yaml
"""
import argparse
from ultralytics import YOLO
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train YOLO11n."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="dataset_nongfu_new/data.yaml",
        help="Path to dataset data.yaml.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n.pt",
        help="Initial YOLO model/weights.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
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
        help="CUDA device, e.g. 0, or cpu.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--name",
        type=str,
        default="v5_newdata_yolo11n",
    )
    return parser.parse_args()
def main():
    args = parse_args()
    print("=== YOLO11n Training ===")
    print(f"Model : {args.model}")
    print(f"Data  : {args.data}")
    print(f"Epochs: {args.epochs}")
    print(f"Image : {args.imgsz}")
    print(f"Batch : {args.batch}")
    print(f"Device: {args.device}")
    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        name=args.name,
    )
if __name__ == "__main__":
    main()
EOF
