"""
Train YOLOv8n for lunar crater detection.
YOLOv8 (nano variant) — upgrade over base paper's YOLOv7.
Faster, more accurate on small objects, Ultralytics PyTorch implementation.
"""
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (YOLO_YAML, YOLO_MODEL, YOLO_EPOCHS, YOLO_IMGSZ,
                    YOLO_BATCH, MODELS_DIR)
from prepare_yolo import prepare_dataset


def train_yolo(epochs: int = None, imgsz: int = None, batch: int = None,
               resume: bool = False) -> str:
    """
    Train YOLOv8n on the lunar crater dataset.
    Returns path to best weights.
    """
    from ultralytics import YOLO

    epochs = epochs or YOLO_EPOCHS
    imgsz  = imgsz  or YOLO_IMGSZ
    batch  = batch  or YOLO_BATCH

    # Prepare YOLO-format dataset if not done
    if not os.path.exists(YOLO_YAML):
        print("Preparing YOLO dataset...")
        prepare_dataset()

    print(f"\n{'='*60}")
    print(f"Training YOLOv8n: {epochs} epochs, imgsz={imgsz}, batch={batch}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"{'='*60}\n")

    model = YOLO(YOLO_MODEL)

    results = model.train(
        data=YOLO_YAML,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=MODELS_DIR,
        name='yolov8n_craters',
        save=True,
        save_period=20,
        patience=30,          # early stopping
        device=0 if torch.cuda.is_available() else 'cpu',
        workers=0,            # Windows: set to 0
        verbose=True,
        exist_ok=True,
        # Augmentation tuned for lunar imagery
        hsv_h=0.0,            # no hue shift (grayscale images)
        hsv_s=0.0,            # no saturation shift
        hsv_v=0.3,            # slight brightness aug
        flipud=0.5,
        fliplr=0.5,
        degrees=90.0,         # rotation (craters are rotation-invariant)
        translate=0.1,
        scale=0.3,
        mosaic=0.5,
    )

    best_weights = os.path.join(MODELS_DIR, 'yolov8n_craters', 'weights', 'best.pt')
    print(f"\nTraining complete. Best weights: {best_weights}")
    return best_weights


def validate_yolo(weights_path: str = None) -> dict:
    """Run YOLOv8 validation and return metrics."""
    from ultralytics import YOLO

    if weights_path is None:
        weights_path = os.path.join(MODELS_DIR, 'yolov8n_craters', 'weights', 'best.pt')

    if not os.path.exists(weights_path):
        print(f"Weights not found: {weights_path}")
        return {}

    model = YOLO(weights_path)
    metrics = model.val(data=YOLO_YAML, imgsz=YOLO_IMGSZ,
                        device=0 if torch.cuda.is_available() else 'cpu',
                        workers=0)

    results = {
        'mAP50':    float(metrics.box.map50),
        'mAP50-95': float(metrics.box.map),
        'precision': float(metrics.box.mp),
        'recall':    float(metrics.box.mr),
    }
    print("\nYOLOv8 Validation Results:")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")
    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train YOLOv8n crater detector')
    parser.add_argument('--epochs', type=int, default=YOLO_EPOCHS)
    parser.add_argument('--imgsz',  type=int, default=YOLO_IMGSZ)
    parser.add_argument('--batch',  type=int, default=YOLO_BATCH)
    parser.add_argument('--val-only', action='store_true')
    args = parser.parse_args()

    if args.val_only:
        validate_yolo()
    else:
        weights = train_yolo(args.epochs, args.imgsz, args.batch)
        validate_yolo(weights)
