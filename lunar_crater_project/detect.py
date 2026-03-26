"""
Crater detection module.
Supports two modes:
  1. Ground-truth: use annotated txt files directly (for algorithm development/testing)
  2. YOLOv8: run trained model on images (for real deployment)
"""
import os
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import MODELS_DIR, YOLO_CONF, YOLO_IOU, MIN_CRATER_PX
from data_loader import load_crater_annotations


def detect_ground_truth(txt_path: str, min_diameter: float = MIN_CRATER_PX) -> np.ndarray:
    """
    Load ground truth crater positions from annotation file.
    Returns (N, 5) array: [cx, cy, w, h, radius]
    """
    craters = load_crater_annotations(txt_path)
    mask = (craters[:, 2] >= min_diameter) & (craters[:, 3] >= min_diameter)
    return craters[mask]


def detect_yolo(image_path: str, weights_path: str = None,
                conf: float = YOLO_CONF, iou: float = YOLO_IOU,
                min_diameter: float = MIN_CRATER_PX) -> np.ndarray:
    """
    Run YOLOv8 inference on a lunar image.
    Returns (N, 5) array: [cx, cy, w, h, radius]
    """
    import torch
    from ultralytics import YOLO

    if weights_path is None:
        weights_path = os.path.join(MODELS_DIR, 'yolov8n_craters', 'weights', 'best.pt')

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"YOLO weights not found: {weights_path}\n"
                                "Run train_yolo.py first.")

    model = YOLO(weights_path)
    results = model.predict(image_path, conf=conf, iou=iou, verbose=False,
                            device=0 if torch.cuda.is_available() else 'cpu')

    craters = []
    for r in results:
        if r.boxes is None:
            continue
        boxes = r.boxes.xyxy.cpu().numpy()  # (x1,y1,x2,y2)
        for box in boxes:
            x1, y1, x2, y2 = box
            w  = x2 - x1
            h  = y2 - y1
            if w < min_diameter or h < min_diameter:
                continue
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            r  = (w + h) / 4.0
            craters.append([cx, cy, w, h, r])

    if not craters:
        return np.zeros((0, 5), dtype=np.float32)
    return np.array(craters, dtype=np.float32)


def detect_auto(image_path: str, txt_path: str = None,
                weights_path: str = None,
                use_gt: bool = True,
                min_diameter: float = MIN_CRATER_PX) -> np.ndarray:
    """
    Auto-select detection mode.
    If use_gt=True and txt_path exists: use ground truth (fast, no training needed).
    Otherwise: use YOLOv8.
    """
    if use_gt and txt_path and os.path.exists(txt_path):
        craters = detect_ground_truth(txt_path, min_diameter)
        return craters
    else:
        return detect_yolo(image_path, weights_path, min_diameter=min_diameter)


def evaluate_detector(test_dir: str, weights_path: str = None,
                      iou_thresh: float = 0.5) -> dict:
    """
    Evaluate YOLOv8 detector on test images against GT annotations.
    Computes: precision, recall, F1, mAP@0.5
    """
    test_path = Path(test_dir)
    all_tp, all_fp, all_fn = 0, 0, 0

    for txt_file in sorted(test_path.glob('*.txt')):
        img_file = txt_file.with_suffix('.png')
        if not img_file.exists():
            continue

        gt_craters  = detect_ground_truth(str(txt_file))
        det_craters = detect_yolo(str(img_file), weights_path)

        tp, fp, fn = _match_detections(gt_craters, det_craters, iou_thresh)
        all_tp += tp
        all_fp += fp
        all_fn += fn

    precision = all_tp / (all_tp + all_fp + 1e-9)
    recall    = all_tp / (all_tp + all_fn + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)

    return {'precision': precision, 'recall': recall, 'f1': f1,
            'tp': all_tp, 'fp': all_fp, 'fn': all_fn}


def _match_detections(gt: np.ndarray, det: np.ndarray,
                      iou_thresh: float = 0.5):
    """Match detections to GT using IoU threshold. Returns (TP, FP, FN)."""
    if len(gt) == 0:
        return 0, len(det), 0
    if len(det) == 0:
        return 0, 0, len(gt)

    # Convert to xyxy format
    gt_boxes  = _centers_to_xyxy(gt)
    det_boxes = _centers_to_xyxy(det)

    matched_gt  = np.zeros(len(gt), dtype=bool)
    matched_det = np.zeros(len(det), dtype=bool)

    for di, db in enumerate(det_boxes):
        best_iou = 0
        best_gi  = -1
        for gi, gb in enumerate(gt_boxes):
            if matched_gt[gi]:
                continue
            iou = _compute_iou(db, gb)
            if iou > best_iou:
                best_iou = iou
                best_gi  = gi
        if best_iou >= iou_thresh and best_gi >= 0:
            matched_gt[best_gi]  = True
            matched_det[di] = True

    tp = matched_det.sum()
    fp = (~matched_det).sum()
    fn = (~matched_gt).sum()
    return int(tp), int(fp), int(fn)


def _centers_to_xyxy(craters: np.ndarray) -> np.ndarray:
    """Convert [cx,cy,w,h,...] to [x1,y1,x2,y2]."""
    cx, cy, w, h = craters[:,0], craters[:,1], craters[:,2], craters[:,3]
    return np.stack([cx-w/2, cy-h/2, cx+w/2, cy+h/2], axis=1)


def _compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Compute IoU between two boxes [x1,y1,x2,y2]."""
    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    a1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    a2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    return inter / (a1 + a2 - inter + 1e-9)


if __name__ == '__main__':
    from config import TEST_DIR
    print("Testing ground truth loader:")
    test_path = Path(TEST_DIR)
    for txt in list(sorted(test_path.glob('*.txt')))[:2]:
        craters = detect_ground_truth(str(txt))
        print(f"  {txt.name}: {len(craters)} craters")
