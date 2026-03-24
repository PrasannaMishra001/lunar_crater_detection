"""
Prepare the LRO crater dataset in YOLO format for YOLOv8n training.
Converts bounding box annotations to YOLO format:  class cx_norm cy_norm w_norm h_norm
Uses 10 train images + 2 as validation split.
"""
import os
import sys
import shutil
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (TRAIN_DIR, TEST_DIR, YOLO_TRAIN_IMG, YOLO_VAL_IMG,
                    YOLO_TRAIN_LABEL, YOLO_VAL_LABEL, YOLO_YAML, MIN_CRATER_PX)
from data_loader import load_crater_annotations


def annotation_to_yolo(craters: np.ndarray, img_w: int, img_h: int) -> list:
    """
    Convert crater annotations to YOLO format lines.
    YOLO format: class_id cx_norm cy_norm w_norm h_norm
    All values normalized to [0,1].
    """
    lines = []
    for row in craters:
        cx, cy, w, h = row[0], row[1], row[2], row[3]
        if w < MIN_CRATER_PX or h < MIN_CRATER_PX:
            continue
        cx_n = cx / img_w
        cy_n = cy / img_h
        w_n  = w  / img_w
        h_n  = h  / img_h
        # Clamp to [0,1]
        cx_n = np.clip(cx_n, 0, 1)
        cy_n = np.clip(cy_n, 0, 1)
        w_n  = np.clip(w_n,  0, 1)
        h_n  = np.clip(h_n,  0, 1)
        lines.append(f"0 {cx_n:.6f} {cy_n:.6f} {w_n:.6f} {h_n:.6f}")
    return lines


def prepare_dataset(val_fraction: float = 0.167):
    """
    Prepare YOLO dataset from training annotations.
    Uses all train images (10 train + 2 val split).
    Test images are kept separate for inference.
    """
    train_path = Path(TRAIN_DIR)
    all_imgs = sorted(train_path.glob('*.png'))

    # Split: last ~2 images as validation
    n_val = max(1, int(len(all_imgs) * val_fraction))
    n_train = len(all_imgs) - n_val

    print(f"Preparing YOLO dataset: {n_train} train, {n_val} val images")

    train_imgs = all_imgs[:n_train]
    val_imgs   = all_imgs[n_train:]

    total_train_craters = 0
    total_val_craters   = 0

    for split_imgs, img_dir, lbl_dir, name in [
        (train_imgs, YOLO_TRAIN_IMG, YOLO_TRAIN_LABEL, 'train'),
        (val_imgs,   YOLO_VAL_IMG,   YOLO_VAL_LABEL,   'val'),
    ]:
        count = 0
        for img_path in split_imgs:
            txt_path = img_path.with_suffix('.txt')
            if not txt_path.exists():
                continue

            # Copy image
            dst_img = os.path.join(img_dir, img_path.name)
            if not os.path.exists(dst_img):
                shutil.copy2(str(img_path), dst_img)

            # Convert annotations
            craters = load_crater_annotations(str(txt_path))
            img_w, img_h = 800, 800  # train images are 800x800
            yolo_lines = annotation_to_yolo(craters, img_w, img_h)

            # Write YOLO label file
            lbl_file = os.path.join(lbl_dir, img_path.stem + '.txt')
            with open(lbl_file, 'w') as f:
                f.write('\n'.join(yolo_lines))
            count += len(yolo_lines)

        if name == 'train':
            total_train_craters = count
        else:
            total_val_craters = count
        print(f"  {name}: {len(split_imgs)} images, {count} crater annotations")

    # Write data.yaml
    yaml_content = f"""path: {os.path.abspath(os.path.join(YOLO_TRAIN_IMG, '..', '..'))}
train: images/train
val: images/val

nc: 1
names: ['crater']

# Dataset info
# LRO NAC lunar crater dataset near Chang'E-4 landing site
# ~{total_train_craters} train craters, ~{total_val_craters} val craters
"""
    with open(YOLO_YAML, 'w') as f:
        f.write(yaml_content)
    print(f"\nYAML written to: {YOLO_YAML}")
    print("Dataset preparation complete.")
    return YOLO_YAML


if __name__ == '__main__':
    prepare_dataset()
