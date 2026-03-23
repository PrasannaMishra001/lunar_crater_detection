"""
Data loader for LRO Lunar Crater Detection Dataset.
Handles train/test annotation files and provides crater data as numpy arrays.
Format: ID, X, Y, W, H, C_X, CY  (comma or whitespace separated, header row)
"""
import os
import numpy as np
from pathlib import Path


def load_crater_annotations(txt_path: str) -> np.ndarray:
    """
    Load crater annotations from a .txt file.
    Returns array of shape (N, 5): [cx, cy, w, h, radius]
    """
    with open(txt_path, 'r') as f:
        lines = f.readlines()

    records = []
    for line in lines[1:]:  # skip header
        line = line.strip()
        if not line:
            continue
        # Handle both comma and whitespace separators
        if ',' in line:
            parts = line.split(',')
        else:
            parts = line.split()
        if len(parts) < 5:
            continue
        try:
            # ID, X, Y, W, H, C_X, CY  → use C_X, CY as center, W/2 as radius
            # X,Y = top-left, W,H = width,height; C_X,CY = center
            idx = int(float(parts[0]))
            x   = float(parts[1])
            y   = float(parts[2])
            w   = float(parts[3])
            h   = float(parts[4])
            if len(parts) >= 7:
                cx = float(parts[5])
                cy = float(parts[6])
            else:
                cx = x + w / 2.0
                cy = y + h / 2.0
            r = (w + h) / 4.0  # average radius
            records.append([cx, cy, w, h, r])
        except ValueError:
            continue

    return np.array(records, dtype=np.float32)  # (N, 5): cx,cy,w,h,r


def load_image_and_craters(image_path: str, txt_path: str, min_diameter: float = 8.0):
    """
    Load an image and its crater annotations.
    Filters out craters smaller than min_diameter pixels.
    Returns: (image_path, craters_array)  where craters: (N,5) [cx,cy,w,h,r]
    """
    craters = load_crater_annotations(txt_path)
    # Filter by minimum size
    mask = (craters[:, 2] >= min_diameter) & (craters[:, 3] >= min_diameter)
    craters = craters[mask]
    return image_path, craters


def load_all_train_data(train_dir: str, min_diameter: float = 8.0):
    """
    Load all training images and annotations.
    Returns list of (image_path, craters_array) tuples.
    """
    results = []
    train_path = Path(train_dir)
    for txt_file in sorted(train_path.glob('*.txt')):
        img_file = txt_file.with_suffix('.png')
        if img_file.exists():
            img_p, craters = load_image_and_craters(str(img_file), str(txt_file), min_diameter)
            results.append((img_p, craters))
    return results


def load_all_test_data(test_dir: str, min_diameter: float = 8.0):
    """
    Load all test images and annotations.
    Returns list of (image_path, craters_array) tuples.
    """
    results = []
    test_path = Path(test_dir)
    for txt_file in sorted(test_path.glob('*.txt')):
        img_file = txt_file.with_suffix('.png')
        if img_file.exists():
            img_p, craters = load_image_and_craters(str(img_file), str(txt_file), min_diameter)
            results.append((img_p, craters))
    return results


def get_crater_centers(craters: np.ndarray) -> np.ndarray:
    """Extract (cx, cy) array from craters array."""
    return craters[:, :2].copy()


def get_crater_radii(craters: np.ndarray) -> np.ndarray:
    """Extract radius array from craters array."""
    return craters[:, 4].copy()


def print_dataset_stats(train_dir: str, test_dir: str):
    """Print summary statistics for the dataset."""
    print("=" * 60)
    print("LUNAR CRATER DATASET STATISTICS")
    print("=" * 60)
    for d, name in [(train_dir, 'TRAIN'), (test_dir, 'TEST')]:
        data = load_all_train_data(d) if name == 'TRAIN' else load_all_test_data(d)
        total = sum(len(c) for _, c in data)
        print(f"\n{name}: {len(data)} images, {total} craters total")
        for img_path, craters in data:
            img_name = os.path.basename(img_path)
            D = craters[:, 2]
            print(f"  {img_name}: n={len(craters)}, diam=[{D.min():.0f},{D.max():.0f}] mean={D.mean():.1f}px")
    print("=" * 60)


if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import TRAIN_DIR, TEST_DIR
    print_dataset_stats(TRAIN_DIR, TEST_DIR)
