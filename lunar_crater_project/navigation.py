"""
Navigation and Position Estimation Module
==========================================
Given matched crater correspondences (obs ↔ map), estimates camera pose.
Computes position estimation error as % of flight altitude.

For 2D navigation (image-to-map matching):
  - Estimate homography H: map_pts → obs_pts
  - H encodes: scale, rotation, translation (camera pose relative to map)
  - Position error: translation magnitude / image_size * 100%

Performance metrics match the base paper:
  - Position error %  in X, Y, Z (X,Y from translation, Z from scale)
  - Reprojection error (pixels): avg, MaxAbsError, RMS
"""
import numpy as np
import cv2
from typing import Dict, Optional, Tuple
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (RANSAC_REPROJ_THRESH, MIN_MATCH_CRATERS,
                    IMAGE_WIDTH, IMAGE_HEIGHT)


# ─── Pose Estimation ──────────────────────────────────────────────────────────

def estimate_homography(obs_pts: np.ndarray,
                        map_pts: np.ndarray,
                        ransac_thresh: float = RANSAC_REPROJ_THRESH
                        ) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """
    Estimate homography H such that: obs_pts ≈ H @ map_pts_homogeneous
    Uses RANSAC for robustness to outliers.

    Returns:
      H      : (3,3) homography matrix or None if failed
      inliers: boolean mask of inlier correspondences
    """
    if len(obs_pts) < 4 or len(map_pts) < 4:
        return None, np.zeros(len(obs_pts), dtype=bool)

    obs_f = obs_pts.astype(np.float32)
    map_f = map_pts.astype(np.float32)

    H, mask = cv2.findHomography(
        map_f, obs_f,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_thresh,
        maxIters=2000,
        confidence=0.995
    )
    if mask is None:
        mask = np.zeros(len(obs_pts), dtype=np.uint8)

    inliers = mask.ravel().astype(bool)
    return H, inliers


def estimate_affine(obs_pts: np.ndarray,
                    map_pts: np.ndarray) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """
    Estimate affine transform (simpler than full homography).
    For near-nadir imagery, affine is a good approximation.
    """
    if len(obs_pts) < 3 or len(map_pts) < 3:
        return None, np.zeros(len(obs_pts), dtype=bool)

    M, inliers = cv2.estimateAffine2D(
        map_pts.astype(np.float32),
        obs_pts.astype(np.float32),
        method=cv2.RANSAC,
        ransacReprojThreshold=RANSAC_REPROJ_THRESH,
        maxIters=2000,
        confidence=0.995
    )
    if inliers is None:
        inliers = np.zeros(len(obs_pts), dtype=np.uint8)
    return M, inliers.ravel().astype(bool)


# ─── Position Error ───────────────────────────────────────────────────────────

def compute_position_error(H: np.ndarray,
                           H_true: Optional[np.ndarray] = None,
                           img_width: int = IMAGE_WIDTH,
                           img_height: int = IMAGE_HEIGHT,
                           altitude_pixels: float = None) -> Dict:
    """
    Compute position estimation error from homography.

    If H_true is given: compare H vs H_true (error in estimated pose).
    If H_true is None:  H should be ~identity; measure deviation from identity.

    Position error = translation / image_size * 100%   (% of altitude)
    Scale error    = |scale - 1| * 100%                 (Z error proxy)

    Returns dict with error_x_pct, error_y_pct, error_z_pct, total_pct
    """
    if H is None:
        return {'error_x_pct': float('nan'), 'error_y_pct': float('nan'),
                'error_z_pct': float('nan'), 'total_pct': float('nan'),
                'tx_px': float('nan'), 'ty_px': float('nan'),
                'scale': float('nan')}

    if altitude_pixels is None:
        altitude_pixels = float(img_width)

    if H_true is not None:
        # Compute residual: H_residual = H @ inv(H_true)
        H_inv_true = np.linalg.inv(H_true)
        H_res = H @ H_inv_true
    else:
        H_res = H

    # Extract translation from residual homography
    # Apply H_res to image center and measure displacement
    center = np.array([[img_width / 2.0, img_height / 2.0]], dtype=np.float64)
    center_h = np.array([[img_width / 2.0, img_height / 2.0, 1.0]])
    warped_h = H_res @ center_h.T  # (3,1)
    warped = (warped_h[:2] / warped_h[2]).T  # (1,2)

    tx = float(warped[0, 0] - center[0, 0])
    ty = float(warped[0, 1] - center[0, 1])

    # Scale: det(H_res[:2,:2])^0.5 ≈ scale change
    scale = float(np.sqrt(abs(np.linalg.det(H_res[:2, :2]))))

    error_x_pct = abs(tx) / altitude_pixels * 100.0
    error_y_pct = abs(ty) / altitude_pixels * 100.0
    error_z_pct = abs(scale - 1.0) * 100.0

    return {
        'error_x_pct': error_x_pct,
        'error_y_pct': error_y_pct,
        'error_z_pct': error_z_pct,
        'total_pct':   np.sqrt(error_x_pct**2 + error_y_pct**2),
        'tx_px': tx,
        'ty_px': ty,
        'scale': scale,
    }


# ─── Reprojection Error ───────────────────────────────────────────────────────

def compute_reprojection_error(obs_pts: np.ndarray,
                               map_pts: np.ndarray,
                               H: np.ndarray) -> Dict:
    """
    Compute reprojection error: project map_pts through H, compare to obs_pts.
    Returns: avg, MaxAbsError, RMS (all in pixels)
    """
    if H is None or len(obs_pts) == 0:
        return {'avg': float('nan'), 'max_abs': float('nan'), 'rms': float('nan')}

    # Project map_pts through H
    map_h  = np.hstack([map_pts, np.ones((len(map_pts), 1))])  # (N,3)
    proj_h = (H @ map_h.T).T  # (N,3)
    proj   = proj_h[:, :2] / (proj_h[:, 2:3] + 1e-9)  # (N,2) normalized

    # Euclidean errors per point
    errors = np.linalg.norm(obs_pts - proj, axis=1)  # (N,)

    # Per-axis absolute errors for MaxAbsError
    abs_errors = np.abs(obs_pts - proj)  # (N,2)

    return {
        'avg':     float(errors.mean()),
        'max_abs': float(abs_errors.max()),
        'rms':     float(np.sqrt((errors**2).mean())),
        'per_pt':  errors,
    }


# ─── Full Navigation Pipeline ─────────────────────────────────────────────────

def navigate(obs_pts: np.ndarray,
             map_pts: np.ndarray,
             H_true: Optional[np.ndarray] = None,
             img_width: int = IMAGE_WIDTH,
             img_height: int = IMAGE_HEIGHT) -> Dict:
    """
    Full navigation pipeline from matched crater pairs.

    1. Estimate homography (map → obs) via RANSAC
    2. Compute position error (% of altitude)
    3. Compute reprojection error

    Returns comprehensive navigation results dict.
    """
    result = {
        'H':         None,
        'n_inliers': 0,
        'position':  {},
        'reproj':    {},
        'success':   False,
    }

    if len(obs_pts) < MIN_MATCH_CRATERS:
        return result

    H, inliers = estimate_homography(obs_pts, map_pts)
    result['H']         = H
    result['n_inliers'] = int(inliers.sum())

    if H is None or inliers.sum() < MIN_MATCH_CRATERS:
        return result

    result['success'] = True

    # Use only inlier correspondences for error computation
    obs_in = obs_pts[inliers]
    map_in = map_pts[inliers]

    result['position'] = compute_position_error(H, H_true, img_width, img_height)
    result['reproj']   = compute_reprojection_error(obs_in, map_in, H)

    return result


def print_navigation_results(nav: Dict):
    """Print navigation results in paper-comparable format."""
    print("\n  --- Navigation Results ---")
    if not nav['success']:
        print("  [FAILED] Not enough matched craters for pose estimation")
        return

    pos = nav['position']
    rep = nav['reproj']
    print(f"  Inliers      : {nav['n_inliers']}")
    print(f"  Position Error X: {pos['error_x_pct']:.4f}%")
    print(f"  Position Error Y: {pos['error_y_pct']:.4f}%")
    print(f"  Position Error Z: {pos['error_z_pct']:.4f}%")
    print(f"  Total XY Error : {pos['total_pct']:.4f}% of altitude")
    print(f"  Tx, Ty (px)  : {pos['tx_px']:.2f}, {pos['ty_px']:.2f}")
    print(f"  Reproj Avg   : {rep['avg']:.3f} px")
    print(f"  Reproj MaxAbs: {rep['max_abs']:.3f} px")
    print(f"  Reproj RMS   : {rep['rms']:.3f} px")


if __name__ == '__main__':
    # Quick test
    np.random.seed(42)
    N = 30
    map_pts = np.random.rand(N, 2) * 800

    # Simulate camera at known offset
    H_true = np.eye(3, dtype=np.float64)
    H_true[0, 2] = 20.0   # 20px translation in X
    H_true[1, 2] = -15.0  # 15px translation in Y

    obs_pts_h = (H_true @ np.hstack([map_pts, np.ones((N,1))]).T).T
    obs_pts   = obs_pts_h[:, :2] / obs_pts_h[:, 2:]
    obs_pts  += np.random.randn(N, 2) * 2.0  # add 2px noise

    nav = navigate(obs_pts, map_pts, H_true=H_true)
    print_navigation_results(nav)
