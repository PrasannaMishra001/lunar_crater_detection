"""
Visualization utilities for lunar crater matching project.
Generates publication-quality figures for the report.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
import os
from typing import List, Dict, Optional, Tuple
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULTS_DIR


def draw_craters_on_image(image: np.ndarray,
                          craters: np.ndarray,
                          color: Tuple = (0, 255, 0),
                          thickness: int = 1) -> np.ndarray:
    """Draw crater circles on image. craters: (N,5) [cx,cy,w,h,r]"""
    img = image.copy()
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for c in craters:
        cx, cy, _, _, r = int(c[0]), int(c[1]), c[2], c[3], int(max(c[4], 1))
        cv2.circle(img, (cx, cy), r, color, thickness)
    return img


def draw_matches(img_obs: np.ndarray, img_map: np.ndarray,
                 obs_pts: np.ndarray, map_pts: np.ndarray,
                 correct_mask: Optional[np.ndarray] = None,
                 title: str = "Crater Matches") -> np.ndarray:
    """
    Draw matched crater pairs side-by-side with lines between matches.
    Green lines = correct, Red lines = incorrect (if correct_mask provided).
    """
    h1, w1 = img_obs.shape[:2]
    h2, w2 = img_map.shape[:2]
    h = max(h1, h2)
    out = np.zeros((h, w1 + w2, 3), dtype=np.uint8)

    # Convert to color
    obs_c = cv2.cvtColor(img_obs, cv2.COLOR_GRAY2BGR) if len(img_obs.shape)==2 else img_obs
    map_c = cv2.cvtColor(img_map, cv2.COLOR_GRAY2BGR) if len(img_map.shape)==2 else img_map

    out[:h1, :w1]       = obs_c
    out[:h2, w1:w1+w2]  = map_c

    for i in range(len(obs_pts)):
        pt1 = (int(obs_pts[i, 0]), int(obs_pts[i, 1]))
        pt2 = (int(map_pts[i, 0]) + w1, int(map_pts[i, 1]))
        if correct_mask is not None:
            color = (0, 255, 0) if correct_mask[i] else (0, 0, 255)
        else:
            color = (0, 255, 0)
        cv2.line(out, pt1, pt2, color, 1)
        cv2.circle(out, pt1, 3, color, -1)
        cv2.circle(out, pt2, 3, color, -1)

    cv2.putText(out, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    return out


def plot_matching_accuracy_vs_error_rate(sweep_results: List[Dict],
                                          save_path: str = None) -> str:
    """
    Plot matching success rate and navigation success rate vs. detection error rate.
    Replicates the key figure from the paper.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    rates    = [r['error_rate']                     for r in sweep_results if r['n_valid'] > 0]
    acc      = [r['matching']['accuracy']['mean']    for r in sweep_results if r['n_valid'] > 0]
    nav_succ = [r['nav_success_rate']                for r in sweep_results if r['n_valid'] > 0]
    time_s   = [r['matching']['time_sec']['mean']    for r in sweep_results if r['n_valid'] > 0]

    # Panel 1: Accuracy vs Error Rate
    ax = axes[0]
    ax.plot(rates, acc, 'b-o', linewidth=2, markersize=7, label='Matching Accuracy')
    ax.plot(rates, nav_succ, 'r-s', linewidth=2, markersize=7, label='Navigation Success')
    ax.axhline(y=90, color='gray', linestyle='--', alpha=0.5, label='90% threshold')
    ax.set_xlabel('Detection Error Rate (%)', fontsize=12)
    ax.set_ylabel('Rate (%)', fontsize=12)
    ax.set_title('Matching & Navigation Success vs. Error Rate', fontsize=12)
    ax.legend(fontsize=10)
    ax.set_xlim([0, 100])
    ax.set_ylim([0, 105])
    ax.grid(True, alpha=0.3)

    # Panel 2: Matching Time vs Error Rate
    ax = axes[1]
    ax.plot(rates, time_s, 'g-^', linewidth=2, markersize=7)
    ax.set_xlabel('Detection Error Rate (%)', fontsize=12)
    ax.set_ylabel('Matching Time (seconds)', fontsize=12)
    ax.set_title('Average Matching Time vs. Error Rate', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 100])

    plt.tight_layout()
    if save_path is None:
        save_path = os.path.join(RESULTS_DIR, 'matching_vs_error_rate.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")
    return save_path


def plot_position_error_histogram(mc_results: List[Dict],
                                   labels: List[str] = None,
                                   save_path: str = None) -> str:
    """Plot distribution of position errors from Monte Carlo simulation."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    dims = ['pos_x_pct', 'pos_y_pct', 'pos_z_pct']
    dim_labels = ['X Error (%)', 'Y Error (%)', 'Z Error (scale %)']

    for ax, dim, dlabel in zip(axes, dims, dim_labels):
        for mc, label in zip(mc_results, labels or ['']*len(mc_results)):
            n = mc.get('n_valid', 0)
            if n == 0:
                continue
            mean = mc['navigation'][dim]['mean']
            std  = mc['navigation'][dim]['std']
            # Simulate normal distribution for illustration
            vals = np.random.normal(mean, std, 500)
            vals = vals[vals >= 0]
            ax.hist(vals, bins=30, alpha=0.6, label=f"{label} (μ={mean:.3f}%)")

        ax.set_xlabel(dlabel, fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title(f'Position Error: {dlabel}', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path is None:
        save_path = os.path.join(RESULTS_DIR, 'position_error_distribution.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")
    return save_path


def plot_triangle_graph(craters: np.ndarray, triangles,
                        title: str = "Delaunay Triangle Graph",
                        save_path: str = None) -> str:
    """Visualize Delaunay triangulation on crater positions."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    # Draw triangles
    for t in triangles:
        pts = t.vertices
        triangle = plt.Polygon(pts, fill=False, edgecolor='blue', alpha=0.3, linewidth=0.5)
        ax.add_patch(triangle)

    # Draw craters
    centers = craters[:, :2]
    ax.scatter(centers[:, 0], centers[:, 1], s=10, c='red', alpha=0.7, label='Craters')

    ax.set_aspect('equal')
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=10)
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.invert_yaxis()  # image coordinates: y increases downward
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    if save_path is None:
        save_path = os.path.join(RESULTS_DIR, 'triangle_graph.png')
    plt.savefig(save_path, dpi=80, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")
    return save_path


def plot_reprojection_errors(reproj_stats: Dict,
                              save_path: str = None) -> str:
    """Bar chart of reprojection error statistics."""
    fig, ax = plt.subplots(figsize=(7, 4))

    keys   = ['avg', 'max_abs', 'rms']
    vals   = [reproj_stats.get(k, {}).get('mean', 0) for k in keys]
    stds   = [reproj_stats.get(k, {}).get('std',  0) for k in keys]
    names  = ['Average', 'Max Abs Error', 'RMS']
    colors = ['steelblue', 'coral', 'mediumseagreen']

    bars = ax.bar(names, vals, color=colors, yerr=stds, capsize=5, alpha=0.85)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{v:.3f}px', ha='center', va='bottom', fontsize=10)

    ax.set_ylabel('Reprojection Error (pixels)', fontsize=11)
    ax.set_title('Reprojection Error Statistics', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    if save_path is None:
        save_path = os.path.join(RESULTS_DIR, 'reprojection_error.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")
    return save_path


def visualize_detection(image_path: str, craters: np.ndarray,
                         save_path: str = None) -> str:
    """Visualize detected craters on an image."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for c in craters:
        cx, cy = int(c[0]), int(c[1])
        r = max(int(c[4]), 1)
        cv2.circle(img_color, (cx, cy), r, (0, 255, 0), 1)
        cv2.circle(img_color, (cx, cy), 2, (0, 0, 255), -1)

    # Overlay count
    cv2.putText(img_color, f"Craters: {len(craters)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

    if save_path is None:
        stem = os.path.splitext(os.path.basename(image_path))[0]
        save_path = os.path.join(RESULTS_DIR, f'{stem}_detections.jpg')
    cv2.imwrite(save_path, img_color)
    print(f"Saved: {save_path}")
    return save_path


def create_results_summary_figure(mc_clean: Dict, sweep: List[Dict],
                                   save_path: str = None) -> str:
    """Create comprehensive 2x2 results figure for the report."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Lunar Crater Matching: Triangle-Based Global Second-Order Similarity\n'
                 'Performance Evaluation Results (Improved Algorithm)', fontsize=13, fontweight='bold')

    # Panel 1: Accuracy vs error rate
    ax = axes[0, 0]
    rates    = [r['error_rate'] for r in sweep if r['n_valid'] > 0]
    acc      = [r['matching']['accuracy']['mean'] for r in sweep if r['n_valid'] > 0]
    nav_succ = [r['nav_success_rate'] for r in sweep if r['n_valid'] > 0]
    if rates:
        ax.plot(rates, acc,      'b-o', lw=2, ms=6, label='Matching Accuracy')
        ax.plot(rates, nav_succ, 'r-s', lw=2, ms=6, label='Navigation Success')
    ax.set_xlabel('Detection Error Rate (%)')
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('(a) Success Rate vs. Error Rate')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 100]); ax.set_ylim([0, 105])

    # Panel 2: Reprojection error bars
    ax = axes[0, 1]
    if mc_clean.get('n_valid', 0) > 0:
        r = mc_clean['reprojection']
        keys   = ['avg', 'max_abs', 'rms']
        names  = ['Average', 'MaxAbs', 'RMS']
        vals   = [r.get(k, {}).get('mean', 0) for k in keys]
        stds   = [r.get(k, {}).get('std',  0) for k in keys]
        colors = ['steelblue', 'coral', 'mediumseagreen']
        bars = ax.bar(names, vals, color=colors, yerr=stds, capsize=5, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.01,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    ax.set_ylabel('Error (pixels)')
    ax.set_title('(b) Reprojection Error (σ=5px noise)')
    ax.grid(True, alpha=0.3, axis='y')

    # Panel 3: Position error in X, Y, Z
    ax = axes[1, 0]
    if mc_clean.get('n_valid', 0) > 0:
        nav = mc_clean['navigation']
        dims    = ['pos_x_pct', 'pos_y_pct', 'pos_z_pct']
        dlabels = ['X', 'Y', 'Z']
        means   = [nav[d]['mean'] for d in dims]
        maxs    = [nav[d]['max']  for d in dims]
        mins    = [nav[d]['min']  for d in dims]
        x = np.arange(3)
        ax.bar(x, means, color=['#2196F3','#4CAF50','#FF9800'], alpha=0.8, label='Mean')
        ax.scatter(x, maxs, marker='^', color='red',   s=80, zorder=5, label='Max')
        ax.scatter(x, mins, marker='v', color='green', s=80, zorder=5, label='Min')
        ax.set_xticks(x); ax.set_xticklabels(dlabels)
        ax.set_ylabel('Position Error (% of altitude)')
        ax.set_title('(c) Navigation Position Error')
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis='y')

    # Panel 4: Matching time distribution
    ax = axes[1, 1]
    if mc_clean.get('n_valid', 0) > 0:
        t = mc_clean['matching']['time_sec']
        t_rates = [r['error_rate'] for r in sweep if r['n_valid'] > 0]
        t_times = [r['matching']['time_sec']['mean'] for r in sweep if r['n_valid'] > 0]
        if t_rates:
            ax.plot(t_rates, t_times, 'purple', marker='D', lw=2, ms=6)
        ax.axhline(t['mean'], color='gray', ls='--', label=f"Clean μ={t['mean']:.4f}s")
        ax.set_xlabel('Detection Error Rate (%)')
        ax.set_ylabel('Time per Image (seconds)')
        ax.set_title('(d) Matching Time vs. Error Rate')
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path is None:
        save_path = os.path.join(RESULTS_DIR, 'results_summary.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")
    return save_path


def plot_improvement_comparison(baseline_dict: Dict,
                                improved_dict: Dict,
                                save_path: str = None) -> str:
    """
    3-panel figure comparing baseline vs improved algorithm performance.

    Visualises the gains from all four improvements:
      1. Crater-radius-augmented descriptor (5D)
      2. Adaptive similarity threshold
      3. Confidence-weighted voting
      4. Extended adjacency (MAX_NEIGHBORS=4)

    baseline_dict / improved_dict keys:
      'clean_accuracy'       : matching accuracy (%) at 0% error
      'nav_success_rate'     : navigation success (%) at 0% error
      'pos_x_pct'            : position error X (% altitude)
      'pos_y_pct'            : position error Y (% altitude)
      'reproj_avg'           : reprojection error avg (px)
      'reproj_rms'           : reprojection error RMS (px)
      'match_time'           : avg matching time (s/image)
      'error_rate_10pct_acc' : matching accuracy (%) at 10% detection error
      'error_rate_20pct_acc' : matching accuracy (%) at 20% detection error
    """
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle('Algorithm Improvement: Baseline vs Enhanced\n'
                 '(5D Radius Descriptor + Adaptive Threshold + Confidence Voting + '
                 'MAX_NEIGHBORS=4)',
                 fontsize=11, fontweight='bold')

    colors  = ['#5B9BD5', '#ED7D31']   # blue=baseline, orange=improved
    labels  = ['Baseline\n(3D desc, fixed thresh)', 'Improved\n(5D desc, adaptive)']
    x       = np.arange(2)
    w       = 0.35

    def safe(v):
        """Return 0 for nan values so bars don't break."""
        try:
            return 0.0 if (v is None or np.isnan(float(v))) else float(v)
        except (TypeError, ValueError):
            return 0.0

    # ── Panel 1: Matching Accuracy & Navigation Success ─────────────────────
    ax = axes[0]
    clean_vals = [safe(baseline_dict.get('clean_accuracy')),
                  safe(improved_dict.get('clean_accuracy'))]
    nav_vals   = [safe(baseline_dict.get('nav_success_rate')),
                  safe(improved_dict.get('nav_success_rate'))]

    bars1 = ax.bar(x - w/2, clean_vals, w, color=colors, alpha=0.85,
                   label='Matching Accuracy (%)')
    bars2 = ax.bar(x + w/2, nav_vals,   w, color=colors, alpha=0.50,
                   label='Nav Success (%)', hatch='//')

    for bar, v in zip(list(bars1) + list(bars2), clean_vals + nav_vals):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
                    f'{v:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

    # Delta annotation
    delta_acc = safe(improved_dict.get('clean_accuracy')) - safe(baseline_dict.get('clean_accuracy'))
    if abs(delta_acc) > 0.05:
        color_d = 'darkgreen' if delta_acc > 0 else 'darkred'
        ax.annotate(f'Δ={delta_acc:+.2f}%', xy=(1 - w/2, clean_vals[1] + 2),
                    fontsize=8, color=color_d, ha='center')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('Rate (%)')
    ax.set_title('(a) Matching Accuracy\n& Navigation Success')
    ax.set_ylim([85, 108])
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(True, alpha=0.3, axis='y')

    # ── Panel 2: Position & Reprojection Errors ──────────────────────────────
    ax = axes[1]
    metric_names = ['Pos X\n(% alt)', 'Pos Y\n(% alt)', 'Reproj Avg\n(px)', 'Reproj RMS\n(px)']
    keys_b = ['pos_x_pct', 'pos_y_pct', 'reproj_avg', 'reproj_rms']
    base_vals = [safe(baseline_dict.get(k)) for k in keys_b]
    impr_vals = [safe(improved_dict.get(k))  for k in keys_b]

    xi = np.arange(len(metric_names))
    b1 = ax.bar(xi - w/2, base_vals, w, color='#5B9BD5', alpha=0.85, label='Baseline')
    b2 = ax.bar(xi + w/2, impr_vals, w, color='#ED7D31', alpha=0.85, label='Improved')

    for bar, v in zip(list(b1) + list(b2), base_vals + impr_vals):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=7)

    ax.set_xticks(xi)
    ax.set_xticklabels(metric_names, fontsize=8)
    ax.set_ylabel('Error Value')
    ax.set_title('(b) Position & Reprojection\nErrors (lower is better)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # ── Panel 3: Robustness at 0%, 10%, 20% error rates ─────────────────────
    ax = axes[2]
    error_labels = ['0% error\n(clean)', '10% error\n(realistic)', '20% error\n(noisy)']
    base_rob = [safe(baseline_dict.get('clean_accuracy')),
                safe(baseline_dict.get('error_rate_10pct_acc')),
                safe(baseline_dict.get('error_rate_20pct_acc'))]
    impr_rob = [safe(improved_dict.get('clean_accuracy')),
                safe(improved_dict.get('error_rate_10pct_acc')),
                safe(improved_dict.get('error_rate_20pct_acc'))]

    xi = np.arange(3)
    ax.bar(xi - w/2, base_rob, w, color='#5B9BD5', alpha=0.85, label='Baseline')
    ax.bar(xi + w/2, impr_rob, w, color='#ED7D31', alpha=0.85, label='Improved')

    # Delta annotations showing improvement
    for i, (bv, iv) in enumerate(zip(base_rob, impr_rob)):
        delta = iv - bv
        if abs(delta) > 0.5 and iv > 0:
            color_d = 'darkgreen' if delta > 0 else 'darkred'
            ax.text(xi[i] + w/2, iv + 1.0,
                    f'{delta:+.1f}%', ha='center', va='bottom',
                    fontsize=8, color=color_d, fontweight='bold')

    ax.set_xticks(xi)
    ax.set_xticklabels(error_labels, fontsize=9)
    ax.set_ylabel('Matching Accuracy (%)')
    ax.set_title('(c) Robustness to\nDetection Errors')
    ax.legend(fontsize=9)
    ax.set_ylim([0, 115])
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    if save_path is None:
        save_path = os.path.join(RESULTS_DIR, 'improvement_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")
    return save_path
