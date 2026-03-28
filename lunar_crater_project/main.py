"""
Main Pipeline: Lunar Crater Matching for Precision Navigation
=============================================================
Replicates: "Lunar Crater Matching With Triangle-Based Global
             Second-Order Similarity for Precision Navigation"

Usage:
  python main.py                    # full pipeline (GT detection mode)
  python main.py --train-yolo       # also train YOLOv8n detector first
  python main.py --yolo-detect      # use trained YOLOv8 for detection
  python main.py --quick            # fast mode (fewer trials)

Outputs to: results/ directory
"""
import os
import sys
import time
import argparse
import json
import numpy as np
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (TRAIN_DIR, TEST_DIR, RESULTS_DIR, MODELS_DIR,
                    MC_TRIALS, MC_SIGMA, MC_FALSE_RATES, SHARED_SCENE_PAIRS,
                    MIN_CRATER_PX)
from data_loader import load_all_train_data, load_all_test_data, get_crater_centers
from detect import detect_ground_truth, detect_yolo, detect_auto
from triangle_matching import TriangleMatcher, build_triangles, compute_second_order_descriptors
from navigation import navigate, print_navigation_results
from metrics import (run_monte_carlo, run_error_rate_sweep,
                     print_mc_results, print_sweep_results)
from visualize import (plot_matching_accuracy_vs_error_rate,
                       plot_position_error_histogram,
                       plot_triangle_graph,
                       plot_reprojection_errors,
                       visualize_detection,
                       create_results_summary_figure)


# --- Step 1: Data Inspection --------------------------------------------------

def step1_inspect_data():
    print("\n" + "="*65)
    print("STEP 1: Dataset Inspection")
    print("="*65)
    train_data = load_all_train_data(TRAIN_DIR)
    test_data  = load_all_test_data(TEST_DIR)

    print(f"Training images : {len(train_data)}")
    for img_p, craters in train_data:
        print(f"  {os.path.basename(img_p)}: {len(craters)} craters")

    print(f"\nTest images : {len(test_data)}")
    for img_p, craters in test_data:
        print(f"  {os.path.basename(img_p)}: {len(craters)} craters")

    # Visualize first training image
    if train_data:
        img_p, craters = train_data[0]
        visualize_detection(img_p, craters,
                            save_path=os.path.join(RESULTS_DIR, 'sample_detections.jpg'))

    return train_data, test_data


# --- Step 2: Triangle Graph Visualization ------------------------------------

def step2_triangle_graph(craters: np.ndarray, title: str = "Triangle Graph"):
    print(f"\n  Building Delaunay triangles for {len(craters)} craters...")

    # Use subset of craters for visualization (first 100)
    sample = craters[:min(100, len(craters))]
    triangles, adj = build_triangles(sample)
    if triangles:
        compute_second_order_descriptors(triangles)
        print(f"  Built {len(triangles)} valid triangles")
        plot_triangle_graph(sample, triangles, title=title)
    return triangles


# --- Step 3: Matching Demo (Single Image Pair) --------------------------------

def step3_matching_demo(use_gt: bool = True, weights_path: str = None):
    print("\n" + "="*65)
    print("STEP 3: Crater Matching Demo (Single Image Pair)")
    print("="*65)

    matcher = TriangleMatcher()

    # Load map craters from first training image
    from pathlib import Path
    train_files = sorted(Path(TRAIN_DIR).glob('*.txt'))
    test_files  = sorted(Path(TEST_DIR).glob('*.txt'))

    if not train_files or not test_files:
        print("No data files found.")
        return None

    all_map = detect_ground_truth(str(train_files[0]))
    map_craters = all_map[:200]   # limit for efficient demo matching

    print(f"\n  Map  craters: {len(map_craters)} (from {train_files[0].stem})")

    # Simulate noisy observation from same map (same region + noise + 20% missed)
    rng = np.random.default_rng(42)
    obs_sim = map_craters.copy()
    obs_sim[:, :2] += rng.normal(0, 3.0, (len(map_craters), 2))
    obs_sim = obs_sim[:int(len(map_craters) * 0.8)]  # miss 20%
    print(f"\n  Simulated obs craters: {len(obs_sim)} (80% of map + 3px noise)")

    print("\n  Running triangle matching...")
    t0 = time.perf_counter()
    result = matcher.match(obs_sim, map_craters)
    t1 = time.perf_counter()

    print(f"  Matching time: {t1-t0:.4f}s")
    print(f"  Obs triangles: {result['n_obs_triangles']}")
    print(f"  Map triangles: {result['n_map_triangles']}")
    print(f"  Tri matches  : {len(result['triangle_matches'])}")
    print(f"  Crater matches: {result['n_matches']}")

    # Navigation
    if result['n_matches'] >= 4:
        nav = navigate(result['obs_pts'], result['map_pts'])
        print_navigation_results(nav)

    return result


# --- Step 4: Monte Carlo Evaluation ------------------------------------------

def step4_monte_carlo(map_craters: np.ndarray, n_trials: int = MC_TRIALS,
                      quick: bool = False):
    print("\n" + "="*65)
    print("STEP 4: Monte Carlo Simulation")
    print(f"  Trials: {n_trials}, sigma={MC_SIGMA}px noise")
    print("="*65)

    matcher = TriangleMatcher()
    results_all = {}

    # 4a: Clean matching (no false detections, no misses)
    print("\n  4a. Clean (0% error rate):")
    mc_clean = run_monte_carlo(map_craters, matcher, n_trials=n_trials,
                               sigma=MC_SIGMA, false_rate=0, miss_rate=0,
                               verbose=True)
    print_mc_results(mc_clean, "CLEAN (0% error)")
    results_all['clean'] = mc_clean

    # 4b: 30% error rate (realistic scenario)
    print("\n  4b. Realistic (30% error rate):")
    mc_30 = run_monte_carlo(map_craters, matcher, n_trials=n_trials,
                            sigma=MC_SIGMA, false_rate=15, miss_rate=15,
                            verbose=True)
    print_mc_results(mc_30, "30% ERROR RATE")
    results_all['30pct_error'] = mc_30

    # Save plots
    if mc_clean.get('n_valid', 0) > 0:
        plot_reprojection_errors(mc_clean['reprojection'])
        plot_position_error_histogram([mc_clean, mc_30],
                                      labels=['0% error', '30% error'])

    return results_all, mc_clean


# --- Step 5: Error Rate Sweep -------------------------------------------------

def step5_error_sweep(map_craters: np.ndarray, n_trials: int = MC_TRIALS,
                      quick: bool = False):
    print("\n" + "="*65)
    print("STEP 5: Detection Error Rate Sweep (0-100%)")
    print("="*65)

    matcher = TriangleMatcher()
    rates = [0, 20, 40, 60, 80, 100] if quick else MC_FALSE_RATES

    print(f"  Sweeping {len(rates)} error rates x {n_trials} trials each...")
    sweep = run_error_rate_sweep(map_craters, matcher,
                                 n_trials=n_trials, sigma=MC_SIGMA,
                                 rates=rates, verbose=True)
    print_sweep_results(sweep)
    plot_matching_accuracy_vs_error_rate(sweep)
    return sweep


# --- Step 6: YOLOv8 Detection Evaluation -------------------------------------

def step6_yolo_detection(train: bool = False):
    print("\n" + "="*65)
    print("STEP 6: YOLOv8n Crater Detection")
    print("="*65)

    if train:
        print("  Training YOLOv8n... (this may take 20-60 min on CPU)")
        from prepare_yolo import prepare_dataset
        from train_yolo import train_yolo, validate_yolo
        prepare_dataset()
        weights = train_yolo()
        val_metrics = validate_yolo(weights)
        return val_metrics

    weights_path = os.path.join(MODELS_DIR, 'yolov8n_craters', 'weights', 'best.pt')
    if os.path.exists(weights_path):
        print(f"  Found trained weights: {weights_path}")
        from train_yolo import validate_yolo
        return validate_yolo(weights_path)
    else:
        print(f"  No trained weights found at: {weights_path}")
        print("  Run with --train-yolo to train first.")
        return {}


# --- Step 7: Full Results Summary --------------------------------------------

def step7_save_results(mc_clean: Dict, sweep: List, yolo_metrics: Dict):
    print("\n" + "="*65)
    print("STEP 7: Saving Results Summary")
    print("="*65)

    # Create summary figure
    create_results_summary_figure(mc_clean, sweep)

    # Save JSON results
    summary = {
        'monte_carlo_clean': {
            'n_trials':        mc_clean.get('n_valid', 0),
            'matching_accuracy_mean': mc_clean.get('matching', {}).get('accuracy', {}).get('mean', None),
            'match_time_mean': mc_clean.get('matching', {}).get('time_sec', {}).get('mean', None),
            'nav_success_rate': mc_clean.get('nav_success_rate', None),
            'reproj_avg':      mc_clean.get('reprojection', {}).get('avg', {}).get('mean', None),
            'reproj_rms':      mc_clean.get('reprojection', {}).get('rms', {}).get('mean', None),
            'pos_x_pct_mean':  mc_clean.get('navigation', {}).get('pos_x_pct', {}).get('mean', None),
            'pos_y_pct_mean':  mc_clean.get('navigation', {}).get('pos_y_pct', {}).get('mean', None),
        },
        'yolo_detection': yolo_metrics,
        'error_rate_sweep': [
            {
                'error_rate': r.get('error_rate'),
                'acc_mean':   r.get('matching', {}).get('accuracy', {}).get('mean'),
                'nav_succ':   r.get('nav_success_rate'),
            }
            for r in sweep if r.get('n_valid', 0) > 0
        ]
    }

    json_path = os.path.join(RESULTS_DIR, 'results_summary.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Results saved to: {json_path}")

    # Print paper comparison table
    print("\n" + "="*65)
    print("  COMPARISON WITH BASE PAPER BASELINES")
    print("="*65)
    print(f"  {'Metric':<35} {'Base Paper':>12} {'Ours':>12}")
    print(f"  {'-'*60}")

    acc_m = mc_clean.get('matching', {}).get('accuracy', {}).get('mean', float('nan'))
    pos_x = mc_clean.get('navigation', {}).get('pos_x_pct', {}).get('mean', float('nan'))
    pos_y = mc_clean.get('navigation', {}).get('pos_y_pct', {}).get('mean', float('nan'))
    r_avg = mc_clean.get('reprojection', {}).get('avg', {}).get('mean', float('nan'))
    r_rms = mc_clean.get('reprojection', {}).get('rms', {}).get('mean', float('nan'))
    t_avg = mc_clean.get('matching', {}).get('time_sec', {}).get('mean', float('nan'))

    # Base paper reported ~99%+ accuracy on clean data, <0.44% pos error
    rows = [
        ("Matching Accuracy (clean)",       "~99%",    f"{acc_m:.2f}%"),
        ("Position Error X (% altitude)",   "0.44%",   f"{pos_x:.4f}%"),
        ("Position Error Y (% altitude)",   "0.44%",   f"{pos_y:.4f}%"),
        ("Reprojection Error Avg (px)",     "N/A",     f"{r_avg:.3f}px"),
        ("Reprojection Error RMS (px)",     "N/A",     f"{r_rms:.3f}px"),
        ("Avg Matching Time (s/img)",       "~0.1s",   f"{t_avg:.4f}s"),
    ]
    if yolo_metrics:
        rows.append(("YOLO mAP@50", "YOLOv7 ~0.80", f"{yolo_metrics.get('mAP50', 'N/A')}"))
        rows.append(("YOLO Precision",       "-",    f"{yolo_metrics.get('precision', 'N/A')}"))
        rows.append(("YOLO Recall",          "-",    f"{yolo_metrics.get('recall', 'N/A')}"))

    for name, base, ours in rows:
        print(f"  {name:<35} {base:>12} {ours:>12}")
    print("="*65)


# --- Main Entry Point ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Lunar Crater Matching - Triangle-Based Global Second-Order Similarity')
    parser.add_argument('--train-yolo', action='store_true',
                        help='Train YOLOv8n detector before running pipeline')
    parser.add_argument('--yolo-detect', action='store_true',
                        help='Use YOLOv8 for detection (requires trained weights)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: 100 trials instead of 1000')
    parser.add_argument('--skip-sweep', action='store_true',
                        help='Skip error rate sweep (faster)')
    parser.add_argument('--image-index', type=int, default=0,
                        help='Which training image to use as map (default=0)')
    args = parser.parse_args()

    n_trials = 100 if args.quick else MC_TRIALS
    print(f"\n{'#'*65}")
    print("  LUNAR CRATER MATCHING -- TRIANGLE-BASED GLOBAL 2ND-ORDER SIMILARITY")
    print(f"  Group 3 AIP Project | Trials={n_trials} | Quick={args.quick}")
    print(f"{'#'*65}\n")

    # Step 1: Inspect data
    train_data, test_data = step1_inspect_data()

    # Choose map craters from selected training image
    # Limit to 200 for fast Monte Carlo (still representative)
    img_p, all_map_craters = train_data[args.image_index]
    map_craters = all_map_craters[:200]
    print(f"\n  Using map: {os.path.basename(img_p)} "
          f"({len(all_map_craters)} total, using {len(map_craters)} for Monte Carlo)")

    # Step 2: Visualize triangle graph
    print("\n" + "="*65)
    print("STEP 2: Triangle Graph Visualization")
    print("="*65)
    step2_triangle_graph(map_craters,
                         title=f"Delaunay Triangle Graph\n({os.path.basename(img_p)})")

    # Step 3: Matching demo
    result = step3_matching_demo(use_gt=not args.yolo_detect)

    # Step 4: Monte Carlo simulation
    mc_results, mc_clean = step4_monte_carlo(map_craters, n_trials=n_trials,
                                              quick=args.quick)

    # Step 5: Error rate sweep
    sweep = []
    if not args.skip_sweep:
        sweep = step5_error_sweep(map_craters, n_trials=min(n_trials, 200),
                                  quick=args.quick)

    # Step 6: YOLOv8 (optional)
    yolo_metrics = step6_yolo_detection(train=args.train_yolo)

    # Step 7: Save + compare
    step7_save_results(mc_clean, sweep, yolo_metrics)

    print(f"\nAll results saved to: {RESULTS_DIR}")
    print("Pipeline complete!\n")


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
