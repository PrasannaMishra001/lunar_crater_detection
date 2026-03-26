"""
Performance Metrics for Lunar Crater Matching
=============================================
Implements ALL metrics from the base paper and our proposal:

MATCHING LEVEL:
  - Matching accuracy (%) over N trials
  - Number of mismatches
  - Matching success rate vs. false/missed detection error rate (0-100%)
  - Average matching time (seconds/image)

NAVIGATION LEVEL:
  - Position estimation error as % of flight altitude (X, Y, Z)
  - Avg, Max, Min values
  - Reprojection error: avg, MaxAbsError, RMS (pixels)

MONTE CARLO:
  - 1000 trials, Gaussian noise σ=5px on crater centers
  - Reports statistical distribution of all above metrics
"""
import numpy as np
import time
from typing import List, Dict, Tuple, Optional
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import MC_SIGMA, MC_TRIALS, MC_FALSE_RATES, MC_MIN_CRATERS
from triangle_matching import TriangleMatcher
from navigation import navigate


# ─── Ground Truth Correspondence Builder ─────────────────────────────────────

def build_gt_correspondence(n_craters: int) -> Dict[int, int]:
    """
    For simulation, GT is identity: obs crater i corresponds to map crater i.
    (Before adding noise and random drops)
    """
    return {i: i for i in range(n_craters)}


def evaluate_matches(pred_pairs: List[Tuple[int, int]],
                     gt_corr: Dict[int, int]) -> Dict:
    """
    Evaluate predicted crater pairs against ground truth correspondences.
    Returns: correct, incorrect, total, accuracy, mismatches
    """
    correct    = 0
    incorrect  = 0
    mismatches = 0

    for (obs_i, map_j) in pred_pairs:
        gt_map = gt_corr.get(obs_i, None)
        if gt_map is None:
            # obs_i is a spurious (false positive) crater → not in GT
            mismatches += 1
        elif map_j == gt_map:
            correct += 1
        else:
            incorrect += 1
            mismatches += 1

    total = len(pred_pairs)
    accuracy = correct / total * 100.0 if total > 0 else 0.0

    return {
        'correct':   correct,
        'incorrect': incorrect,
        'total':     total,
        'accuracy':  accuracy,
        'mismatches': mismatches,
    }


# ─── Noise & Perturbation Simulation ──────────────────────────────────────────

def simulate_observation(map_craters: np.ndarray,
                         sigma: float = MC_SIGMA,
                         false_rate: float = 0.0,
                         miss_rate: float = 0.0,
                         rng: Optional[np.random.Generator] = None
                         ) -> Tuple[np.ndarray, Dict[int, int]]:
    """
    Simulate a noisy crater observation from ground-truth map craters.

    Steps:
    1. Add Gaussian noise σ to crater centers (position uncertainty)
    2. Randomly miss some craters (false negative = miss_rate %)
    3. Add random spurious craters (false detection = false_rate %)

    Returns:
      obs_craters : (M, 5) noisy observation array
      gt_corr     : {obs_idx → map_idx} ground truth correspondence
                    (only for true craters, not spurious ones)
    """
    if rng is None:
        rng = np.random.default_rng()

    N = len(map_craters)
    obs = map_craters.copy().astype(np.float64)

    # 1. Add position noise
    noise = rng.normal(0, sigma, (N, 2))
    obs[:, :2] += noise

    # 2. Miss some craters
    miss_mask = rng.random(N) < (miss_rate / 100.0)
    keep_mask = ~miss_mask
    obs_true  = obs[keep_mask]
    true_map_idx = np.where(keep_mask)[0]  # original map indices

    # 3. Add spurious craters
    n_false = int(N * false_rate / 100.0)
    gt_corr: Dict[int, int] = {}
    obs_craters = obs_true.copy()

    if n_false > 0:
        # Random positions within the image bounds
        img_size = obs[:, :2].max() + 50
        spurious = rng.random((n_false, obs_true.shape[1]))
        spurious[:, :2] *= img_size
        spurious[:, 2:4] = 10.0  # set size
        spurious[:, 4]   = 5.0
        obs_craters = np.vstack([obs_craters, spurious])

    # Build GT correspondence: obs index → map index
    for obs_i, map_i in enumerate(true_map_idx):
        gt_corr[obs_i] = int(map_i)
    # Spurious craters have no GT correspondence

    return obs_craters.astype(np.float32), gt_corr


# ─── Single Trial ─────────────────────────────────────────────────────────────

def run_single_trial(map_craters: np.ndarray,
                     matcher: TriangleMatcher,
                     sigma: float = MC_SIGMA,
                     false_rate: float = 0.0,
                     miss_rate: float = 0.0,
                     rng: Optional[np.random.Generator] = None
                     ) -> Dict:
    """
    Run one Monte Carlo trial: simulate obs → match → evaluate.
    Returns metrics dict.
    """
    if rng is None:
        rng = np.random.default_rng()

    obs_craters, gt_corr = simulate_observation(
        map_craters, sigma, false_rate, miss_rate, rng)

    if len(obs_craters) < MC_MIN_CRATERS:
        return {'valid': False}

    # Time the matching
    t0 = time.perf_counter()
    match_result = matcher.match(obs_craters, map_craters)
    t1 = time.perf_counter()
    match_time = t1 - t0

    # Use RANSAC-refined pairs for accuracy (global geometric consistency)
    pred_pairs   = match_result.get('crater_pairs_refined', match_result['crater_pairs'])
    match_eval   = evaluate_matches(pred_pairs, gt_corr)

    # Navigation
    nav_result = {'success': False, 'position': {}, 'reproj': {}}
    if len(match_result['obs_pts']) >= 4:
        nav_result = navigate(match_result['obs_pts'], match_result['map_pts'])

    return {
        'valid':        True,
        'match_time':   match_time,
        'n_matches':    match_result['n_matches'],
        'n_obs':        len(obs_craters),
        'accuracy':     match_eval['accuracy'],
        'mismatches':   match_eval['mismatches'],
        'correct':      match_eval['correct'],
        'nav_success':  nav_result['success'],
        'pos_x_pct':    nav_result['position'].get('error_x_pct', float('nan')),
        'pos_y_pct':    nav_result['position'].get('error_y_pct', float('nan')),
        'pos_z_pct':    nav_result['position'].get('error_z_pct', float('nan')),
        'pos_total_pct':nav_result['position'].get('total_pct', float('nan')),
        'reproj_avg':   nav_result['reproj'].get('avg', float('nan')),
        'reproj_max':   nav_result['reproj'].get('max_abs', float('nan')),
        'reproj_rms':   nav_result['reproj'].get('rms', float('nan')),
    }


# ─── Monte Carlo Simulation ───────────────────────────────────────────────────

def run_monte_carlo(map_craters: np.ndarray,
                    matcher: TriangleMatcher,
                    n_trials: int = MC_TRIALS,
                    sigma: float = MC_SIGMA,
                    false_rate: float = 0.0,
                    miss_rate: float = 0.0,
                    seed: int = 42,
                    verbose: bool = True) -> Dict:
    """
    Monte Carlo simulation: n_trials trials with Gaussian noise σ on crater centers.
    false_rate and miss_rate in %.

    Returns aggregate statistics over all valid trials.
    """
    rng = np.random.default_rng(seed)
    trials = []

    for trial_i in range(n_trials):
        r = run_single_trial(map_craters, matcher, sigma, false_rate, miss_rate, rng)
        if r['valid']:
            trials.append(r)

        if verbose and (trial_i + 1) % 200 == 0:
            print(f"    Trial {trial_i+1}/{n_trials} done ({len(trials)} valid)")

    if not trials:
        return {'n_valid': 0}

    def stats(key):
        vals = [t[key] for t in trials if not np.isnan(t[key])]
        if not vals:
            return {'mean': float('nan'), 'max': float('nan'),
                    'min': float('nan'), 'std': float('nan')}
        return {
            'mean': float(np.mean(vals)),
            'max':  float(np.max(vals)),
            'min':  float(np.min(vals)),
            'std':  float(np.std(vals)),
        }

    n_valid = len(trials)
    nav_successes = sum(t['nav_success'] for t in trials)

    return {
        'n_valid':          n_valid,
        'n_trials':         n_trials,
        'sigma':            sigma,
        'false_rate':       false_rate,
        'miss_rate':        miss_rate,
        'nav_success_rate': nav_successes / n_valid * 100.0,
        'matching': {
            'accuracy':   stats('accuracy'),
            'mismatches': stats('mismatches'),
            'n_matches':  stats('n_matches'),
            'time_sec':   stats('match_time'),
        },
        'navigation': {
            'pos_x_pct':   stats('pos_x_pct'),
            'pos_y_pct':   stats('pos_y_pct'),
            'pos_z_pct':   stats('pos_z_pct'),
            'pos_total':   stats('pos_total_pct'),
        },
        'reprojection': {
            'avg':     stats('reproj_avg'),
            'max_abs': stats('reproj_max'),
            'rms':     stats('reproj_rms'),
        },
    }


def run_error_rate_sweep(map_craters: np.ndarray,
                         matcher: TriangleMatcher,
                         n_trials: int = MC_TRIALS,
                         sigma: float = MC_SIGMA,
                         rates: List[float] = MC_FALSE_RATES,
                         seed: int = 42,
                         verbose: bool = True) -> List[Dict]:
    """
    Sweep over detection error rates (combined false/miss rate).
    At each rate, run Monte Carlo and record matching success rate.
    This replicates the paper's Fig. comparing success rate vs error rate.
    """
    results = []
    for rate in rates:
        print(f"  Error rate {rate:3.0f}%  ...", end='', flush=True)
        mc = run_monte_carlo(map_craters, matcher,
                             n_trials=n_trials,
                             sigma=sigma,
                             false_rate=rate / 2,   # split between false & miss
                             miss_rate=rate / 2,
                             seed=seed,
                             verbose=False)
        mc['error_rate'] = rate
        results.append(mc)
        if mc['n_valid'] > 0:
            acc = mc['matching']['accuracy']['mean']
            nav = mc['nav_success_rate']
            print(f"  acc={acc:.1f}%  nav={nav:.1f}%")
        else:
            print("  no valid trials")

    return results


# ─── Pretty Printing ──────────────────────────────────────────────────────────

def print_mc_results(mc: Dict, title: str = "Monte Carlo Results"):
    """Print Monte Carlo results in table format comparable to the paper."""
    if mc.get('n_valid', 0) == 0:
        print(f"{title}: NO VALID TRIALS")
        return

    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"  Trials: {mc['n_valid']}/{mc['n_trials']}  "
          f"sigma={mc['sigma']}px  "
          f"FD={mc['false_rate']:.0f}%  Miss={mc['miss_rate']:.0f}%")
    print(f"{'='*65}")

    m = mc['matching']
    print(f"  MATCHING:")
    print(f"    Accuracy   : {m['accuracy']['mean']:.2f}%  "
          f"(min={m['accuracy']['min']:.2f}%, max={m['accuracy']['max']:.2f}%)")
    print(f"    Mismatches : {m['mismatches']['mean']:.2f}  "
          f"(std={m['mismatches']['std']:.2f})")
    print(f"    Matches/img: {m['n_matches']['mean']:.1f}")
    print(f"    Time/img   : {m['time_sec']['mean']:.4f}s")

    n = mc['navigation']
    print(f"\n  NAVIGATION:")
    print(f"    Success Rate: {mc['nav_success_rate']:.2f}%")
    px  = n['pos_x_pct']
    py  = n['pos_y_pct']
    pz  = n['pos_z_pct']
    pt  = n['pos_total']
    print(f"    Pos Err X (%): avg={px['mean']:.4f} max={px['max']:.4f} min={px['min']:.4f}")
    print(f"    Pos Err Y (%): avg={py['mean']:.4f} max={py['max']:.4f} min={py['min']:.4f}")
    print(f"    Pos Err Z (%): avg={pz['mean']:.4f} max={pz['max']:.4f} min={pz['min']:.4f}")
    print(f"    Pos Err XY(%): avg={pt['mean']:.4f}")

    r = mc['reprojection']
    print(f"\n  REPROJECTION ERROR (pixels):")
    print(f"    Avg        : {r['avg']['mean']:.3f}")
    print(f"    MaxAbsError: {r['max_abs']['mean']:.3f}")
    print(f"    RMS        : {r['rms']['mean']:.3f}")
    print(f"{'='*65}\n")


def print_sweep_results(sweep: List[Dict]):
    """Print error rate sweep results as a table."""
    print(f"\n{'='*65}")
    print("  MATCHING SUCCESS RATE vs DETECTION ERROR RATE")
    print(f"  {'ErrRate':>8}  {'MatchAcc':>10}  {'NavSucc':>10}  {'Time(s)':>8}")
    print(f"  {'-'*50}")
    for mc in sweep:
        if mc['n_valid'] == 0:
            continue
        rate = mc['error_rate']
        acc  = mc['matching']['accuracy']['mean']
        nav  = mc['nav_success_rate']
        t    = mc['matching']['time_sec']['mean']
        print(f"  {rate:>8.0f}%  {acc:>10.2f}%  {nav:>10.2f}%  {t:>8.4f}s")
    print(f"{'='*65}\n")


if __name__ == '__main__':
    # Quick test
    from triangle_matching import TriangleMatcher
    np.random.seed(42)
    N = 60
    map_craters = np.zeros((N, 5), dtype=np.float32)
    map_craters[:, :2] = np.random.rand(N, 2) * 800
    map_craters[:, 2:4] = 12.0
    map_craters[:, 4]   = 6.0

    matcher = TriangleMatcher()
    print("Running mini Monte Carlo (100 trials)...")
    mc = run_monte_carlo(map_craters, matcher, n_trials=100, verbose=True)
    print_mc_results(mc, "MINI TEST (100 trials)")
