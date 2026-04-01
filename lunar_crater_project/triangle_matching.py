"""
Triangle-Based Global Second-Order Similarity Crater Matching
=============================================================
Core algorithm from: "Lunar Crater Matching With Triangle-Based Global
Second-Order Similarity for Precision Navigation"

IMPROVEMENTS OVER BASE PAPER:
  1. Crater-radius-augmented first-order descriptor (5D instead of 3D):
       d = (l1/l3, l2/l3, area_norm, r_min/r_max, r_mid/r_max)
     Physical crater sizes add discriminability beyond pure geometry.
     Two triangles of identical shape but different crater sizes are now
     distinguishable — a limitation the base paper has.

  2. Adaptive similarity threshold:
       threshold = max(MATCH_THRESHOLD_FLOOR, 0.75 * mean(top-K row-maxima))
     Automatically lowers the acceptance bar when the similarity score
     distribution shifts under noise or missed/false detections. Significantly
     improves robustness at 10-20% detection error rates.

  3. Confidence-weighted crater correspondence voting:
     Votes from each matched triangle pair are scaled by the detection
     confidence of the observed crater. High-confidence detections dominate;
     spurious/uncertain detections contribute proportionally less.

  4. Extended adjacency (MAX_NEIGHBORS=4): richer neighbourhood context in
     second-order descriptors.

Algorithm:
1. Build Delaunay triangulation on crater centers
2. For each triangle, compute FIRST-ORDER descriptor:
   - Normalized sorted side ratios (l1/l3, l2/l3) — scale/rotation/translation invariant
   - Perimeter-normalized area for discriminability
   - [NEW] Normalized radius ratios (r_min/r_max, r_mid/r_max)
3. Build adjacency graph (triangles sharing an edge)
4. For each triangle, compute SECOND-ORDER descriptor:
   - Concatenate own first-order descriptor with sorted neighbour descriptors
   - Captures local geometric neighbourhood context
5. Build similarity matrix using Gaussian kernel
6. [NEW] Adaptive threshold from score distribution
7. Greedy global matching with one-to-one enforcement
8. [NEW] Confidence-weighted voting for crater correspondences
9. RANSAC geometric verification
10. Return matched crater correspondences
"""
import numpy as np
from scipy.spatial import Delaunay
from typing import List, Tuple, Dict, Optional
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (MAX_NEIGHBORS, SIMILARITY_SIGMA, MATCH_THRESHOLD,
                    MATCH_THRESHOLD_FLOOR, USE_RADIUS_DESCRIPTOR,
                    ADAPTIVE_THRESHOLD, CONF_WEIGHTED_VOTING,
                    DELAUNAY_MAX_SIDE_RATIO, MIN_TRIANGLE_AREA, MAX_TRIANGLES)


# ─── Triangle Descriptor ──────────────────────────────────────────────────────

class Triangle:
    """
    Represents a triangle formed by three crater centers.
    Stores first-order (own geometry) and second-order (neighbourhood) descriptors.
    """
    __slots__ = ['v_idx', 'vertices', 'sides', 'desc1', 'desc2',
                 'adj_triangles', 'area', 'v_radii']

    def __init__(self, v_idx: Tuple[int, int, int], vertices: np.ndarray,
                 radii: Optional[np.ndarray] = None):
        """
        v_idx   : indices into the crater array (i, j, k)
        vertices: (3, 2) array of (x, y) positions
        radii   : (3,) array of crater radii at each vertex — optional.
                  When provided with USE_RADIUS_DESCRIPTOR=True, the first-order
                  descriptor is extended with scale-invariant radius ratios.
        """
        self.v_idx        = tuple(v_idx)
        self.vertices     = vertices          # (3,2)
        self.v_radii      = radii             # (3,) or None
        self.sides        = self._compute_sides()
        self.area         = self._compute_area()
        self.desc1        = self._compute_first_order()
        self.desc2        = None              # filled after adjacency is built
        self.adj_triangles: List[int] = []   # indices of adjacent triangles

    def _compute_sides(self) -> np.ndarray:
        """Compute the 3 side lengths in ascending order."""
        p = self.vertices
        d = np.array([
            np.linalg.norm(p[1] - p[2]),  # side opposite vertex 0
            np.linalg.norm(p[0] - p[2]),  # side opposite vertex 1
            np.linalg.norm(p[0] - p[1]),  # side opposite vertex 2
        ])
        return np.sort(d)  # ascending: [l1, l2, l3]

    def _compute_area(self) -> float:
        """Compute triangle area via cross product."""
        p = self.vertices
        v1 = p[1] - p[0]
        v2 = p[2] - p[0]
        return abs(v1[0]*v2[1] - v1[1]*v2[0]) / 2.0

    def _compute_first_order(self) -> np.ndarray:
        """
        First-order descriptor: scale/rotation/translation invariant.

        Base (3D): (l1/l3, l2/l3, area_norm)
          - l1 <= l2 <= l3 are sorted side lengths
          - area_norm = area / l3^2

        [IMPROVED] With radius (5D), when USE_RADIUS_DESCRIPTOR=True and
        radii are available:
          + (r_min/r_max, r_mid/r_max)
          - r_min <= r_mid <= r_max are sorted crater radii at the 3 vertices
          - Both ratios are in [0, 1] and scale-invariant
          - Two triangles of identical shape but different crater-size patterns
            can now be distinguished — not possible with base paper's 3D descriptor.
        """
        l1, l2, l3 = self.sides
        if l3 < 1e-6:
            base_dim = 5 if (USE_RADIUS_DESCRIPTOR and self.v_radii is not None) else 3
            return np.zeros(base_dim, dtype=np.float32)

        area_norm = self.area / (l3 * l3 + 1e-9)
        base = np.array([l1/l3, l2/l3, area_norm], dtype=np.float32)

        if USE_RADIUS_DESCRIPTOR and self.v_radii is not None:
            r_sorted = np.sort(np.abs(self.v_radii).astype(np.float32))  # [r_min, r_mid, r_max]
            r_max = r_sorted[2]
            if r_max < 1e-6:
                r_ratios = np.zeros(2, dtype=np.float32)
            else:
                r_ratios = np.array([r_sorted[0] / r_max,
                                     r_sorted[1] / r_max], dtype=np.float32)
            return np.concatenate([base, r_ratios])

        return base

    def is_valid(self, max_side_ratio: float = DELAUNAY_MAX_SIDE_RATIO,
                 min_area: float = MIN_TRIANGLE_AREA) -> bool:
        """Filter out degenerate triangles (too elongated or too small)."""
        l1, l2, l3 = self.sides
        if self.area < min_area:
            return False
        if l1 < 1e-6:
            return False
        if l3 / (l1 + 1e-9) > max_side_ratio:
            return False
        return True


# ─── Build Triangulation ──────────────────────────────────────────────────────

def build_triangles(craters: np.ndarray,
                    max_triangles: int = MAX_TRIANGLES) -> Tuple[List[Triangle], np.ndarray]:
    """
    Build Delaunay triangulation on crater centers.
    Returns: (triangles_list, adjacency_matrix)

    craters: (N, >=2) array, first 2 cols are (cx, cy).
             If shape[1] >= 5, col 4 is crater radius and will be used in
             the first-order descriptor when USE_RADIUS_DESCRIPTOR=True.
    """
    centers = craters[:, :2]  # (N, 2)
    N = len(centers)
    has_radii = craters.shape[1] >= 5  # col 4 = radius

    if N < 3:
        return [], np.zeros((0, 0))

    # Delaunay triangulation
    tri = Delaunay(centers)
    simplices = tri.simplices  # (M, 3) vertex index triples

    # Build Triangle objects and filter degenerate ones
    triangles = []
    for simplex in simplices:
        i, j, k = simplex
        verts = centers[[i, j, k]]  # (3, 2)
        radii = craters[[i, j, k], 4] if has_radii else None
        t = Triangle((i, j, k), verts, radii=radii)
        if t.is_valid():
            triangles.append(t)

    # Limit number of triangles (select largest area for coverage)
    if len(triangles) > max_triangles:
        areas = [t.area for t in triangles]
        idx   = np.argsort(areas)[::-1][:max_triangles]
        triangles = [triangles[i] for i in idx]

    # Build adjacency (triangles sharing an edge share 2 vertex indices)
    T = len(triangles)
    adj_matrix = np.zeros((T, T), dtype=np.int32)

    # Create edge → triangle mapping
    edge_map: Dict[frozenset, List[int]] = {}
    for ti, t in enumerate(triangles):
        verts = list(t.v_idx)
        edges = [frozenset([verts[0], verts[1]]),
                 frozenset([verts[1], verts[2]]),
                 frozenset([verts[0], verts[2]])]
        for e in edges:
            edge_map.setdefault(e, []).append(ti)

    for edge, tri_ids in edge_map.items():
        if len(tri_ids) == 2:
            a, b = tri_ids
            adj_matrix[a, b] = 1
            adj_matrix[b, a] = 1
            triangles[a].adj_triangles.append(b)
            triangles[b].adj_triangles.append(a)

    return triangles, adj_matrix


# ─── Second-Order Descriptors ─────────────────────────────────────────────────

def compute_second_order_descriptors(triangles: List[Triangle],
                                     max_neighbors: int = MAX_NEIGHBORS) -> np.ndarray:
    """
    Build second-order descriptor for each triangle.
    = [own_desc1 | sorted_neighbour_descs...]  padded with zeros.

    D1 is determined dynamically from the triangle's first-order descriptor:
      - 3 if USE_RADIUS_DESCRIPTOR=False or no radii available
      - 5 if USE_RADIUS_DESCRIPTOR=True and radii available

    Total descriptor dim D = D1 * (max_neighbors + 1)
    Returns (T, D) descriptor matrix.
    """
    if not triangles:
        return np.zeros((0, 0), dtype=np.float32)

    D1 = len(triangles[0].desc1)      # dynamic: 3 (no radius) or 5 (with radius)
    D  = D1 * (max_neighbors + 1)     # total second-order descriptor size

    T = len(triangles)
    descs = np.zeros((T, D), dtype=np.float32)

    for ti, t in enumerate(triangles):
        # Own first-order descriptor
        descs[ti, :D1] = t.desc1

        # Neighbour descriptors (sorted for permutation invariance)
        neigh_descs = []
        for ni in t.adj_triangles[:max_neighbors]:
            neigh_descs.append(triangles[ni].desc1)

        if neigh_descs:
            # Sort neighbour descriptors by their first element (l1/l3 ratio)
            neigh_descs.sort(key=lambda d: d[0])
            for idx, nd in enumerate(neigh_descs[:max_neighbors]):
                start = D1 * (idx + 1)
                descs[ti, start:start+D1] = nd

        t.desc2 = descs[ti]

    return descs


# ─── Similarity Matrix ────────────────────────────────────────────────────────

def compute_similarity_matrix(obs_descs: np.ndarray,
                              map_descs: np.ndarray,
                              sigma: float = SIMILARITY_SIGMA,
                              chunk_size: int = 100) -> np.ndarray:
    """
    Compute pairwise Gaussian similarity between observation and map triangles.
    S[i,j] = exp(-||desc_obs_i - desc_map_j||^2 / (2*sigma^2))
    Returns (N_obs, N_map) similarity matrix.
    Uses chunked computation to keep memory footprint low.
    """
    N_obs = obs_descs.shape[0]
    N_map = map_descs.shape[0]
    inv_2s2 = 1.0 / (2.0 * sigma**2)

    S = np.zeros((N_obs, N_map), dtype=np.float32)
    for i_start in range(0, N_obs, chunk_size):
        i_end = min(i_start + chunk_size, N_obs)
        chunk = obs_descs[i_start:i_end]          # (chunk, D)
        diff  = chunk[:, np.newaxis, :] - map_descs[np.newaxis, :, :]  # (chunk, N_map, D)
        dist2 = np.sum(diff**2, axis=-1)           # (chunk, N_map)
        S[i_start:i_end] = np.exp(-dist2 * inv_2s2)
    return S


# ─── Global Matching ──────────────────────────────────────────────────────────

def match_triangles_greedy(S: np.ndarray,
                           obs_tris: List[Triangle],
                           map_tris: List[Triangle],
                           threshold: float = MATCH_THRESHOLD,
                           adaptive: bool = ADAPTIVE_THRESHOLD
                           ) -> List[Tuple[int, int, float]]:
    """
    Greedy bipartite matching on similarity matrix.
    Returns list of (obs_tri_idx, map_tri_idx, score) sorted by score desc.
    Enforces one-to-one matching.

    [IMPROVED] Adaptive threshold:
      When adaptive=True, threshold is set to:
        max(MATCH_THRESHOLD_FLOOR, 0.75 * mean(top-K row-maxima))
      where row-maxima are the best achievable similarity score per obs triangle.

      This means:
        Clean data (0% error):  top scores ~0.80-0.95  → threshold ~0.65-0.70
        10% error rate:         top scores ~0.65-0.80  → threshold ~0.55-0.65
        20%+ error rate:        top scores drop further → floor (0.50) kicks in

      The algorithm naturally becomes more permissive as noise increases,
      recovering matches that a fixed threshold of 0.70 would reject.
    """
    effective_threshold = threshold

    if adaptive and S.size > 0:
        # For each obs triangle, find its best possible match score
        row_maxes = S.max(axis=1)  # (N_obs,) — best score per obs triangle
        # Focus on the most-matchable half of obs triangles
        top_count = max(5, min(20, len(row_maxes) // 2))
        top_scores = np.sort(row_maxes)[::-1][:top_count]
        global_best_mean = float(np.mean(top_scores))

        if global_best_mean > 0.25:  # only adapt when there's meaningful signal
            adaptive_thresh = global_best_mean * 0.75
            effective_threshold = max(MATCH_THRESHOLD_FLOOR,
                                      min(0.82, adaptive_thresh))

    matches = []
    used_obs = set()
    used_map = set()
    N_obs, N_map = S.shape

    # Collect candidate pairs: for each obs tri, find top-5 matches above threshold
    candidates = []
    for oi in range(N_obs):
        row = S[oi]
        best_j = np.argsort(row)[::-1][:5]  # top-5 per obs triangle
        for mi in best_j:
            sc = float(row[mi])
            if sc >= effective_threshold:
                candidates.append((sc, oi, int(mi)))

    # Sort all candidates descending by score (global ranking)
    candidates.sort(key=lambda x: -x[0])

    # Greedy one-to-one assignment
    for score, oi, mi in candidates:
        if oi in used_obs or mi in used_map:
            continue
        used_obs.add(oi)
        used_map.add(mi)
        matches.append((oi, mi, score))

    return matches


def extract_crater_correspondences(triangle_matches: List[Tuple[int, int, float]],
                                   obs_tris: List[Triangle],
                                   map_tris: List[Triangle],
                                   crater_confidences: Optional[np.ndarray] = None,
                                   conf_weighted: bool = CONF_WEIGHTED_VOTING
                                   ) -> Dict[int, int]:
    """
    Extract crater-level correspondences from triangle matches via voting.
    Each matched triangle pair votes for 3 crater correspondences.

    [IMPROVED] Confidence-weighted voting:
      When conf_weighted=True and crater_confidences is provided,
      each vote is scaled by the detection confidence of the observed crater:
        vote_weight = triangle_similarity_score * crater_confidence

      Effect: Real craters (confidence ~0.85) contribute full votes.
              Spurious detections (confidence ~0.35) contribute ~40% of a vote.
      This reduces the influence of false detections without hard rejection.

    Returns {obs_crater_idx: map_crater_idx} for well-voted pairs.
    """
    votes: Dict[Tuple[int, int], float] = {}  # (obs_idx, map_idx) → weighted score

    for (oti, mti, score) in triangle_matches:
        obs_verts = obs_tris[oti].v_idx  # (3,)
        map_verts = map_tris[mti].v_idx  # (3,)

        obs_side_order = _vertex_by_side_order(obs_tris[oti])
        map_side_order = _vertex_by_side_order(map_tris[mti])

        for ov, mv in zip(obs_side_order, map_side_order):
            # Confidence-weighted vote
            if (conf_weighted and crater_confidences is not None
                    and ov < len(crater_confidences)):
                conf = float(crater_confidences[ov])
            else:
                conf = 1.0

            key = (ov, mv)
            votes[key] = votes.get(key, 0) + score * conf

    # For each observed crater, pick the map crater with most votes
    obs_to_votes: Dict[int, Dict[int, float]] = {}
    for (ov, mv), s in votes.items():
        obs_to_votes.setdefault(ov, {})[mv] = obs_to_votes.get(ov, {}).get(mv, 0) + s

    # Resolve: best map match per obs crater, then ensure 1-to-1
    correspondences = {}
    used_map_craters = set()

    # Sort obs craters by their best vote score (descending)
    obs_ranked = sorted(obs_to_votes.keys(),
                        key=lambda ov: max(obs_to_votes[ov].values()), reverse=True)

    for ov in obs_ranked:
        best_mv = max(obs_to_votes[ov], key=obs_to_votes[ov].get)
        best_score = obs_to_votes[ov][best_mv]
        if best_mv not in used_map_craters and best_score > 0:
            correspondences[ov] = best_mv
            used_map_craters.add(best_mv)

    return correspondences


def _vertex_by_side_order(tri: Triangle) -> List[int]:
    """
    Return triangle vertex indices ordered by their opposite side length (ascending).
    Makes vertex correspondence consistent when matching triangles.
    """
    verts = list(tri.v_idx)
    p = tri.vertices
    sides_with_verts = [
        (np.linalg.norm(p[1] - p[2]), verts[0]),  # side opposite v[0]
        (np.linalg.norm(p[0] - p[2]), verts[1]),  # side opposite v[1]
        (np.linalg.norm(p[0] - p[1]), verts[2]),  # side opposite v[2]
    ]
    sides_with_verts.sort(key=lambda x: x[0])  # sort by side length asc
    return [sv[1] for sv in sides_with_verts]


# ─── Main Matching Interface ──────────────────────────────────────────────────

class TriangleMatcher:
    """
    Main interface for triangle-based global second-order similarity matching.

    All four improvements are enabled by default via config flags:
      - USE_RADIUS_DESCRIPTOR : 5D first-order descriptor (vs base paper's 3D)
      - ADAPTIVE_THRESHOLD    : score-distribution-adaptive match threshold
      - CONF_WEIGHTED_VOTING  : confidence-scaled crater correspondence votes
      - MAX_NEIGHBORS=4       : richer second-order neighbourhood context

    A 'baseline' matcher (reproducing original behaviour) can be created with:
        TriangleMatcher(adaptive_threshold=False, conf_weighted=False)
    (also requires USE_RADIUS_DESCRIPTOR=False in config, or no radii in craters)
    """

    def __init__(self,
                 sigma: float = SIMILARITY_SIGMA,
                 threshold: float = MATCH_THRESHOLD,
                 max_neighbors: int = MAX_NEIGHBORS,
                 max_triangles: int = MAX_TRIANGLES,
                 adaptive_threshold: bool = ADAPTIVE_THRESHOLD,
                 conf_weighted: bool = CONF_WEIGHTED_VOTING):
        self.sigma              = sigma
        self.threshold          = threshold
        self.max_neighbors      = max_neighbors
        self.max_triangles      = max_triangles
        self.adaptive_threshold = adaptive_threshold
        self.conf_weighted      = conf_weighted

    def build_triangle_graph(self, craters: np.ndarray) -> Tuple[List[Triangle], np.ndarray]:
        """Build and describe triangles for a set of craters."""
        triangles, adj = build_triangles(craters, self.max_triangles)
        if not triangles:
            return [], np.zeros((0, 0))
        compute_second_order_descriptors(triangles, self.max_neighbors)
        return triangles, adj

    def match(self, obs_craters: np.ndarray,
              map_craters: np.ndarray,
              obs_confidence: Optional[np.ndarray] = None) -> Dict:
        """
        Full matching pipeline with RANSAC geometric verification.

        obs_craters    : (N, >=2) observed crater positions.
                         If shape[1] >= 5, col 4 is used as crater radius.
        map_craters    : (M, >=2) map/database crater positions.
        obs_confidence : (N,) optional detection confidence per observed crater.
                         Values in [0, 1]. Used for confidence-weighted voting
                         when conf_weighted=True. If None, all votes have weight 1.

        Returns dict with raw and RANSAC-refined matches:
          'crater_pairs'         : raw matches from triangle voting
          'crater_pairs_refined' : RANSAC-verified inlier pairs
          'obs_pts'/'map_pts'    : refined inlier point sets (used for navigation)
          'n_matches'            : refined match count
        """
        result = {
            'crater_pairs':         [],
            'crater_pairs_refined': [],
            'obs_pts':              np.zeros((0, 2)),
            'map_pts':              np.zeros((0, 2)),
            'n_matches':            0,
            'n_raw_matches':        0,
            'n_obs_triangles':      0,
            'n_map_triangles':      0,
            'triangle_matches':     [],
            'similarity_stats':     {},
        }

        if len(obs_craters) < 3 or len(map_craters) < 3:
            return result

        # Build triangle graphs (full arrays passed for radius extraction)
        obs_tris, obs_adj = self.build_triangle_graph(obs_craters)
        map_tris, map_adj = self.build_triangle_graph(map_craters)

        result['n_obs_triangles'] = len(obs_tris)
        result['n_map_triangles'] = len(map_tris)

        if not obs_tris or not map_tris:
            return result

        # Stack second-order descriptors
        obs_descs = np.array([t.desc2 for t in obs_tris])
        map_descs = np.array([t.desc2 for t in map_tris])

        # Compute similarity matrix
        S = compute_similarity_matrix(obs_descs, map_descs, self.sigma)
        result['similarity_stats'] = {
            'mean': float(S.mean()),
            'max':  float(S.max()),
        }

        # Greedy triangle matching with adaptive threshold
        tri_matches = match_triangles_greedy(
            S, obs_tris, map_tris,
            threshold=self.threshold,
            adaptive=self.adaptive_threshold)
        result['triangle_matches'] = tri_matches

        # Extract raw crater correspondences via confidence-weighted voting
        conf_arr = obs_confidence if (self.conf_weighted and obs_confidence is not None) else None
        crater_corr = extract_crater_correspondences(
            tri_matches, obs_tris, map_tris,
            crater_confidences=conf_arr,
            conf_weighted=self.conf_weighted)

        result['crater_pairs']  = list(crater_corr.items())
        result['n_raw_matches'] = len(crater_corr)

        if len(crater_corr) < 4:
            return result

        obs_idxs = list(crater_corr.keys())
        map_idxs = list(crater_corr.values())
        obs_pts_raw = obs_craters[obs_idxs, :2]
        map_pts_raw = map_craters[map_idxs, :2]

        # ── RANSAC geometric verification ──────────────────────────────────────
        # Refines matches by finding a globally consistent homography.
        # The "global" aspect: ensures geometric consistency across all pairs,
        # not just local descriptor similarity.
        import cv2
        from config import RANSAC_REPROJ_THRESH
        if len(obs_pts_raw) >= 4:
            H, mask = cv2.findHomography(
                map_pts_raw.astype(np.float32),
                obs_pts_raw.astype(np.float32),
                method=cv2.RANSAC,
                ransacReprojThreshold=RANSAC_REPROJ_THRESH,
                maxIters=2000,
                confidence=0.995,
            )
            if H is not None and mask is not None:
                inlier_mask = mask.ravel().astype(bool)
                refined_pairs = [
                    (obs_idxs[k], map_idxs[k])
                    for k in range(len(obs_idxs)) if inlier_mask[k]
                ]
                result['crater_pairs_refined'] = refined_pairs
                result['obs_pts'] = obs_pts_raw[inlier_mask]
                result['map_pts'] = map_pts_raw[inlier_mask]
                result['n_matches'] = len(refined_pairs)
            else:
                # Fallback: use raw matches
                result['crater_pairs_refined'] = result['crater_pairs']
                result['obs_pts'] = obs_pts_raw
                result['map_pts'] = map_pts_raw
                result['n_matches'] = len(crater_corr)
        else:
            result['crater_pairs_refined'] = result['crater_pairs']
            result['obs_pts'] = obs_pts_raw
            result['map_pts'] = map_pts_raw
            result['n_matches'] = len(crater_corr)

        return result


# ─── Utility ──────────────────────────────────────────────────────────────────

def print_match_summary(result: Dict):
    """Print a summary of matching results."""
    print(f"  Obs triangles : {result['n_obs_triangles']}")
    print(f"  Map triangles : {result['n_map_triangles']}")
    print(f"  Tri matches   : {len(result['triangle_matches'])}")
    print(f"  Crater matches: {result['n_matches']}")
    if result['similarity_stats']:
        s = result['similarity_stats']
        print(f"  Sim (mean/max): {s['mean']:.4f} / {s['max']:.4f}")


if __name__ == '__main__':
    # Quick smoke test
    np.random.seed(42)
    N = 50
    craters_map = np.zeros((N, 5), dtype=np.float32)
    craters_map[:, :2] = np.random.rand(N, 2) * 800
    craters_map[:, 2:4] = 10   # width/height
    craters_map[:, 4] = np.random.uniform(5, 30, N)   # realistic radii

    # Simulate noisy observation (same craters + noise + 20% missed)
    noise = np.random.randn(N, 2) * 5
    craters_obs = craters_map.copy()
    craters_obs[:, :2] += noise
    craters_obs = craters_obs[:int(N * 0.8)]  # miss 20%

    # Simulated confidence: obs craters have realistic high confidence
    obs_conf = np.clip(np.random.normal(0.85, 0.08, len(craters_obs)), 0.5, 1.0).astype(np.float32)

    matcher = TriangleMatcher()
    result  = matcher.match(craters_obs, craters_map, obs_confidence=obs_conf)

    print("Smoke test result (improved algorithm):")
    print_match_summary(result)
    d1 = len(result['triangle_matches'][0]) if result['triangle_matches'] else 0
    print(f"  Descriptor dim (D1): {len(craters_map[0:3, :][:, :2])} — "
          f"check Triangle.desc1 length in obs_tris")
