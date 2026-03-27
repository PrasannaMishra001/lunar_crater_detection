"""
Triangle-Based Global Second-Order Similarity Crater Matching
=============================================================
Core algorithm from: "Lunar Crater Matching With Triangle-Based Global
Second-Order Similarity for Precision Navigation"

Algorithm:
1. Build Delaunay triangulation on crater centers
2. For each triangle, compute FIRST-ORDER descriptor:
   - Normalized sorted side ratios (l1/l3, l2/l3) — invariant to scale/rotation/translation
3. Build adjacency graph (triangles sharing an edge)
4. For each triangle, compute SECOND-ORDER descriptor:
   - Concatenate own first-order descriptor with sorted neighbor descriptors
   - Captures local geometric neighborhood context
5. Build similarity matrix using Gaussian kernel
6. Greedy global matching with consistency enforcement
7. Return matched crater correspondences
"""
import numpy as np
from scipy.spatial import Delaunay
from typing import List, Tuple, Dict, Optional
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (MAX_NEIGHBORS, SIMILARITY_SIGMA, MATCH_THRESHOLD,
                    DELAUNAY_MAX_SIDE_RATIO, MIN_TRIANGLE_AREA, MAX_TRIANGLES)


# ─── Triangle Descriptor ──────────────────────────────────────────────────────

class Triangle:
    """Represents a triangle formed by three crater centers."""
    __slots__ = ['v_idx', 'vertices', 'sides', 'desc1', 'desc2',
                 'adj_triangles', 'area']

    def __init__(self, v_idx: Tuple[int, int, int], vertices: np.ndarray):
        """
        v_idx   : indices into the crater array (i, j, k)
        vertices: (3, 2) array of (x, y) positions
        """
        self.v_idx       = tuple(v_idx)
        self.vertices    = vertices          # (3,2)
        self.sides       = self._compute_sides()
        self.area        = self._compute_area()
        self.desc1       = self._compute_first_order()
        self.desc2       = None              # filled after adjacency is built
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
        d = (l1/l3, l2/l3)  where l1 <= l2 <= l3 are sorted side lengths.
        Also include perimeter-normalized area for extra discriminability.
        """
        l1, l2, l3 = self.sides
        if l3 < 1e-6:
            return np.zeros(3, dtype=np.float32)
        # Normalized side ratios + normalized area
        area_norm = self.area / (l3 * l3 + 1e-9)
        return np.array([l1/l3, l2/l3, area_norm], dtype=np.float32)

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
    craters: (N, >=2) array, first 2 cols are (cx, cy)
    """
    centers = craters[:, :2]  # (N, 2)
    N = len(centers)

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
        t = Triangle((i, j, k), verts)
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
    = [own_desc1 | sorted_neighbor_descs...]  padded with zeros.
    Returns (T, D) descriptor matrix.
    """
    D1 = 3      # first-order descriptor size
    D  = D1 * (max_neighbors + 1)   # total descriptor: own + up to 3 neighbors

    T = len(triangles)
    descs = np.zeros((T, D), dtype=np.float32)

    for ti, t in enumerate(triangles):
        # Own first-order descriptor
        descs[ti, :D1] = t.desc1

        # Neighbor descriptors (sorted for permutation invariance)
        neigh_descs = []
        for ni in t.adj_triangles[:max_neighbors]:
            neigh_descs.append(triangles[ni].desc1)

        if neigh_descs:
            # Sort neighbor descriptors by their first element (l1/l3 ratio)
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
    Always uses chunked computation to keep memory footprint low.
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
                           threshold: float = MATCH_THRESHOLD) -> List[Tuple[int, int, float]]:
    """
    Greedy bipartite matching on similarity matrix.
    Returns list of (obs_tri_idx, map_tri_idx, score) sorted by score desc.
    Enforces one-to-one matching.
    Memory-efficient: uses argsort per row instead of flattening full matrix.
    """
    matches = []
    used_obs = set()
    used_map = set()
    N_obs, N_map = S.shape

    # Collect candidate pairs efficiently: for each obs, find top matches
    candidates = []
    for oi in range(N_obs):
        row = S[oi]
        best_j = np.argsort(row)[::-1][:5]  # top-5 per obs tri
        for mi in best_j:
            sc = float(row[mi])
            if sc >= threshold:
                candidates.append((sc, oi, int(mi)))

    # Sort all candidates descending by score
    candidates.sort(key=lambda x: -x[0])

    for score, oi, mi in candidates:
        if oi in used_obs or mi in used_map:
            continue
        used_obs.add(oi)
        used_map.add(mi)
        matches.append((oi, mi, score))

    return matches


def extract_crater_correspondences(triangle_matches: List[Tuple[int, int, float]],
                                   obs_tris: List[Triangle],
                                   map_tris: List[Triangle]) -> Dict[int, int]:
    """
    Extract crater-level correspondences from triangle matches via voting.
    Each matched triangle pair votes for 3 crater correspondences.
    Returns {obs_crater_idx: map_crater_idx} for well-voted pairs.
    """
    votes: Dict[Tuple[int, int], float] = {}  # (obs_idx, map_idx) → score

    for (oti, mti, score) in triangle_matches:
        obs_verts = obs_tris[oti].v_idx  # (3,)
        map_verts = map_tris[mti].v_idx  # (3,)

        # Match vertices by sorted side contributions
        # Triangle vertices are associated with their opposite sides
        # obs side order matches map side order (both sorted ascending)
        # Get side lengths to match vertex ordering
        obs_side_order = _vertex_by_side_order(obs_tris[oti])
        map_side_order = _vertex_by_side_order(map_tris[mti])

        for ov, mv in zip(obs_side_order, map_side_order):
            key = (ov, mv)
            votes[key] = votes.get(key, 0) + score

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
    This makes vertex correspondence consistent when matching triangles.
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
    """

    def __init__(self,
                 sigma: float = SIMILARITY_SIGMA,
                 threshold: float = MATCH_THRESHOLD,
                 max_neighbors: int = MAX_NEIGHBORS,
                 max_triangles: int = MAX_TRIANGLES):
        self.sigma         = sigma
        self.threshold     = threshold
        self.max_neighbors = max_neighbors
        self.max_triangles = max_triangles

    def build_triangle_graph(self, craters: np.ndarray) -> Tuple[List[Triangle], np.ndarray]:
        """Build and describe triangles for a set of craters."""
        triangles, adj = build_triangles(craters, self.max_triangles)
        if not triangles:
            return [], np.zeros((0, 0))
        compute_second_order_descriptors(triangles, self.max_neighbors)
        return triangles, adj

    def match(self, obs_craters: np.ndarray,
              map_craters: np.ndarray) -> Dict:
        """
        Full matching pipeline with RANSAC geometric verification.
        obs_craters: (N, >=2) observed crater positions
        map_craters: (M, >=2) map/database crater positions

        Returns dict with raw and RANSAC-refined matches.
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

        # Build triangle graphs
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

        # Greedy triangle matching
        tri_matches = match_triangles_greedy(S, obs_tris, map_tris, self.threshold)
        result['triangle_matches'] = tri_matches

        # Extract raw crater correspondences via voting
        crater_corr = extract_crater_correspondences(tri_matches, obs_tris, map_tris)
        result['crater_pairs']  = list(crater_corr.items())
        result['n_raw_matches'] = len(crater_corr)

        if len(crater_corr) < 4:
            return result

        obs_idxs = list(crater_corr.keys())
        map_idxs = list(crater_corr.values())
        obs_pts_raw = obs_craters[obs_idxs, :2]
        map_pts_raw = map_craters[map_idxs, :2]

        # ── RANSAC geometric verification ─────────────────────────────────
        # Refines matches by finding a globally consistent homography.
        # This is the "global" aspect of the algorithm — ensures geometric
        # consistency across all matched pairs, not just local similarity.
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
    craters_map = np.random.rand(N, 5) * 800
    craters_map[:, 2:4] = 10  # set w,h

    # Simulate noisy observation (same craters + noise + 20% missed)
    noise = np.random.randn(N, 2) * 5
    craters_obs = craters_map.copy()
    craters_obs[:, :2] += noise
    craters_obs = craters_obs[:int(N * 0.8)]  # miss 20%

    matcher = TriangleMatcher()
    result  = matcher.match(craters_obs, craters_map)

    print("Smoke test result:")
    print_match_summary(result)
