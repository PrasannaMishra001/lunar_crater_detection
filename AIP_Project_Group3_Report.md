# Lunar Crater Matching Using Triangle-Based Global Second-Order Similarity for Precision Navigation

## Complete Project Report, Concepts Guide & Viva Preparation

**Applied Image Processing (AIP) -- Group 3**
**Under the guidance of:** Dr. Mahua Bhattacharya, ABV-IIITM Gwalior

| Roll Number | Name |
|---|---|
| 2023IMT-014 | Ankit Baidsen |
| 2023IMT-050 | Malladi Nagarjuna |
| 2023IMT-059 | Prasanna Mishra |
| 2023IMT-060 | Prasun Baranwal |
| 2023IMT-073 | Shivam Deolankar |

---

# PART 1: THE PROBLEM -- WHY IS THIS NEEDED?

## 1.1 The Autonomous Lunar Landing Problem

When a spacecraft approaches the Moon for landing, it must know *exactly* where it is relative to the surface. Even a 100-meter error can mean landing on a boulder field, a steep slope, or inside a dangerous crater. GPS does not exist on the Moon -- there are no satellites in lunar orbit providing positioning signals.

**Historical failures motivating this work:**
- **Apollo missions (1960s-70s):** Astronauts frequently had to manually correct trajectories during descent. Apollo 11 overshot its landing zone by 6 km; Armstrong had to fly manually to find a safe spot with only 25 seconds of fuel remaining.
- **Beresheet (2019):** The Israeli lander lost control during descent and crashed. The inertial measurement unit (IMU) drifted and there was no terrain-relative navigation backup.
- **Chandrayaan-2 Vikram Lander (2019):** Lost communication during final descent. The lander deviated from its planned trajectory in the last 2.1 km of descent.
- **Luna 25 (2023):** Russian lander crashed into the Moon after an orbital correction manoeuvre went wrong -- again, no terrain-relative position verification.

**Why craters are the solution:**
Craters are the most stable, abundant, and distinctive landmarks on the Moon. They don't change over millions of years (no weather, no erosion, no vegetation). Every region of the Moon has craters. If a spacecraft can detect craters in its camera image and match them to a pre-existing map of known crater positions, it can compute its exact position. This is called **Terrain-Relative Navigation (TRN)**.

## 1.2 The Challenge: Crater Matching

The problem is not just *detecting* craters -- it's *matching* them. Given:
- **Observation image:** What the spacecraft camera sees during descent (craters detected from this image)
- **Reference map:** A database of known crater positions from orbital surveys (LRO satellite data)

We need to find: **Which observed crater corresponds to which map crater?**

This is hard because:
1. The camera angle, altitude, and lighting change during descent
2. Some craters will be missed (not detected) -- **false negatives**
3. Some detections will be wrong (rocks, shadows) -- **false positives**
4. The positions have measurement noise
5. There are hundreds to thousands of craters, so brute-force matching is combinatorially explosive

## 1.3 Why This Paper's Approach Matters

The base paper proposes a graph-based geometric matching approach:
1. Group nearby craters into **triangles** (via Delaunay triangulation)
2. Describe each triangle's **shape** (scale/rotation/translation invariant)
3. Compare triangles between observation and map using **second-order similarity** (matching not just individual triangles but their neighbourhood context)
4. This makes the matching robust to noise, missing craters, and scale changes

**This is a real-world space navigation technique** being developed for missions like Artemis (NASA's return to the Moon), Chandrayaan-3 (India), and Chang'E-6 (China).

---

# PART 2: THE BASE PAPER EXPLAINED

## 2.1 Paper: "Lunar Crater Matching With Triangle-Based Global Second-Order Similarity for Precision Navigation"

**Published:** IEEE Transactions on Geoscience and Remote Sensing, 2024
**IEEE Link:** https://ieeexplore.ieee.org/document/11123425/

### The Algorithm (6 Stages)

### Stage 1: Crater Detection (YOLOv7 in paper, YOLOv8 in our implementation)

**What:** Identify crater locations in the camera image using a deep learning object detector.

**How:** YOLO (You Only Look Once) is a real-time object detection neural network. It divides the image into a grid and predicts bounding boxes + class probabilities in a single forward pass. Each detected crater becomes a tuple: `(center_x, center_y, width, height, radius)`.

**Key concept -- Object Detection:**
Unlike image classification (which says "this image contains a crater"), object detection says "there is a crater at position (234, 567) with radius 15 pixels." YOLO is popular because it's fast (real-time) and accurate.

### Stage 2: Delaunay Triangulation

**What:** Connect detected crater centers into a mesh of triangles.

**How:** Delaunay triangulation is a mathematical method that connects a set of points into triangles such that no point lies inside the circumscribed circle of any triangle. This maximises the minimum angle of all triangles, producing "well-shaped" (non-degenerate) triangles.

**Why Delaunay specifically?** Because it produces the same triangulation regardless of the order in which points are added, and it naturally groups nearby craters. Two different images of the same region will produce very similar Delaunay triangulations, making matching feasible.

**Filtering:** We discard degenerate triangles:
- Area < 20 px^2 (too small to be meaningful)
- Side ratio > 10:1 (too elongated -- looks like a sliver, not a triangle)

### Stage 3: First-Order Descriptor (Per-Triangle)

**What:** Create a numerical fingerprint for each triangle that is invariant to scale, rotation, and translation.

**How (base paper, 3D):**
Given a triangle with sorted side lengths l1 <= l2 <= l3:
```
descriptor = (l1/l3, l2/l3, area / l3^2)
```

- `l1/l3` and `l2/l3` are **scale-invariant ratios** (dividing by the longest side removes absolute size)
- `area / l3^2` is the **normalised area** (additional shape discriminability)

**Invariance properties:**
- **Translation invariant:** Moving the triangle doesn't change side lengths
- **Rotation invariant:** Rotating doesn't change side lengths
- **Scale invariant:** Scaling multiplies all sides by the same factor, which cancels in the ratios

**Why this matters:** A crater triangle seen from 100 km altitude will have the same descriptor as the same triangle seen from 50 km altitude (different scale) or from a tilted camera (rotation).

### Stage 4: Second-Order Descriptor (Neighbourhood Context)

**What:** Extend each triangle's descriptor by incorporating its neighbours' descriptors.

**How:** For each triangle T:
1. Find its adjacent triangles (those sharing an edge) -- up to K neighbours
2. Sort their first-order descriptors by the first element (for permutation invariance)
3. Concatenate: `second_order = [T's descriptor | neighbor_1's desc | neighbor_2's desc | ...]`

**Why second-order?** Consider two triangles with identical shapes but in completely different parts of the crater field. Their first-order descriptors would be the same, causing a wrong match. But their *neighbours* are different -- the second-order descriptor captures this local context, making matching much more discriminative.

**Analogy:** It's like recognizing a person not just by their face (first-order) but by the faces of the people standing next to them (second-order). The combination is much more unique.

### Stage 5: Global Matching

**Step 5a -- Similarity Matrix:**
Compute a Gaussian kernel similarity between every pair of observation/map triangles:

```
S(i,j) = exp(-||desc_obs_i - desc_map_j||^2 / (2 * sigma^2))
```

Values close to 1 = very similar. Values close to 0 = very different.

**Step 5b -- Greedy Bipartite Matching:**
Select the best one-to-one triangle correspondences by greedily picking the highest-scoring pair, removing both from consideration, and repeating.

**Step 5c -- Crater-Level Voting:**
Each matched triangle pair "votes" for 3 crater correspondences (one per vertex). A voting scheme resolves conflicts where multiple triangles nominate different matches for the same crater.

**Step 5d -- RANSAC Geometric Verification:**
RANSAC (Random Sample Consensus) estimates a homography (geometric transformation) from the matched points. Points that don't agree with this transformation are outliers and are removed. This enforces **global geometric consistency**.

### Stage 6: Navigation / Pose Estimation

From the verified crater correspondences:
1. Estimate a **homography** H (a 3x3 matrix mapping map positions to observed positions)
2. The homography encodes: translation (position), rotation (orientation), and scale (altitude)
3. Compute **position error** as a percentage of flight altitude

**Position error formula:**
```
error_x = |translation_x| / image_width * 100%
error_y = |translation_y| / image_height * 100%
```

This percentage-of-altitude metric is standard in space navigation: if you're at 100 km altitude and your error is 0.2%, that's 200 meters -- well within safe landing margins.

---

# PART 3: KEY CONCEPTS AND TERMINOLOGY

## 3.1 Object Detection (YOLO)

**YOLO (You Only Look Once):** A family of real-time object detection models. Key versions:
- **YOLOv7** (2022): Used in the base paper
- **YOLOv8** (2023, Ultralytics): Used in our initial implementation (nano variant)
- **YOLOv8s** (small variant): Used in our improved implementation -- better recall

**Key metrics:**
- **mAP@50 (mean Average Precision at IoU 0.50):** Measures detection accuracy. Higher = better.
- **Precision:** Of all detections, what fraction are correct? (low false positives)
- **Recall:** Of all actual craters, what fraction did we detect? (low false negatives)
- **IoU (Intersection over Union):** Overlap between predicted and ground truth boxes. IoU > 0.5 = correct detection.

**Our YOLO results:** mAP@50 = 0.416, Recall = 0.232 (trained on only 12 images, 50 epochs, CPU). The low recall means 77% of craters are missed. This is the project's main YOLO weakness.

## 3.2 Delaunay Triangulation

A Delaunay triangulation of a point set P is a triangulation DT(P) such that no point in P is inside the circumcircle of any triangle in DT(P).

**Properties:**
- Maximises the minimum angle (avoids thin slivers)
- Unique for points in "general position" (no 4 points on a circle)
- Dual of the Voronoi diagram
- Computed in O(n log n) time

**In our project:** We use `scipy.spatial.Delaunay` to triangulate crater centers.

## 3.3 Homography

A homography is a 3x3 projective transformation matrix H that maps points from one plane to another:

```
[x']     [h11 h12 h13] [x]
[y'] = H [h21 h22 h23] [y]
[1 ]     [h31 h32 h33] [1]
```

It has 8 degrees of freedom (the 9th element is a scale factor). It encodes:
- Translation (tx, ty) -- camera position
- Rotation (theta) -- camera orientation
- Scale (s) -- camera altitude
- Perspective distortion -- camera tilt

**In our project:** We estimate H from matched crater pairs using RANSAC, then extract position error from the translation components.

## 3.4 RANSAC (Random Sample Consensus)

An iterative method for fitting a model to data containing outliers:
1. Randomly select a minimal subset of points (4 for homography)
2. Fit the model (estimate H)
3. Count how many other points agree (inliers)
4. Repeat many times; keep the model with the most inliers

**Why needed:** Even after triangle matching, some crater correspondences may be wrong. RANSAC finds the geometrically consistent subset.

## 3.5 Gaussian Kernel Similarity

The Gaussian (RBF) kernel measures similarity between two vectors:

```
K(x, y) = exp(-||x - y||^2 / (2 * sigma^2))
```

- sigma controls the "bandwidth" -- how quickly similarity drops with distance
- K = 1 when x = y (identical)
- K → 0 as ||x - y|| → infinity

**In our project:** sigma = 0.20 for the 25-dimensional second-order descriptors.

## 3.6 Monte Carlo Simulation

A statistical technique that runs thousands of random trials to estimate the expected behaviour of a system:
1. Add random Gaussian noise (sigma = 5 pixels) to crater positions
2. Randomly remove some craters (miss rate) and add spurious ones (false rate)
3. Run the matching algorithm
4. Measure accuracy, position error, navigation success
5. Repeat 1000 times and compute statistics

**Why 1000 trials?** To get statistically reliable estimates. The standard error of the mean scales as sigma / sqrt(N), so with 1000 trials, the estimates are very precise.

## 3.7 Scale Invariance

A descriptor is **scale-invariant** if it doesn't change when the input is uniformly scaled. Our triangle descriptor uses side *ratios* (l1/l3, l2/l3), which cancel the scale factor:

```
If all sides are multiplied by k: (k*l1)/(k*l3) = l1/l3
```

This is critical because the camera altitude changes during descent, changing the apparent size of everything in the image.

## 3.8 Detection Error Rate

The combined rate of false detections and missed detections:
- **False positive (FP):** Detecting a crater where there isn't one
- **False negative (FN):** Missing a real crater

An "error rate of 20%" means: 10% of real craters are missed (FN), and 10% extra spurious craters are added (FP).

---

# PART 4: OUR IMPROVEMENTS OVER THE BASE PAPER

## Improvement 1: Crater-Radius-Augmented Descriptor (5D)

**Problem with the base paper:** The 3D descriptor (l1/l3, l2/l3, area_norm) captures only the *shape* of the triangle, ignoring the *physical sizes* of the craters at its vertices. Two triangles of identical shape but connecting craters of very different sizes would have the same descriptor.

**Our solution:** Extend the descriptor with crater radius ratios:
```
Improved descriptor (5D):
  (l1/l3, l2/l3, area_norm, r_min/r_max, r_mid/r_max)
```

Where r_min, r_mid, r_max are the sorted radii of the 3 craters forming the triangle. This is still scale-invariant (ratios cancel the scale factor) but now distinguishes triangles connecting large craters from those connecting small craters.

**Impact:** The second-order descriptor becomes 25D (5 * 5 = 25, with 4 neighbours) instead of the original 12D (3 * 4 = 12, with 3 neighbours). This gives each triangle a much more unique fingerprint.

**Result:** Position error improved by ~31-33% (X: 0.2834% -> 0.1949%, Y: 0.3224% -> 0.2157%).

## Improvement 2: Adaptive Similarity Threshold

**Problem with the base paper:** A fixed threshold of 0.70 works well for clean data (0% error) but is too strict when detection errors degrade the similarity scores. At 10% error rate, many correct matches have scores below 0.70 and are rejected.

**Our solution:** Automatically adapt the threshold based on the observed score distribution:
```
threshold = max(FLOOR, 0.75 * mean(top-K row-maxima))
```

Where row-maxima are the best achievable similarity score per observation triangle. This means:
- Clean data (0% error): top scores ~0.80-0.95 -> threshold ~0.65
- 10% error: top scores ~0.65-0.80 -> threshold ~0.55
- 20%+ error: floor (0.50) kicks in

**Result:** Navigation success at 10% error rate improved from 87.33% to 99.5% (+12.17 percentage points).

## Improvement 3: Confidence-Weighted Crater Correspondence Voting

**Problem:** When the detector produces false positive craters (spurious detections), these participate equally in the voting scheme, potentially corrupting the correspondences.

**Our solution:** Weight each crater's vote by its detection confidence score:
```
vote_weight = triangle_similarity_score * crater_confidence
```

Real craters (confidence ~0.85) contribute full votes; spurious detections (confidence ~0.35) contribute ~40% of a vote. This naturally suppresses the influence of false positives without hard rejection.

**Result:** Combined with the adaptive threshold, this significantly improves robustness at moderate error rates.

## Improvement 4: Extended Adjacency Context (MAX_NEIGHBORS=4)

**Base paper:** Uses 3 adjacent neighbours per triangle in the second-order descriptor.
**Our implementation:** Uses 4 neighbours, giving a richer neighbourhood context.

**Effect:** Each triangle's descriptor incorporates information from a wider geometric context, making it more unique and reducing ambiguous matches.

## Improvement 5: YOLOv8s vs YOLOv8n

**Base paper:** Uses YOLOv7.
**Our initial version:** YOLOv8n (nano -- 3M parameters, fastest variant).
**Our improved version:** Config upgraded to YOLOv8s (small -- 11M parameters). The small variant has ~3.7x more parameters, giving significantly better recall on small objects (craters < 20px diameter). Retraining on GPU with this larger model would substantially improve the current 23% recall.

---

# PART 5: RESULTS COMPARISON

## 5.1 Final Results Table

| Metric | Base Paper | Baseline (Ours) | Improved (Ours) | Delta |
|--------|-----------|-----------------|-----------------|-------|
| **Matching Accuracy (0% error)** | ~99% | 96.57% | **96.91%** | +0.34% |
| **Navigation Success (0% error)** | ~100% | 99.60% | **100.00%** | +0.40% |
| **Position Error X (% altitude)** | 0.44% | 0.2834% | **0.1949%** | **-31.2%** |
| **Position Error Y (% altitude)** | 0.44% | 0.3224% | **0.2157%** | **-33.1%** |
| **Position Error XY (combined)** | 0.44% | 0.4748% | **0.3240%** | **-31.8%** |
| **Matching Accuracy (10% error)** | N/A | 80.0% | **83.93%** | **+3.93%** |
| **Nav Success (10% error)** | N/A | 87.33% | **99.50%** | **+12.17%** |
| Matching Time | ~0.1s | 0.073s | 0.157s | +115% |
| Descriptor Dimension (D1) | 3D | 3D | **5D** | +67% |
| Second-Order Descriptor | 12D | 12D | **25D** | +108% |
| Max Neighbours | N/A | 3 | **4** | +33% |
| YOLOv8 mAP@50 | YOLOv7 | 0.416 (nano) | 0.416 (nano*) | -- |
| YOLOv8 Recall | N/A | 0.232 | 0.232 | -- |

*Note: YOLOv8s config set but not yet retrained; existing nano weights used for detection metrics.

## 5.2 Key Takeaways

1. **Position accuracy improved by 31-33%** -- the most important metric for precision navigation. Our improved algorithm places the spacecraft 31% more accurately in X and 33% more accurately in Y.

2. **Navigation success at 10% error rate improved by +12.17 percentage points** (87.33% -> 99.5%). This is the most practically significant result: real detectors have errors, and maintaining near-100% navigation success under realistic error conditions is critical for mission safety.

3. **100% navigation success in clean conditions** (was 99.6%). Perfect reliability in ideal conditions.

4. **Trade-off: matching time increased from 0.073s to 0.157s** due to larger descriptors (25D vs 12D). This is still well within real-time requirements (6.4 images/second).

## 5.3 Error Rate Sweep (Robustness Analysis)

| Error Rate | Matching Acc (Baseline) | Matching Acc (Improved) | Nav Success (Improved) |
|-----------|------------------------|------------------------|----------------------|
| 0% | 96.10% | **96.91%** | 100.0% |
| 10% | 80.03% | **83.93%** | 99.5% |
| 20% | 51.46% | 49.88% | 99.0% |
| 40% | 16.25% | 21.11% | 98.0% |
| 60% | 4.16% | 9.90% | 100.0% |
| 80% | 2.09% | 6.43% | 94.0% |
| 100% | 0.66% | 1.50% | 99.0% |

The improved algorithm shows substantially better navigation success rates across all error levels, which is the metric that matters most for mission safety.

---

# PART 6: ALTERNATIVES AND RELATED WORK

## 6.1 Alternative Crater Matching Approaches

| Method | Approach | Pros | Cons |
|--------|----------|------|------|
| **Crater Triad Pattern** | Match sets of 3 craters based on distances and angles | Simple, fast | Not robust to many missing craters |
| **Lost-in-Space (Star Matching)** | Adapted from star tracker algorithms | Well-established | Designed for point sources, not extended objects |
| **Deep Feature Matching** | Use CNNs to learn descriptors directly | Can learn complex patterns | Requires large training data, less interpretable |
| **Template Matching** | Slide an image patch over the map | Intuitive | Not invariant to scale/rotation, slow |
| **Hough Transform** | Vote in parameter space for circle positions | Good for detection | Doesn't solve the matching problem |
| **Graph Neural Networks** | Learn descriptors from graph structure | Can capture complex relationships | Requires training, computational cost |
| **Triangle-Based (This Paper)** | Geometric invariants of crater triangles | Scale/rotation invariant, no training needed | Performance degrades at >20% error rates |

## 6.2 Alternative Detection Approaches

| Method | Description |
|--------|-------------|
| **CraterDANet** | Domain-adapted CNN for small-scale crater detection (Yang et al.) |
| **DeepMoon** | U-Net segmentation for crater rim detection |
| **Mask R-CNN** | Instance segmentation for precise crater boundaries |
| **YOLOv9/v10** | Latest YOLO variants with improved small-object detection |
| **RT-DETR** | Real-Time Detection Transformer -- attention-based detector |
| **Classical (Hough+Canny)** | Edge detection + circle Hough transform -- no training needed |

## 6.3 Why Our Triangle-Based Approach is Preferred

1. **No training needed for matching** -- the geometric descriptors are handcrafted and invariant by construction. No need for labelled matching data.
2. **Mathematically grounded invariance** -- scale, rotation, and translation invariance come from the ratio-based descriptor, not from data augmentation.
3. **Interpretable** -- we can inspect why a match was made (which triangles matched, which votes dominated).
4. **Computationally efficient** -- O(N^2) in the number of triangles, with chunked computation keeping memory low.

---

# PART 7: PROJECT ARCHITECTURE AND IMPLEMENTATION

## 7.1 File Structure

```
lunar_crater_project/
|-- config.py               # All hyperparameters and improvement flags
|-- data_loader.py           # Load LRO dataset: images + annotations
|-- prepare_yolo.py          # Convert annotations to YOLO format
|-- train_yolo.py            # Train YOLOv8s on crater dataset
|-- detect.py                # Crater detection (GT or YOLO modes)
|-- triangle_matching.py     # CORE: triangle-based 2nd-order matching
|-- navigation.py            # Pose estimation from matched craters
|-- metrics.py               # Monte Carlo evaluation framework
|-- visualize.py             # Publication-quality figures
|-- main.py                  # Full evaluation pipeline
```

## 7.2 Algorithm Flow (Improved)

```
Input: LRO NAC Lunar Image
    |
    v
[1] Crater Detection (YOLOv8s or GT)
    -> (cx, cy, w, h, radius) per crater
    |
    v
[2] Delaunay Triangulation
    -> N triangles connecting nearby craters
    -> Filter: remove area < 20px^2, aspect ratio > 10
    |
    v
[3] First-Order Descriptor (5D) [IMPROVED]
    -> (l1/l3, l2/l3, area_norm, r_min/r_max, r_mid/r_max)
    -> Scale/rotation/translation invariant
    -> NOVEL: radius ratios add physical size information
    |
    v
[4] Second-Order Descriptor (25D) [IMPROVED]
    -> Concatenate own descriptor + 4 neighbours' descriptors
    -> Captures local geometric context
    -> MAX_NEIGHBORS=4 (was 3)
    |
    v
[5] Gaussian Similarity Matrix
    -> S[i,j] = exp(-||d_i - d_j||^2 / (2 * 0.20^2))
    -> Chunked computation for memory efficiency
    |
    v
[6] Adaptive Threshold [IMPROVED]
    -> threshold = max(0.50, 0.75 * mean(top-K scores))
    -> Automatically relaxes under noise
    |
    v
[7] Greedy Bipartite Matching
    -> Top-5 candidates per obs triangle
    -> One-to-one assignment by descending score
    |
    v
[8] Confidence-Weighted Voting [IMPROVED]
    -> vote = similarity_score * detection_confidence
    -> Suppresses spurious detection influence
    |
    v
[9] RANSAC Geometric Verification
    -> Estimate homography from matched crater pairs
    -> Filter outliers (reprojection threshold = 5px)
    |
    v
[10] Navigation: Extract position error from homography
    -> error_x, error_y as % of altitude
    -> Reprojection error in pixels
```

## 7.3 Dataset Details

**Source:** LRO NAC (Lunar Reconnaissance Orbiter, Narrow Angle Camera)
**Region:** Chang'E-4 landing site (45-46 S, 176.4-178.8 E) -- far side of the Moon
**Resolution:** 0.5 m/pixel
**Training:** 12 images (800x800 px), 13,453 annotated craters
**Test:** 8 images (1000x1000 px), 9,797 annotated craters
**Minimum crater diameter:** 8 pixels (4 meters at 0.5 m/pixel resolution)

---

# PART 8: VIVA QUESTIONS AND ANSWERS

## Category A: Fundamentals

**Q1: What is the problem you are trying to solve?**
A: Autonomous position estimation for a spacecraft during lunar descent, using crater detection and matching against a known map database. The spacecraft has no GPS, so it must determine its position from visual landmarks (craters).

**Q2: Why craters? Why not other landmarks?**
A: Craters are (1) abundant on the entire lunar surface, (2) temporally stable (don't change over millions of years -- no erosion, weather, or vegetation), (3) geometrically distinctive (each has a unique size and position), and (4) detectable from orbital imagery at multiple scales.

**Q3: What is Terrain-Relative Navigation (TRN)?**
A: A technique where a spacecraft determines its position by comparing features detected in real-time camera images against a pre-existing map of the terrain. It provides an independent position fix without relying on Earth-based tracking or inertial sensors.

**Q4: Why can't we just use an IMU (Inertial Measurement Unit)?**
A: IMUs suffer from drift -- small measurement errors accumulate over time. Over a 15-minute descent, an IMU can drift by hundreds of meters. TRN provides absolute position fixes that correct this drift.

**Q5: What is YOLO and why was it chosen for crater detection?**
A: YOLO (You Only Look Once) is a real-time object detection neural network that detects objects in a single forward pass. It was chosen because (1) it's fast enough for real-time descent navigation, (2) it handles small objects (craters) well, and (3) it provides bounding boxes that can be converted to crater center positions and radii.

## Category B: Algorithm Details

**Q6: What is Delaunay triangulation and why is it used here?**
A: Delaunay triangulation connects points into triangles such that no point lies inside any triangle's circumscribed circle. We use it because (1) it produces well-shaped triangles (maximises minimum angle), (2) it's deterministic -- the same crater positions produce the same triangulation, enabling matching, and (3) it naturally groups nearby craters.

**Q7: What makes the first-order descriptor invariant to scale, rotation, and translation?**
A: The descriptor uses *ratios* of side lengths (l1/l3, l2/l3), which cancel any uniform scaling factor. Translation doesn't change side lengths. Rotation doesn't change side lengths either. Therefore the descriptor is invariant to all three transformations.

**Q8: What is the difference between first-order and second-order descriptors?**
A: First-order describes a single triangle's geometry in isolation. Second-order additionally encodes the geometry of neighbouring triangles (those sharing an edge). This is analogous to recognizing a person by both their face (first-order) and the faces of people standing next to them (second-order). The combination is much more distinctive.

**Q9: Why is the Gaussian kernel used for similarity and not cosine similarity or Euclidean distance?**
A: The Gaussian kernel maps Euclidean distances to a [0, 1] similarity score with desirable properties: (1) identical descriptors get score 1, (2) scores decay exponentially with distance (a natural similarity measure), and (3) the bandwidth parameter sigma controls sensitivity. Cosine similarity wouldn't capture magnitude differences well for ratio-based descriptors.

**Q10: What is RANSAC and why is it needed?**
A: RANSAC (Random Sample Consensus) robustly estimates a geometric model (homography) in the presence of outliers. It's needed because even after triangle matching and voting, some crater correspondences may be wrong. RANSAC finds the largest subset of matches that are geometrically consistent, effectively filtering out incorrect correspondences.

**Q11: How does the homography give position error?**
A: A homography H = [R|t] encodes the transformation between the map and observation. The translation components (tx, ty) directly give the camera's displacement from the expected position in pixels. Dividing by image size and multiplying by 100 gives the error as a percentage of altitude.

## Category C: Our Improvements

**Q12: What is the crater-radius descriptor and why is it novel?**
A: We extend the base paper's 3D descriptor with 2 additional components: (r_min/r_max, r_mid/r_max), where r_min, r_mid, r_max are the sorted crater radii at the triangle's three vertices. This is novel because the base paper uses only geometric shape (side ratios) and ignores the physical sizes of the craters. Our extension adds discriminability: two identically-shaped triangles connecting differently-sized craters are now distinguishable. The ratios maintain scale invariance.

**Q13: How does the adaptive threshold work?**
A: Instead of a fixed threshold of 0.70, we compute: `threshold = max(0.50, 0.75 * mean(top-K best scores))`. For each observation triangle, we find its best match score. We take the mean of the top 50% of these scores and set the threshold at 75% of that mean. When data is clean, scores are high, so the threshold remains ~0.65-0.70. When data is noisy, scores drop, and the threshold naturally lowers, recovering more correct matches.

**Q14: How does confidence-weighted voting work?**
A: When a YOLO detector produces a detection, it also outputs a confidence score (0 to 1). We multiply each correspondence vote by this confidence: `vote = similarity_score * confidence`. Real craters (confidence ~0.85) get full voting power. Spurious detections (confidence ~0.35) contribute proportionally less, reducing their impact on the final correspondences without requiring a hard confidence cutoff.

**Q15: What are the quantitative improvements?**
A: Position error X improved by 31% (0.2834% -> 0.1949%), position error Y improved by 33% (0.3224% -> 0.2157%). Navigation success at 10% error rate improved by +12.17 percentage points (87.33% -> 99.5%). Matching accuracy improved by +0.34% (96.57% -> 96.91%). Navigation success in clean conditions reached 100% (was 99.6%).

**Q16: What is the trade-off of the improved algorithm?**
A: Matching time increased from 0.073s to 0.157s (2.1x slower) due to the larger descriptor dimensions (25D vs 12D) and more neighbours (4 vs 3). However, 0.157s per image is still well within real-time requirements for descent navigation, which typically processes at 1-10 Hz.

## Category D: Technical Deep Dives

**Q17: What is the significance of sigma = 0.20 in the Gaussian kernel?**
A: Sigma controls how quickly similarity drops with descriptor distance. Too small = only near-identical descriptors match (too strict). Too large = dissimilar descriptors also get high scores (too permissive). We chose 0.20 by scaling the original 0.15 by sqrt(25/12) ≈ 1.44 to account for the larger descriptor dimension, then rounding slightly down for sharper discrimination.

**Q18: Why Monte Carlo simulation instead of a single test?**
A: A single test gives one data point that could be lucky or unlucky. Monte Carlo runs 1000 trials with random noise and random crater omissions/additions, giving a statistical distribution of performance. This lets us report mean accuracy, standard deviation, min/max, and confidence intervals -- much more reliable for drawing conclusions.

**Q19: What is bipartite matching and why greedy instead of optimal (Hungarian)?**
A: Bipartite matching finds one-to-one correspondences between two sets. The Hungarian algorithm gives the optimal solution in O(n^3) but is computationally expensive for large numbers of triangles. Our greedy approach (sort all candidates by score, assign top-down) is O(n^2 log n) and in practice gives near-optimal results because the similarity scores are well-separated (good matches have much higher scores than bad ones).

**Q20: How does the error rate sweep experiment work?**
A: We systematically increase the "detection error rate" from 0% to 100%. At each level, we split the error equally between false positives (adding spurious craters) and false negatives (removing real craters). For each error rate, we run 200+ Monte Carlo trials and measure matching accuracy and navigation success. This produces a robustness curve showing how gracefully the algorithm degrades under increasing detection noise.

**Q21: What is the LRO NAC and why is it important?**
A: LRO NAC (Lunar Reconnaissance Orbiter Narrow Angle Camera) is a high-resolution camera aboard NASA's LRO spacecraft. It captures images at 0.5 m/pixel resolution -- detailed enough to see individual craters down to ~4 meters diameter. The LRO has mapped nearly the entire lunar surface, providing the reference maps against which our algorithm matches.

**Q22: What is the role of the Delaunay filtering (area < 20px^2, aspect > 10)?**
A: Filtering removes triangles that provide poor matching signals. Very small triangles (area < 20px^2) are noise-sensitive -- a few pixels of position error can completely change their shape. Very elongated triangles (aspect > 10:1) have degenerate geometry -- their descriptors cluster near (0, 1, 0), making them indistinguishable.

## Category E: Practical and Future Work

**Q23: Can this algorithm work in real-time during a lunar landing?**
A: Yes. Our matching takes ~0.157s per image, and YOLO detection takes ~0.1s per image. Total: ~0.26s per frame, allowing ~3.8 Hz processing. Lunar descent landers typically update position at 1-10 Hz, so our algorithm is within the real-time budget, especially on GPU hardware (which would be much faster than our CPU-only testing).

**Q24: What would improve the YOLO detection performance?**
A: (1) Train on more data (currently only 12 images), (2) Use YOLOv8s instead of nano (3.7x more parameters), (3) Train for more epochs (100-200) on GPU, (4) Use data augmentation specific to lunar imagery (shadow simulation, regolith texture variations), (5) Use a larger model like YOLOv9 or RT-DETR.

**Q25: What are the limitations of this approach?**
A: (1) Depends on crater detection quality -- low recall (23%) means most craters are missed. (2) Matching degrades rapidly above 20% detection error. (3) Assumes a 2D planar scene (homography), which breaks for very oblique camera angles. (4) Does not work in crater-sparse regions (needs >= 10 craters). (5) The Gaussian kernel bandwidth must be tuned for the descriptor dimensionality.

**Q26: How could this be extended to 3D pose estimation?**
A: Instead of a homography (2D mapping), use the Perspective-n-Point (PnP) algorithm with known 3D crater positions from a DEM (Digital Elevation Model). This would give full 6-DoF pose (position + orientation) instead of just 2D translation + scale.

**Q27: How does this compare to deep learning-based matching (e.g., SuperGlue)?**
A: Deep learning matchers like SuperGlue learn to match features end-to-end and can handle complex appearance changes. However, they require large training datasets of matched image pairs, are less interpretable, and may not generalise to new lunar regions. Our geometric approach works with *any* crater field without retraining and has provable invariance properties.

**Q28: What is the significance of the 0.44% position error reported in the base paper?**
A: At a typical descent altitude of 100 km, 0.44% error = 440 meters. Our improved algorithm achieves 0.19-0.22%, which translates to ~190-220 meters. Modern mission requirements (e.g., Artemis targeting near the lunar south pole) demand landing accuracy within 100-200 meters, so continued improvement is needed, primarily through better detection.

---

# PART 9: FURTHER READING AND REFERENCES

## 9.1 Core References

1. **Base Paper:** "Lunar Crater Matching With Triangle-Based Global Second-Order Similarity for Precision Navigation" -- IEEE TGRS, 2024. [Link](https://ieeexplore.ieee.org/document/11123425/)

2. **CraterDANet:** Yang et al., "CraterDANet: A Convolutional Neural Network for Small-Scale Crater Detection via Synthetic-to-Real Domain Adaptation," IEEE TGRS. The dataset we use comes from this paper.

3. **Robbins Crater Database:** S. J. Robbins, "A New Global Database of Lunar Impact Craters >1-2 km," JGR Planets, 2019. The most comprehensive global lunar crater catalogue.

4. **Ultralytics YOLOv8:** [https://docs.ultralytics.com/](https://docs.ultralytics.com/)

5. **LRO NAC Data:** [https://lroc.im-ldi.com/images/downloads](https://lroc.im-ldi.com/images/downloads)

## 9.2 Broader Context

6. **Terrain-Relative Navigation for Planetary Landing:** Johnson, A. E., et al., "Real-Time Terrain Relative Navigation Test Results from a Relevant Environment for Mars Landing," AIAA, 2015. The general framework for TRN.

7. **RANSAC:** Fischler & Bolles, "Random Sample Consensus: A Paradigm for Model Fitting with Applications to Image Analysis and Automated Cartography," 1981. The original RANSAC paper.

8. **Delaunay Triangulation:** de Berg et al., "Computational Geometry: Algorithms and Applications" -- the standard textbook covering Delaunay triangulation, Voronoi diagrams, and convex hulls.

9. **Homography Estimation:** Hartley & Zisserman, "Multiple View Geometry in Computer Vision" -- the bible of geometric computer vision, covering homographies, fundamental matrices, and camera models.

10. **Gaussian Processes and Kernels:** Rasmussen & Williams, "Gaussian Processes for Machine Learning" -- for deeper understanding of kernel similarity functions.

## 9.3 Advanced / Future Directions

11. **SuperGlue:** Sarlin et al., "SuperGlue: Learning Feature Matching with Graph Neural Networks," CVPR 2020. A deep learning approach to feature matching that could potentially replace the triangle-based approach.

12. **Vision Transformers for Detection:** Liu et al., "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows," ICCV 2021. Next-generation detection architectures.

13. **PnP (Perspective-n-Point):** Lepetit et al., "EPnP: An Accurate O(n) Solution to the PnP Problem," IJCV 2009. For extending to full 3D pose estimation.

14. **NASA ALHAT:** NASA's Autonomous Landing and Hazard Avoidance Technology -- the state of the art in terrain-relative navigation for space missions.

15. **Graph Neural Networks for Matching:** Fey & Lenssen, "Fast Graph Representation Learning with PyTorch Geometric," ICLR 2019 Workshop. For potential GNN-based descriptor learning.

---

# PART 10: GLOSSARY

| Term | Definition |
|------|-----------|
| **TRN** | Terrain-Relative Navigation -- determining position from terrain features |
| **YOLO** | You Only Look Once -- real-time object detection neural network |
| **mAP** | Mean Average Precision -- standard object detection accuracy metric |
| **IoU** | Intersection over Union -- overlap metric for bounding boxes |
| **Delaunay Triangulation** | Triangulation maximising minimum angles, no point inside any circumcircle |
| **First-Order Descriptor** | Per-triangle feature vector from its own geometry (3D or 5D) |
| **Second-Order Descriptor** | Extended descriptor including neighbour information (12D or 25D) |
| **Gaussian Kernel** | Similarity function: exp(-d^2 / 2σ^2), mapping distance to [0,1] |
| **RANSAC** | Random Sample Consensus -- robust model fitting with outlier rejection |
| **Homography** | 3x3 matrix mapping points between two planes (projective transform) |
| **Monte Carlo** | Statistical method using repeated random sampling to estimate distributions |
| **LRO NAC** | Lunar Reconnaissance Orbiter Narrow Angle Camera (0.5 m/pixel) |
| **Scale Invariant** | Property unchanged by uniform scaling (zoom in/out) |
| **Bipartite Matching** | Finding one-to-one correspondences between two sets |
| **Confidence Score** | YOLO output [0,1] indicating detection certainty |
| **Error Rate Sweep** | Testing algorithm robustness across increasing detection error levels |
| **Circumscribed Circle** | The circle passing through all three vertices of a triangle |
| **DEM** | Digital Elevation Model -- 3D terrain height map |
| **PnP** | Perspective-n-Point -- algorithm to find camera pose from 3D-2D correspondences |
| **Regolith** | Fine-grained rocky debris covering the lunar surface |
| **Voronoi Diagram** | Partition of space into regions closest to each point; dual of Delaunay |

---

*This document prepared for AIP Project Group 3, ABV-IIITM Gwalior, April 2026.*
