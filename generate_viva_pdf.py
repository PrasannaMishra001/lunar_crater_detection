"""
Generate a comprehensive viva preparation PDF for the Lunar Crater Detection project.
Uses xhtml2pdf to convert HTML -> PDF.
"""
from xhtml2pdf import pisa
import os

OUTPUT_PDF = os.path.join(os.path.dirname(__file__), "AIP_Viva_Preparation_Guide.pdf")

HTML = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<style>
  @page {
    size: A4;
    margin: 18mm 16mm 18mm 16mm;
  }
  body { font-family: Arial, sans-serif; font-size: 10pt; color: #1a1a1a; line-height: 1.55; }
  h1   { font-size: 17pt; color: #1a3a6b; border-bottom: 2.5px solid #1a3a6b; padding-bottom: 4px; margin-top: 0; page-break-before: always; }
  h2   { font-size: 13pt; color: #1a3a6b; margin-top: 16px; margin-bottom: 4px; }
  h3   { font-size: 11pt; color: #2e5ca8; margin-top: 12px; margin-bottom: 3px; }
  h4   { font-size: 10pt; color: #3d6fbd; margin-top: 10px; margin-bottom: 2px; }
  .cover { text-align: center; margin-top: 80px; page-break-after: always; }
  .cover h1 { border-bottom: none; font-size: 22pt; }
  .cover .subtitle { font-size: 13pt; color: #444; margin-top: 10px; }
  .cover .meta { font-size: 10pt; color: #666; margin-top: 30px; line-height: 2; }
  .warning-box { background: #fff3cd; border: 2px solid #e6a817; border-radius: 4px; padding: 10px 14px; margin: 10px 0; }
  .warning-box h3 { color: #7d5800; margin-top: 0; }
  .info-box { background: #e8f4fd; border-left: 4px solid #2e5ca8; padding: 8px 12px; margin: 8px 0; }
  .formula-box { background: #f0f0f0; border: 1px solid #ccc; border-radius: 3px; padding: 8px 12px; font-family: Courier, monospace; font-size: 9.5pt; margin: 6px 0; }
  table { width: 100%; border-collapse: collapse; margin: 8px 0; font-size: 9pt; }
  th    { background: #1a3a6b; color: #fff; padding: 6px 8px; text-align: left; }
  td    { padding: 5px 8px; border-bottom: 1px solid #ddd; vertical-align: top; }
  tr:nth-child(even) td { background: #f5f7fa; }
  .q    { background: #eaf0fb; border-left: 4px solid #2e5ca8; padding: 6px 10px; margin: 10px 0 2px 0; font-weight: bold; font-size: 10pt; }
  .a    { padding: 4px 10px 10px 14px; border-left: 4px solid #a0b8e0; margin-bottom: 6px; }
  .tag-new { background: #d4edda; color: #155724; border-radius: 3px; padding: 1px 5px; font-size: 8.5pt; font-weight: bold; }
  .tag-critical { background: #f8d7da; color: #721c24; border-radius: 3px; padding: 1px 5px; font-size: 8.5pt; font-weight: bold; }
  .tag-term { background: #fff3cd; color: #856404; border-radius: 3px; padding: 1px 5px; font-size: 8.5pt; font-weight: bold; }
  ul li { margin-bottom: 3px; }
  .toc-item { margin: 3px 0; }
  .page-first h1 { page-break-before: avoid; }
  .metric-good { color: #155724; font-weight: bold; }
  .metric-bad  { color: #721c24; font-weight: bold; }
  .metric-neutral { color: #856404; font-weight: bold; }
  hr { border: none; border-top: 1px solid #ccc; margin: 10px 0; }
</style>
</head>
<body>

<!-- COVER PAGE -->
<div class="cover">
  <h1>Lunar Crater Matching<br/>Viva Preparation Guide</h1>
  <div class="subtitle">Triangle-Based Global Second-Order Similarity for Precision Navigation</div>
  <hr/>
  <div class="meta">
    <b>Course:</b> Applied Image Processing (AIP) &nbsp;&nbsp;|&nbsp;&nbsp; <b>Group:</b> 03<br/>
    <b>Supervisor:</b> Dr. Mahua Bhattacharya, ABV-IIITM Gwalior<br/><br/>
    <b>Members:</b><br/>
    Ankit Baidsen (2023IMT-014) &nbsp;|&nbsp; Malladi Nagarjuna (2023IMT-050)<br/>
    Prasanna Mishra (2023IMT-059) &nbsp;|&nbsp; Prasun Baranwal (2023IMT-060)<br/>
    Shivam Deolankar (2023IMT-073)<br/><br/>
    <b>Date:</b> April 2026
  </div>
  <div style="margin-top:50px; font-size:9pt; color:#888;">
    This document covers: PPT error corrections &bull; all algorithm terms explained &bull; all metrics with formulas &bull;
    detailed graph/result interpretation &bull; 28 cross-questions with full answers
  </div>
</div>

<!-- ================================================================ -->
<h1 class="page-first">PART 0 &mdash; CRITICAL PPT ERRORS TO FIX</h1>
<!-- ================================================================ -->

<p>These errors will cause immediate problems in the viva. Fix them before presenting.</p>

<div class="warning-box">
  <h3>&#9888; Error 1 &mdash; Slide 4 (Objectives): COMPLETELY WRONG CONTENT</h3>
  <p>The Objectives slide shows <b>DistilBERT, RoBERTa, MUDI, drug recommendations, 300k reviews</b>. This is from a
  completely different project. The slide content is copied from a medicine recommendation system template.
  Replace with the correct objectives below:</p>
  <ul>
    <li><b>Primary:</b> Implement and evaluate the triangle-based crater matching algorithm from the base IEEE paper</li>
    <li><b>Secondary:</b> Evaluate on LRO NAC dataset using Monte Carlo simulation under realistic detection noise</li>
    <li><b>Technical:</b> Introduce 5 novel improvements (5D descriptor, adaptive threshold, confidence voting,
    YOLOv8s upgrade, extended adjacency) that reduce position error by &gt;30%</li>
  </ul>
</div>

<div class="warning-box">
  <h3>&#9888; Error 2 &mdash; Slide 11 (Results Table): ALL NUMBERS HAVE WRONG DECIMAL POINTS</h3>
  <p>Every percentage value has decimal points in the wrong place, making the results look absurd.
  Correct values:</p>
  <table>
    <tr><th>Column</th><th>PPT Shows (WRONG)</th><th>Correct Value</th></tr>
    <tr><td>Baseline Matching Accuracy</td><td class="metric-bad">9.657%</td><td class="metric-good">96.57%</td></tr>
    <tr><td>Improved Matching Accuracy</td><td class="metric-bad">9.691%</td><td class="metric-good">96.91%</td></tr>
    <tr><td>Baseline Navigation Success</td><td class="metric-bad">9.960%</td><td class="metric-good">99.60%</td></tr>
    <tr><td>Improved Navigation Success</td><td class="metric-bad">1.000%</td><td class="metric-good">100.0%</td></tr>
    <tr><td>Base Paper Position Error X</td><td class="metric-bad">44%</td><td class="metric-good">0.44%</td></tr>
    <tr><td>Baseline Position Error X</td><td class="metric-bad">2.834%</td><td class="metric-good">0.2834%</td></tr>
    <tr><td>Improved Position Error X</td><td class="metric-bad">1.949%</td><td class="metric-good">0.1949%</td></tr>
    <tr><td>Base Paper Position Error Y</td><td class="metric-bad">44%</td><td class="metric-good">0.44%</td></tr>
    <tr><td>Baseline Position Error Y</td><td class="metric-bad">3.224%</td><td class="metric-good">0.3224%</td></tr>
    <tr><td>Improved Position Error Y</td><td class="metric-bad">2.157%</td><td class="metric-good">0.2157%</td></tr>
    <tr><td>Change in Position Error X</td><td class="metric-bad">&minus;312%</td><td class="metric-good">&minus;31.2%</td></tr>
    <tr><td>Change in Position Error Y</td><td class="metric-bad">&minus;331%</td><td class="metric-good">&minus;33.1%</td></tr>
    <tr><td>Change in Matching Accuracy</td><td class="metric-bad">34%</td><td class="metric-good">+0.34%</td></tr>
    <tr><td>Change in Navigation Success</td><td class="metric-bad">4%</td><td class="metric-good">+0.4%</td></tr>
    <tr><td>Change in Reprojection Avg</td><td class="metric-bad">67%</td><td class="metric-good">+6.7%</td></tr>
    <tr><td>Change in Reprojection RMS</td><td class="metric-bad">54%</td><td class="metric-good">+5.4%</td></tr>
  </table>
</div>

<div class="warning-box">
  <h3>&#9888; Error 3 &mdash; Slide 16 (Performance Metrics): &ldquo;IDENTITY THEFT&rdquo; heading</h3>
  <p>The third box is titled <b>&ldquo;IDENTITY THEFT&rdquo;</b> &mdash; this is a placeholder from the slide template.
  Rename it to <b>&ldquo;Monte Carlo Simulation&rdquo;</b> or merge those bullets into the Matching Level box.</p>
</div>


<!-- ================================================================ -->
<h1>PART 1 &mdash; ALL TERMS EXPLAINED IN DETAIL</h1>
<!-- ================================================================ -->

<h2>1.1 Navigation and Space Terms</h2>

<table>
  <tr><th>Term</th><th>Full Form / Definition</th><th>Relevance to Project</th></tr>
  <tr>
    <td><b>TRN</b></td>
    <td>Terrain-Relative Navigation. The spacecraft determines its position by comparing its camera image to a
    pre-stored map, using recognizable surface landmarks.</td>
    <td>Our entire system is a TRN implementation &mdash; craters are the landmarks, the Robbins database is the map.</td>
  </tr>
  <tr>
    <td><b>LRO</b></td>
    <td>Lunar Reconnaissance Orbiter. NASA satellite launched June 2009. Still operational. Carries the NAC camera
    that has mapped the entire lunar surface at 0.5 m/pixel resolution.</td>
    <td>Our training and test images are LRO NAC captures. The camera provides the reference map imagery.</td>
  </tr>
  <tr>
    <td><b>NAC</b></td>
    <td>Narrow Angle Camera on LRO. Two cameras (NAC-L and NAC-R) each capturing 5 km wide swaths.
    Resolution: 0.5 m/pixel from 50 km orbit. Used to create the highest-resolution systematic lunar map.</td>
    <td>Our dataset uses 800&times;800 mosaics and 1000&times;1000 CDR tiles created from NAC imagery.</td>
  </tr>
  <tr>
    <td><b>CDR</b></td>
    <td>Calibrated Data Record. A standardized scientific data product from LRO with radiometric corrections applied.
    More processed than raw EDR (Experiment Data Record).</td>
    <td>Our 8 test images are 1000&times;1000 px NAC CDR tiles.</td>
  </tr>
  <tr>
    <td><b>Chang&rsquo;E-4</b></td>
    <td>China&rsquo;s fourth lunar exploration mission (CNSA, January 2019). First spacecraft to land on the Moon&rsquo;s
    far side, in the Von K&aacute;rm&aacute;n crater (45&ndash;46&deg;S, 176.4&ndash;178.8&deg;E). Carried Yutu-2 rover.</td>
    <td>Our dataset covers the Chang&rsquo;E-4 landing site, chosen because dense crater annotations were available
    for that region.</td>
  </tr>
  <tr>
    <td><b>Beresheet</b></td>
    <td>Israeli lunar lander (SpaceIL, April 2019). Failed during descent due to inertial measurement unit
    error causing the main engine to cut off, resulting in crash landing.</td>
    <td>Cited in problem statement as example of navigation failure during lunar descent.</td>
  </tr>
  <tr>
    <td><b>Chandrayaan-2 Vikram</b></td>
    <td>ISRO lander (September 2019). Lost communication at ~400 m altitude during &ldquo;fine braking&rdquo; phase.
    Post-analysis showed velocity reduction was too aggressive, causing tilt beyond correctable range.</td>
    <td>Cited as motivation: robust TRN systems could have detected and corrected the trajectory deviation.</td>
  </tr>
  <tr>
    <td><b>CraterDANet</b></td>
    <td>Crater Detection Adversarial Network. The paper by Yang et al. (secondary reference) that created our
    dataset using synthetic-to-real domain adaptation with a GAN to augment training data.</td>
    <td>We use the publicly released LRO NAC dataset from this paper as our training and test set.</td>
  </tr>
  <tr>
    <td><b>Robbins Database</b></td>
    <td>A global lunar crater catalog by Stuart Robbins (JGR Planets, 2019) containing 1.3 million craters
    &gt;1&ndash;2 km diameter derived from LRO data. The most complete such catalog available.</td>
    <td>In a real navigation system, this would be the reference map. We simulate the map from our GT annotations.</td>
  </tr>
</table>

<h2>1.2 Computer Vision and Detection Terms</h2>

<table>
  <tr><th>Term</th><th>Definition</th><th>Relevance</th></tr>
  <tr>
    <td><b>YOLO</b></td>
    <td>You Only Look Once. A single-stage real-time object detector. Divides image into S&times;S grid;
    each cell predicts B bounding boxes with objectness score and class probabilities simultaneously in
    one neural network forward pass.</td>
    <td>We use YOLO to detect craters as bounding boxes from LRO images.</td>
  </tr>
  <tr>
    <td><b>YOLOv7 vs v8</b></td>
    <td>YOLOv7 (Wang et al., 2022): anchor-based, ~37M parameters, E-ELAN backbone.
    YOLOv8 (Ultralytics, 2023): anchor-free, cleaner API, C2f backbone, built-in pose/seg heads.
    v8n = nano (3.2M params), v8s = small (11.2M params).</td>
    <td>Base paper used v7. We upgraded to v8n baseline and v8s in improved pipeline.</td>
  </tr>
  <tr>
    <td><b>Anchor-free detection</b></td>
    <td>Object detectors traditionally pre-define &ldquo;anchor boxes&rdquo; of fixed sizes/ratios and predict
    offsets from those anchors. Anchor-free detectors (YOLOv8) directly predict the box center and size
    from each feature map location without pre-defined templates.</td>
    <td>Advantage: no need to tune anchor sizes for crater bounding boxes which vary from 8px to 100s of px.</td>
  </tr>
  <tr>
    <td><b>mAP</b></td>
    <td>Mean Average Precision. Average of per-class Average Precision (area under Precision-Recall curve).
    For single-class detection (craters), mAP = AP.</td>
    <td>Primary detection quality metric. Our mAP@50 = 0.416.</td>
  </tr>
  <tr>
    <td><b>IoU</b></td>
    <td>Intersection over Union. For predicted box P and ground-truth box G:
    IoU = area(P&cap;G) / area(P&cup;G). Ranges 0 (no overlap) to 1 (perfect overlap).</td>
    <td>Used to decide if a detection is a true positive (IoU &ge; threshold).</td>
  </tr>
  <tr>
    <td><b>Precision</b></td>
    <td>TP / (TP + FP). Of all craters we detected, what fraction are real craters.
    High precision = few false alarms. Our value: 0.617 (61.7% of detections are real craters).</td>
    <td>Low precision means spurious detections &mdash; handled by our confidence-weighted voting.</td>
  </tr>
  <tr>
    <td><b>Recall</b></td>
    <td>TP / (TP + FN). Of all real craters, what fraction did we detect.
    High recall = few missed craters. Our value: 0.232 (23.2% of craters detected).</td>
    <td>Low recall is acceptable: we only need ~15&ndash;30 craters per scene for reliable triangulation.</td>
  </tr>
  <tr>
    <td><b>C2f block</b></td>
    <td>Cross Stage Partial with 2 sub-bottlenecks. YOLOv8&rsquo;s improvement over YOLOv5/7&rsquo;s C3 block.
    Better gradient flow, more feature reuse, fewer parameters for similar accuracy.</td>
    <td>Reason why YOLOv8 is more efficient than YOLOv7 on small datasets.</td>
  </tr>
  <tr>
    <td><b>Inference speed</b></td>
    <td>Time to process one image after model is loaded. Our CPU speed: 101.7 ms/image.
    GPU typical: 5&ndash;15 ms/image for YOLOv8n.</td>
    <td>At 0.1 s/detection + 0.157 s/matching = ~0.26 s/cycle = ~4 Hz update rate.</td>
  </tr>
</table>

<h2>1.3 Triangulation and Descriptor Terms</h2>

<table>
  <tr><th>Term</th><th>Definition</th><th>Relevance</th></tr>
  <tr>
    <td><b>Delaunay Triangulation</b></td>
    <td>A triangulation of a point set such that no point lies inside the circumcircle of any triangle.
    Maximizes the minimum angle (avoids thin slivers). Dual graph of the Voronoi diagram.
    For N points, produces ~2N triangles, ~3N edges.</td>
    <td>Creates the triangle mesh from crater centers. Guarantees compact, non-degenerate triangles.
    Implemented via scipy.spatial.Delaunay.</td>
  </tr>
  <tr>
    <td><b>Circumcircle</b></td>
    <td>The unique circle passing through all three vertices of a triangle. Delaunay condition:
    no other point may lie inside any triangle&rsquo;s circumcircle.</td>
    <td>Mathematical basis for Delaunay&rsquo;s optimality property.</td>
  </tr>
  <tr>
    <td><b>Degenerate triangle</b></td>
    <td>A triangle that is nearly collinear (all three points nearly on a line) or too small.
    Has very small area and/or extreme side-length ratios. Causes unstable descriptors
    because tiny positional noise leads to large ratio changes.</td>
    <td>Filtered out before building descriptors. Improves matching reliability.</td>
  </tr>
  <tr>
    <td><b>Scale invariance</b></td>
    <td>A property of a descriptor: multiplying all coordinates by a constant k does not change the descriptor.
    Achieved by using ratios (dimensionless quantities) instead of absolute distances.</td>
    <td>Essential for matching craters from different altitudes (different image scales).</td>
  </tr>
  <tr>
    <td><b>Rotation invariance</b></td>
    <td>A descriptor that is the same regardless of the rotation of the input.
    Achieved by sorting side lengths before computing ratios (canonical ordering).</td>
    <td>Essential for matching: the spacecraft can be at any heading when it images a crater field.</td>
  </tr>
  <tr>
    <td><b>Translation invariance</b></td>
    <td>A descriptor unchanged by adding a constant to all coordinates.
    Ratios of distances are translation-invariant because subtracting any offset doesn&rsquo;t change relative distances.</td>
    <td>Trivially satisfied by using relative geometry (side lengths, areas) rather than absolute positions.</td>
  </tr>
  <tr>
    <td><b>First-order descriptor (3D)</b></td>
    <td>Three-dimensional feature vector for each triangle:
    [l1/l3, l2/l3, area/(perimeter&sup2;)] where l1 &le; l2 &le; l3 are sorted side lengths.
    Captures triangle shape, not size. Base paper used this.</td>
    <td>Our baseline used this. Our improved version extends it to 5D.</td>
  </tr>
  <tr>
    <td><b>First-order descriptor (5D) [NEW]</b></td>
    <td>Five-dimensional vector: [l1/l3, l2/l3, area/(perimeter&sup2;), r_small/r_large, r_mid/r_large]
    where r values are sorted crater radii. The two new dimensions encode the physical size distribution
    of the craters at each triangle vertex.</td>
    <td>Our improvement. Breaks degeneracy when two triangles have identical geometry but different crater sizes.</td>
  </tr>
  <tr>
    <td><b>Second-order descriptor</b></td>
    <td>Concatenation of a triangle&rsquo;s own first-order descriptor with the sorted first-order descriptors
    of its neighbouring triangles (those sharing an edge). Dimension = D1 &times; (1 + K) where D1 is the
    first-order dimension and K is the number of neighbours. Our improved: 5 &times; (1+4) = 25D.</td>
    <td>The core innovation of the base paper. Captures spatial context, not just local geometry.</td>
  </tr>
  <tr>
    <td><b>Adjacency in triangulation</b></td>
    <td>Two triangles are adjacent if they share exactly one edge (two vertices). In Delaunay triangulation,
    each triangle has at most 3 adjacent triangles (one per edge). We extend to the next ring for K=4.</td>
    <td>Defines which triangles contribute to each second-order descriptor. K=4 vs K=3 adds one more contextual triangle.</td>
  </tr>
</table>

<h2>1.4 Matching and Similarity Terms</h2>

<table>
  <tr><th>Term</th><th>Definition</th><th>Relevance</th></tr>
  <tr>
    <td><b>Gaussian Kernel</b></td>
    <td>S(i,j) = exp(&minus;||d_i &minus; d_j||&sup2; / (2&sigma;&sup2;)).
    Maps L2 distance between descriptor vectors to similarity score in [0,1].
    &sigma; is the bandwidth: small &sigma; = strict (only very similar descriptors get high scores),
    large &sigma; = lenient. Our &sigma; = 0.20.</td>
    <td>Used to build the similarity matrix between all observation-map triangle pairs.</td>
  </tr>
  <tr>
    <td><b>Sigma (&sigma;) selection</b></td>
    <td>We compute: &sigma;_new = &sigma;_old &times; &radic;(D_new / D_old) = 0.15 &times; &radic;(25/12) &asymp; 0.217 &rarr; 0.20.
    The scaling accounts for the fact that L2 distance in higher dimensions grows as &radic;D for random unit vectors.
    Without rescaling, the Gaussian would be too tight for the 25D space.</td>
    <td>Critical for correct similarity computation after extending from 12D to 25D descriptors.</td>
  </tr>
  <tr>
    <td><b>Similarity matrix</b></td>
    <td>An M&times;N matrix where M = number of observation triangles, N = number of map triangles.
    Entry S[i,j] = Gaussian kernel similarity between observation triangle i and map triangle j.
    Computed in chunks to avoid memory overflow for large crater sets.</td>
    <td>Foundation of the matching process. Every possible pair is scored before greedy selection.</td>
  </tr>
  <tr>
    <td><b>Adaptive threshold [NEW]</b></td>
    <td>Threshold computed from the data: effective_thresh = clip(0.75 &times; mean(top-K row-maxima), 0.50, 0.82).
    K = max(5, min(20, N/2)). Row-maximum = best score for each observation triangle.
    The threshold tracks scene quality: high scores &rarr; high threshold; low scores &rarr; lower threshold.</td>
    <td>Replaces the fixed threshold (0.55). Prevents missing valid matches in noisy scenes and accepting
    false matches in clean scenes.</td>
  </tr>
  <tr>
    <td><b>Greedy bipartite matching</b></td>
    <td>Enforces one-to-one correspondence between observation and map triangles.
    All (i,j) pairs sorted by S[i,j] descending. Accept each pair if neither i nor j already matched.
    O(MN log(MN)) time. Suboptimal vs Hungarian algorithm (O(N&sup3;)) but practically equivalent for
    high-quality similarity matrices.</td>
    <td>Ensures each map crater is matched to at most one observed crater.</td>
  </tr>
  <tr>
    <td><b>Voting / crater correspondence</b></td>
    <td>Each matched triangle pair (i,j) &ldquo;votes&rdquo; for each of the 3 vertex correspondences it implies.
    After all triangle matches, crater correspondences with &ge; min_vote votes are accepted as verified pairs.
    Multiple triangles must agree on a crater pair for it to be accepted.</td>
    <td>Converts triangle-level matches to crater-level matches needed for homography estimation.</td>
  </tr>
  <tr>
    <td><b>Confidence-weighted voting [NEW]</b></td>
    <td>Vote weight = &radic;(conf_A &times; conf_B &times; conf_C) for a triangle with crater confidences A,B,C
    (geometric mean). True craters have YOLO confidence ~0.85; spurious: ~0.35.
    Vote from a triangle with 3 confident craters (~0.85) is ~2.4&times; more than from a spurious triangle (~0.35).
    Naturally downweights false detections without explicit filtering.</td>
    <td>One of our 5 improvements. Particularly effective at high error rates.</td>
  </tr>
  <tr>
    <td><b>RANSAC</b></td>
    <td>Random Sample Consensus (Fischler &amp; Bolles, 1981). Robust estimation algorithm:
    1. Randomly sample minimal subset (4 point pairs for homography).
    2. Fit model (estimate H).
    3. Count inliers (pairs with reprojection error &lt; threshold).
    4. Repeat N times. Best model = most inliers.
    Robust to ~50% outlier rate.</td>
    <td>Final geometric verification step that removes geometrically inconsistent crater correspondences
    before pose estimation.</td>
  </tr>
  <tr>
    <td><b>Homography</b></td>
    <td>A 3&times;3 projective transformation matrix H such that p&rsquo; = H&middot;p (in homogeneous coordinates).
    Maps points from one image plane to another. For planar scenes (lunar surface viewed from above),
    a homography exactly models the camera motion. Has 8 degrees of freedom (9 elements, 1 scale).</td>
    <td>Estimated from 4+ crater correspondences. Encodes spacecraft position (tx, ty) and altitude (scale).</td>
  </tr>
</table>

<h2>1.5 Evaluation Terms</h2>

<table>
  <tr><th>Term</th><th>Definition</th><th>Relevance</th></tr>
  <tr>
    <td><b>Monte Carlo simulation</b></td>
    <td>Statistical technique: run many random trials to estimate a distribution.
    Each trial: randomly perturb inputs (add Gaussian noise to crater positions, randomly remove/add craters),
    run the algorithm, record performance. After N trials, compute mean, std, percentiles.</td>
    <td>We run N=100&ndash;1000 trials with &sigma;=5px position noise. Gives statistically robust performance estimates.</td>
  </tr>
  <tr>
    <td><b>Position error (% altitude)</b></td>
    <td>After estimating homography H, the predicted spacecraft position (projected onto lunar surface)
    is compared to the true position. Error is expressed as percentage of flight altitude Z.
    If altitude = 1000m and position error = 2m, error = 0.2% altitude.</td>
    <td>Primary navigation metric from the base paper. Our improved: X=0.195%, Y=0.216% (vs baseline 0.283%, 0.322%).</td>
  </tr>
  <tr>
    <td><b>Reprojection error</b></td>
    <td>For a verified correspondence (p_obs, p_map): reprojection error = ||p_obs &minus; H&middot;p_map||.
    Measures how well the estimated homography explains each matched pair.
    Average, MaxAbs, and RMS variants reported.</td>
    <td>Quality metric for homography fit. Our improved: avg=2.17px, RMS=2.43px.</td>
  </tr>
  <tr>
    <td><b>Matching accuracy</b></td>
    <td>Fraction of correctly matched craters out of total observed craters:
    Accuracy = |correct correspondences| / |total observed craters|.
    A match is correct if it links to the ground-truth map crater (within position tolerance).</td>
    <td>Our improved: 96.91% at 0% detection error rate.</td>
  </tr>
  <tr>
    <td><b>Navigation success rate</b></td>
    <td>Fraction of Monte Carlo trials in which the homography was successfully estimated
    (at least 4 RANSAC inliers) and the position error was below threshold.
    A trial &ldquo;fails&rdquo; if RANSAC cannot find enough inliers to fit a valid homography.</td>
    <td>Our improved: 100.0% at 0% error rate; still &ge;94% even at 80% detection error rate.</td>
  </tr>
  <tr>
    <td><b>Detection error rate</b></td>
    <td>In the Monte Carlo sweep: the fraction of craters that are either randomly removed (false negatives)
    or replaced with random positions (false positives). 0% = perfect detection. 100% = all craters wrong.
    Simulates YOLO making mistakes at various severity levels.</td>
    <td>Used to test robustness: real landing scenarios will have 5&ndash;20% detection error from shadows/dust.</td>
  </tr>
  <tr>
    <td><b>Gaussian noise (&sigma;=5px)</b></td>
    <td>Each crater centre coordinate is perturbed by a random value drawn from N(0, 5&sup2;).
    At LRO NAC resolution (0.5 m/pixel), 5 pixels = 2.5m position uncertainty in crater centre estimate.
    This simulates subpixel localisation error in the YOLO bounding box centre.</td>
    <td>Applied in every Monte Carlo trial to simulate realistic detection noise.</td>
  </tr>
  <tr>
    <td><b>RMS error</b></td>
    <td>Root Mean Square error = &radic;(mean(e_i&sup2;)). More sensitive to large errors than the simple mean.
    If most errors are small but a few are large, RMS &gt; mean. Used for reprojection error.</td>
    <td>RMS reprojection: 2.43px improved vs 2.31px baseline (slightly worse due to accepting harder matches).</td>
  </tr>
</table>


<!-- ================================================================ -->
<h1>PART 2 &mdash; ALL METRICS: FORMULAS, INTERPRETATION, AND USE</h1>
<!-- ================================================================ -->

<h2>2.1 Detection Metrics (YOLO)</h2>

<h3>Precision</h3>
<div class="formula-box">
Precision = TP / (TP + FP)

where:
  TP = True Positive  = detected box with IoU >= 0.50 against a real crater
  FP = False Positive = detected box with no matching real crater (IoU < 0.50)
  FN = False Negative = real crater with no detected box

Our value: 0.617
Interpretation: 61.7% of our YOLO detections correspond to real craters.
38.3% are false alarms. This is acceptable because our confidence-weighted
voting suppresses low-confidence false alarms during matching.
</div>

<h3>Recall</h3>
<div class="formula-box">
Recall = TP / (TP + FN)

Our value: 0.232
Interpretation: We detect only 23.2% of all ground-truth craters.
This seems low but is acceptable: our matching algorithm only needs 15-30
correctly detected craters per scene to establish reliable correspondences.
The matching is more sensitive to precision (no false alarms) than recall
(no missed craters), because false alarms create misleading triangle vertices.
</div>

<h3>Average Precision (AP) and mAP</h3>
<div class="formula-box">
AP = area under the Precision-Recall curve
   = sum over recall thresholds of (precision_at_threshold * delta_recall)

mAP@50   = AP computed when IoU threshold = 0.50 (loose localisation)
mAP@50-95 = average of AP at IoU thresholds {0.50, 0.55, 0.60, ..., 0.95}
           = stricter: penalises imprecise bounding box localisation

Our values: mAP@50 = 0.416, mAP@50-95 = 0.147
Gap between the two shows our boxes are detected (mAP50 is decent) but not
tightly localised around craters (mAP50-95 drops to 0.147).
For navigation we only need crater centres, so this gap is acceptable.
We extract crater centre = (x_box + w_box/2, y_box + h_box/2).
</div>

<h2>2.2 Matching Metrics</h2>

<h3>Gaussian Kernel Similarity</h3>
<div class="formula-box">
S(i, j) = exp( -||d_i - d_j||^2 / (2 * sigma^2) )

where:
  d_i = 25D second-order descriptor of observation triangle i
  d_j = 25D second-order descriptor of map triangle j
  sigma = 0.20 (bandwidth parameter)
  ||...|| = L2 (Euclidean) norm

Range: [0, 1]
  S = 1.0 when d_i = d_j (identical descriptors)
  S = 0.5 when ||d_i - d_j|| = sigma * sqrt(2*ln(2)) ~ 0.235
  S -> 0 when descriptors are very different

Why Gaussian? It is smooth, differentiable, bounded, and its sigma gives
intuitive control over how forgiving the matching is.
</div>

<h3>Matching Accuracy</h3>
<div class="formula-box">
Matching Accuracy = |correct crater correspondences| / |total observed craters| * 100%

"Correct" = the matched map crater is within position_tolerance of the
            true ground-truth map crater for that observed crater.

Our values:
  Baseline (0% noise): 96.57%
  Improved (0% noise): 96.91%
  Improved (10% noise): 83.93%
  Improved (20% noise): 49.88%

Note: accuracy degrades with detection error rate because spurious craters
create false triangle vertices that compete with true matches.
</div>

<h2>2.3 Navigation Metrics</h2>

<h3>Position Error (% of altitude)</h3>
<div class="formula-box">
After estimating homography H from crater correspondences:

  predicted_position = H^{-1} * image_centre
  true_position      = known from simulation

  position_error_m = ||predicted_position - true_position||

  position_error_X_pct = (error_x_m / altitude_m) * 100%
  position_error_Y_pct = (error_y_m / altitude_m) * 100%

Why % of altitude? Because pixel scale = altitude / focal_length.
A 0.2% error means: at 1000m altitude -> 2m error; at 5000m -> 10m error.
The metric is altitude-independent and comparable across mission phases.

Our improved results:
  X error: 0.1949% (baseline: 0.2834%) -> -31.2%
  Y error: 0.2157% (baseline: 0.3224%) -> -33.1%
  Base paper reported: 0.44% for both axes
</div>

<h3>Reprojection Error</h3>
<div class="formula-box">
For each verified correspondence (p_obs, p_map):

  reprojection_error_i = ||p_obs - H * p_map||  (in pixels)

Reported statistics:
  Average:   mean of reprojection_error_i across all correspondences
  MaxAbs:    max of reprojection_error_i
  RMS:       sqrt(mean(reprojection_error_i^2))

Our improved values:
  Average: 2.172 px (baseline: 2.035 px) -> slightly worse
  MaxAbs:  4.028 px
  RMS:     2.434 px (baseline: 2.309 px) -> slightly worse

The slight increase in reprojection error is expected: our improved matching
accepts harder matches (in noisy scenes) that the baseline would reject.
These harder matches have slightly larger geometric residuals but still
contribute to correct overall pose estimation, hence BETTER position error.
</div>

<h3>Navigation Success Rate</h3>
<div class="formula-box">
Success = (RANSAC found >= 4 inliers AND position error < threshold)

Navigation Success Rate = (successful trials) / (total trials) * 100%

Our improved values:
  0% error rate:   100.0% (baseline: 99.6%)
  10% error rate:  99.5%  (baseline: 87.33%) -> +12.17%
  20% error rate:  99.0%  (baseline: 61.33%)
  40% error rate:  98.0%  (baseline: 25.67%)
  60% error rate:  100.0% (baseline: 21.00%)
  80% error rate:  94.0%  (baseline: 15.00%)

The massive improvement at high error rates is the most significant result
of our improvements, especially the confidence-weighted voting which
effectively filters out the majority of spurious craters.
</div>


<!-- ================================================================ -->
<h1>PART 3 &mdash; DETAILED GRAPH AND RESULT INTERPRETATION</h1>
<!-- ================================================================ -->

<h2>Graph 1: Comprehensive Results Summary (results_summary.png) &mdash; 4 panels</h2>

<h3>Panel (a): Success Rate vs. Detection Error Rate</h3>
<ul>
  <li><b>X-axis:</b> Detection error rate (0% to 100%), where higher means more missed/spurious craters</li>
  <li><b>Blue line (Matching Accuracy):</b> Drops steeply from ~97% at 0% error to ~50% at 20% error to near 0% at 100%.
  This is expected: with 50%+ wrong craters, most triangles contain at least one false vertex.</li>
  <li><b>Red line (Navigation Success):</b> Stays near 100% until ~80% error rate, then gradually falls.
  The key insight: <b>navigation success decouples from matching accuracy</b>.
  Even when only 20% of craters match correctly, RANSAC can still find 4&ndash;6 correct correspondences
  to estimate a valid homography, so navigation still succeeds.</li>
  <li><b>Why the gap between lines?</b> Navigation needs a minimum (4 correct inliers); matching accuracy
  measures the fraction of ALL observed craters correctly matched. A few correct matches = navigation success
  even if most matches fail.</li>
</ul>

<h3>Panel (b): Reprojection Error (&sigma;=5px noise)</h3>
<ul>
  <li><b>Three bars:</b> Average (2.172px), MaxAbs (4.028px), RMS (2.434px)</li>
  <li>Average &lt; RMS because RMS penalises larger errors more. The gap (2.172 vs 2.434) shows
  a few correspondences have larger errors than the average, consistent with RANSAC accepting some
  borderline inliers.</li>
  <li>MaxAbs = 4.028px: the worst single reprojection error. Still under 5px (the noise level we injected),
  showing RANSAC successfully rejected the large outliers.</li>
  <li><b>What is a &ldquo;good&rdquo; reprojection error?</b> For navigation at 0.5m/pixel resolution,
  2.17px &times; 0.5m = 1.09m average spatial error in the matched point, which is excellent.</li>
</ul>

<h3>Panel (c): Navigation Position Error</h3>
<ul>
  <li><b>Three groups (X, Y, Z):</b> Each shows min (green), mean (blue bar), max (red triangle)</li>
  <li><b>X and Y errors</b> are tiny (~0.19%, ~0.22% of altitude). The bar barely shows above zero,
  with min near 0% and max around 0.5&ndash;1%. This tight distribution shows the algorithm is very
  consistent.</li>
  <li><b>Z error (scale &asymp; 3.9% mean)</b> is larger because Z is estimated from the scale change in the
  homography, which is more sensitive to outliers than the translation. Z error appears as scale %,
  meaning the altitude estimate can be off by ~4%. This is acceptable for terrain-relative navigation
  because altitude is usually estimated separately by a radar altimeter.</li>
  <li>The tight concentration of X and Y errors near their means confirms the Monte Carlo trials are
  consistent and not driven by a few lucky/unlucky outliers.</li>
</ul>

<h3>Panel (d): Average Matching Time vs. Error Rate</h3>
<ul>
  <li>Time fluctuates around ~0.152s with small variance (0.148&ndash;0.154s)</li>
  <li>Slight decrease at 40% error rate: fewer valid triangles from noisy crater set = smaller similarity matrix</li>
  <li>At 60% and 80% error: time increases slightly because adaptive threshold keeps more candidates alive
  before rejection, requiring more processing in RANSAC</li>
  <li>The dashed line at 0.1565s is the clean (0% error) mean. All error rates are near this value,
  confirming time complexity is O(M&times;N) where N (map triangles) is fixed and M varies only slightly.</li>
</ul>

<h2>Graph 2: Improvement Comparison (improvement_comparison.png) &mdash; 3 panels</h2>

<h3>Panel (a): Matching Accuracy &amp; Navigation Success</h3>
<ul>
  <li>Two grouped bars: Baseline (blue, solid) vs Improved (orange, hatched)</li>
  <li>Left group: Matching Accuracy 96.6% &rarr; 96.9% (&Delta;=+0.34%) &mdash; small but positive improvement</li>
  <li>Right group: Navigation Success 99.6% &rarr; 100.0% (&Delta;=+0.4%) &mdash;
  going from 99.6% to 100% means we eliminated the last few failure cases, which are the most important
  for a real landing system (one navigation failure could mean mission loss)</li>
</ul>

<h3>Panel (b): Position &amp; Reprojection Errors (lower is better)</h3>
<ul>
  <li>Four pairs of bars. The first two (Pos X, Pos Y) show dramatic improvement: orange bars are
  noticeably shorter than blue (0.195 vs 0.283 for X, 0.216 vs 0.322 for Y)</li>
  <li>The last two (Reproj Avg, Reproj RMS) show slight orange increase &mdash; a known trade-off:
  we accept harder matches that improve navigation but have slightly larger reprojection residuals</li>
  <li>This trade-off is clearly worth it: 31% better navigation for 7% worse reprojection</li>
</ul>

<h3>Panel (c): Robustness to Detection Errors</h3>
<ul>
  <li>Three scenarios: 0% (clean), 10% (realistic), 20% (noisy)</li>
  <li>At 0%: both algorithms nearly equal (tiny +0.34% delta shown)</li>
  <li>At 10%: orange bar shows improved algorithm is clearly better</li>
  <li>At 20%: the delta annotation shows &minus;1.6% &mdash; improved is slightly worse.
  This is because at 20% error, matching accuracy degrades for both, but the adaptive threshold
  in the improved version sometimes becomes too lenient, accepting more spurious matches.
  However, this does not affect navigation success (which stays high for both).</li>
  <li><b>Key message:</b> In real-world conditions (10% realistic error), the improved algorithm
  is significantly more robust.</li>
</ul>

<h2>Graph 3: Matching Accuracy vs. Detection Error Rate (matching_vs_error_rate.png)</h2>
<ul>
  <li>Blue line: Matching Accuracy; Red line: Navigation Success; Grey dashed: 90% threshold reference</li>
  <li>Navigation success (red) stays above 90% threshold until ~80% error &mdash; remarkable robustness</li>
  <li>Matching accuracy (blue) crosses 90% threshold at ~5% error rate, meaning even small numbers of
  wrong detections reduce individual match quality</li>
  <li>The divergence between the two lines is the algorithm&rsquo;s key strength: RANSAC effectively
  &ldquo;salvages&rdquo; navigation from imperfect matching</li>
</ul>

<h2>Graph 4: Position Error Distribution (position_error_distribution.png)</h2>
<ul>
  <li>Three histograms: X error (%), Y error (%), Z error (scale %)</li>
  <li>X and Y histograms: right-skewed distribution concentrated near 0&ndash;1% with long tail to 5%.
  Mean 0.447% (X), 0.492% (Y) for the BASELINE run shown in this plot.
  For IMPROVED run: means drop to 0.195% (X), 0.216% (Y)</li>
  <li>Z error distribution spans 0&ndash;50% with most mass in 5&ndash;15% &mdash;
  altitude estimation is inherently noisier than lateral position</li>
  <li>The right-skew shows that most trials have very low error (algorithm works well) with a tail
  of a few &ldquo;hard&rdquo; cases. The RANSAC step prevents catastrophic outliers (no cases at 100% error)</li>
</ul>

<h2>Graph 5: Delaunay Triangle Graph (triangle_graph.png)</h2>
<ul>
  <li>Red dots = 100 crater centres sampled from a training image</li>
  <li>Blue lines = 180 triangle edges from Delaunay triangulation</li>
  <li>Note the uniform coverage: every region has triangles, no large empty areas (Delaunay maximises coverage)</li>
  <li>Triangles at the edges are larger (fewer craters near borders); triangles in the centre are
  smaller and more uniform (higher crater density in the chosen region)</li>
  <li>After filtering degenerate triangles (very elongated), 180 out of ~200 Delaunay triangles remain</li>
</ul>


<!-- ================================================================ -->
<h1>PART 4 &mdash; ALL CROSS QUESTIONS WITH FULL ANSWERS</h1>
<!-- ================================================================ -->

<h2>Section A: Problem Statement &amp; Motivation</h2>

<div class="q">Q1. Why use craters as landmarks instead of rocks, ridges, or other surface features?</div>
<div class="a">Craters are formed by hypervelocity impacts and are geologically stable over billions of years &mdash;
they do not move, erode significantly, or change appearance across mission timescales. Unlike rocks (unstable),
shadows (lighting-dependent), or ridges (viewpoint-sensitive), craters have a roughly circular geometry
identifiable from any direction and lighting condition. They occur at all scales (8px to hundreds of km),
are distributed globally, and a complete global catalog (Robbins, 1.3M craters) already exists.
Their circular shape also allows reliable radius estimation from bounding boxes.</div>

<div class="q">Q2. What exactly went wrong in Chandrayaan-2? Could TRN have saved it?</div>
<div class="a">Vikram&rsquo;s onboard computer commanded stronger-than-planned braking during the &ldquo;fine braking&rdquo; phase
at ~7.4 km altitude, causing velocity error to accumulate. By ~2.1 km, the lander tilted 10&deg; beyond
its control authority (13&deg; max), and the thrusters couldn&rsquo;t recover. Communication was lost at ~400m.
TRN could have helped by providing an independent position fix at 2&ndash;5 km altitude, allowing the
navigation computer to detect the growing position/velocity error earlier and initiate corrective thrust
before reaching the unrecoverable state.</div>

<div class="q">Q3. Why express position error as % of altitude rather than in metres?</div>
<div class="a">The apparent size of ground features in the camera image is inversely proportional to altitude.
At 1 km altitude: 1 pixel = 0.5m (NAC resolution at 50 km orbit scales down proportionally).
At 10 km: 1 pixel = ~5m. By expressing error as % of altitude, the metric is altitude-independent.
0.2% error at any altitude means the position is known within 0.2% of the spacecraft&rsquo;s height, which
translates directly to landing zone accuracy. The base paper and all TRN literature use this metric for
cross-comparison between systems operating at different altitudes.</div>

<h2>Section B: YOLO &amp; Detection</h2>

<div class="q">Q4. Your recall is only 23%. Isn&rsquo;t that too low? How does matching still work?</div>
<div class="a">Recall of 23% means we detect 1 in 4 ground-truth craters. This is acceptable because:
(1) Even 20&ndash;30 detected craters per scene yield ~50&ndash;90 Delaunay triangles, more than enough for
reliable second-order matching. (2) The base paper similarly does not require detecting all craters.
(3) The matching algorithm is more sensitive to false alarms (FP craters create misleading triangle vertices)
than to missed craters (FN simply means fewer triangles). Our precision of 0.617 means only 38% false alarms,
which is manageable by confidence-weighted voting.</div>

<div class="q">Q5. How is YOLOv8 better than YOLOv7 specifically for crater detection?</div>
<div class="a">Four key advantages: (1) <b>Anchor-free detection:</b> Craters range from 8px to 100+ px diameter.
YOLOv7&rsquo;s pre-defined anchors need careful tuning for this range; YOLOv8 predicts box dimensions freely.
(2) <b>C2f backbone:</b> Better gradient flow for small datasets like ours (12 training images).
(3) <b>Cleaner API:</b> Easier fine-tuning and export. (4) <b>Model sizes:</b> YOLOv8n (3.2M params)
gives faster inference than YOLOv7 (37M params), critical for real-time navigation.</div>

<div class="q">Q6. What is the difference between mAP@50 and mAP@50-95, and why does ours differ so much (0.416 vs 0.147)?</div>
<div class="a">mAP@50 counts a detection as correct if IoU &ge; 0.5 (box overlaps &ge;50% with truth).
mAP@50-95 averages over IoU thresholds {0.5, 0.55, ..., 0.95} &mdash; it requires increasingly tight localisation.
Our large gap shows: we successfully detect craters (mAP@50=0.416 is reasonable) but our bounding boxes
are not precisely localised (mAP@50-95=0.147 drops sharply). This is fine because we only need the crater
centre coordinate (cx = (x+w/2), cy = (y+h/2)), which can still be accurate even if the box boundary is off.</div>

<h2>Section C: Delaunay Triangulation &amp; Descriptors</h2>

<div class="q">Q7. What is Delaunay triangulation? Why not just use random triangles?</div>
<div class="a">Delaunay triangulation maximises the minimum angle of all triangles, guaranteed by the circumcircle
property (no point inside any triangle&rsquo;s circumcircle). Random triangles would produce many thin, elongated
&ldquo;sliver&rdquo; triangles whose descriptors are numerically unstable &mdash; a tiny position change for nearly-collinear
craters causes large changes in the side ratios. Delaunay also produces the unique &ldquo;natural&rdquo; neighbourhood
structure, so the same set of craters always produces the same triangles deterministically. It also minimises
the total number of triangles needed to cover all craters.</div>

<div class="q">Q8. Why are descriptors sorted (using sorted side lengths, sorted neighbour descriptors)?</div>
<div class="a">Sorting achieves rotation and labelling invariance. If three craters A, B, C form a triangle,
which vertex is &ldquo;1st&rdquo;, &ldquo;2nd&rdquo;, &ldquo;3rd&rdquo; is arbitrary. Without sorting, the same triangle seen from
different starting vertices would produce different descriptors. By sorting side lengths (l1 &le; l2 &le; l3)
and sorting neighbour descriptors lexicographically, we create a canonical representation that is the
same regardless of vertex labelling or triangle orientation.</div>

<div class="q">Q9. Explain the 5D descriptor. Why specifically radius ratios? Why not use absolute radii?</div>
<div class="a">Absolute radii are not scale-invariant: at 1 km altitude, a 50m crater has radius=100px;
at 2 km altitude, the same crater has radius=50px. Using ratios (r_small/r_large, r_mid/r_large) preserves
scale invariance because all radii scale by the same factor when altitude changes. Ratios capture whether
craters are similar in size (r1/r3 &asymp; 1.0) or vastly different (r1/r3 &asymp; 0.1). This breaks geometric
degeneracy: two triangles with identical shape but one having craters of sizes {10, 10, 10}px and another
{10, 10, 100}px (very different sizes) will now have different 5D descriptors.</div>

<div class="q">Q10. How does the second-order descriptor grow from 12D to 25D?</div>
<div class="a">Formula: D2 = D1 &times; (1 + K) where D1 = first-order dimension, K = number of neighbours.
Old: D2 = 3 &times; (1 + 3) = 12D.
New: D2 = 5 &times; (1 + 4) = 25D.
The change combines two improvements: extending D1 from 3 to 5 (adding radius ratios) AND extending K
from 3 to 4 (one more neighbour). Together: 5 &times; 5 = 25 vs 3 &times; 4 = 12. The 25D space has much higher
dimensionality, making it extremely unlikely for two different triangle neighbourhoods to have similar descriptors.</div>

<h2>Section D: Matching &amp; RANSAC</h2>

<div class="q">Q11. Write the Gaussian kernel formula and explain each parameter.</div>
<div class="a">
<div class="formula-box">S(i,j) = exp( -||d_i - d_j||^2 / (2 * sigma^2) )

  d_i, d_j : 25D second-order descriptor vectors (L2-normalised)
  sigma     : bandwidth = 0.20 (chosen as 0.15 * sqrt(25/12))
  ||...||   : L2 norm (Euclidean distance)

S=1 when descriptors identical; S->0 when very different.
Half-max (S=0.5) when distance = sigma*sqrt(2*ln2) ~ 0.235
</div>
The sigma value 0.20 was derived by scaling the original &sigma;=0.15 (for 3D descriptors) by
&radic;(25/12) = &radic;2.08 &asymp; 1.44, giving 0.15 &times; 1.44 &asymp; 0.217 &rarr; rounded to 0.20.
This scaling ensures the Gaussian &ldquo;spread&rdquo; is appropriate for the larger 25D space.</div>

<div class="q">Q12. Explain your adaptive threshold. Why 0.75 multiplier? Why floor 0.50, ceiling 0.82?</div>
<div class="a">The 0.75 multiplier means we accept matches that score at least 75% of the best-scoring
matches. If the best matches score 0.9, threshold = 0.675. The rationale: in a good scene, the best scores
are high and we want strict filtering; in a noisy scene, best scores may only be 0.7 so we relax to 0.525.
The floor 0.50 prevents the threshold from dropping so low that random matches pass (S=0.5 corresponds to
a fairly distant descriptor pair). The ceiling 0.82 prevents over-restriction in very clean scenes from
accidentally rejecting valid but slightly imperfect matches.</div>

<div class="q">Q13. What is greedy bipartite matching? Why not use the Hungarian algorithm?</div>
<div class="a">Greedy bipartite: sort all (obs, map) pairs by similarity score descending, accept each pair
if both elements are unmatched, skip otherwise. Time: O(MN log(MN)).
Hungarian algorithm: finds the globally optimal one-to-one assignment. Time: O(N&sup3;) where N = max(M, map_size).
For M=100 obs triangles and N=500 map triangles: greedy runs in milliseconds; Hungarian would be
500&sup3; = 125 million operations. In practice, because our similarity matrix has strong signal (correct
pairs score much higher than incorrect ones), greedy finds nearly the same result as optimal at a fraction
of the cost. The small sub-optimality is irrelevant because RANSAC handles the few wrong assignments.</div>

<div class="q">Q14. What is RANSAC? How many iterations does it run? How does it handle outliers?</div>
<div class="a">RANSAC randomly samples the minimum number of points needed to fit a model (4 for homography),
fits the model, counts how many of the remaining points are &ldquo;inliers&rdquo; (reprojection error &lt; threshold,
typically 3&ndash;5px). Repeats N times. With outlier fraction &epsilon; and desired success probability p,
the required iterations N = log(1&ndash;p) / log(1&ndash;(1&ndash;&epsilon;)&sup4;).
At &epsilon;=0.5 (50% outliers), p=0.99: N = log(0.01)/log(1&ndash;0.0625) &asymp; 72 iterations.
We typically run 100&ndash;500 iterations. RANSAC can tolerate up to ~50% outliers, which is sufficient because
our matching gives &gt;80% accuracy even at 10% detection error.</div>

<div class="q">Q15. What is a homography and how do you extract position from it?</div>
<div class="a">Homography H is a 3&times;3 matrix (8 DOF after fixing scale) that maps homogeneous image coordinates.
For a flat surface viewed from directly above: H encodes translation (tx, ty) and scale (altitude ratio).
Given matched crater pairs in observation (u_i, v_i) and map (x_i, y_i):
H maps (x_i, y_i) &rarr; (u_i, v_i) approximately.
The spacecraft position is recovered from: [tx, ty] = translation component of H,
representing how far the image centre is from the map origin. Scale = altitude change.
Error = ||[tx,ty]_estimated &minus; [tx,ty]_true|| / altitude.</div>

<h2>Section E: Results &amp; Improvements</h2>

<div class="q">Q16. Your position error improved 31% but reprojection error got 7% worse. Isn&rsquo;t that a contradiction?</div>
<div class="a">No, these metrics measure different things. Reprojection error measures per-match geometric
residual after homography fitting. Position error measures final navigation accuracy.
The improved algorithm accepts matches in harder scenes (noisy, high error rate) that the baseline
rejects. These harder matches have slightly larger per-match residuals (higher reprojection error)
but are still geometrically consistent and provide the diversity of spatial correspondences needed
for a more accurate global homography. Think of it like fitting a line: using more data points
(even noisier ones) often gives a better fit than using only a few very precise points clustered
in one region.</div>

<div class="q">Q17. Navigation success is 100% at 60% detection error. That seems implausibly good. Why?</div>
<div class="a">At 60% error, 40% of craters are still correctly detected. From 100 craters &rarr; 40 true, 60 spurious.
These 40 true craters still form ~80 valid Delaunay triangles with correct descriptors. The confidence-weighted
voting assigns ~2.4&times; more weight to these triangles (conf ~0.85) vs spurious ones (conf ~0.35). Combined with
the adaptive threshold which tightens on the good matches, the algorithm extracts 8&ndash;15 correct crater
correspondences. RANSAC then finds &ge;4 inliers reliably, enabling successful homography estimation.
The 100% at 60% is specific to this simulation's parameter regime; at 80% error it drops to 94% as
even the good matches become harder to find.</div>

<div class="q">Q18. You claim to beat the base paper&rsquo;s 0.44% error. But isn&rsquo;t your dataset/simulation different?</div>
<div class="a">Yes, this is an important caveat. Our 0.195% (improved) vs 0.44% (base paper) comparison is
approximate because: (1) our dataset is the Chang&rsquo;E-4 site with higher crater density; (2) our Monte Carlo
uses different noise parameters; (3) the base paper tests on actual LRO imagery while we simulate.
The fair comparison is our improved vs our baseline (same conditions): X error &minus;31.2%, Y error &minus;33.1%.
We exceed the base paper metric as a bonus, likely because our RANSAC implementation is more rigorous and
the crater density in our dataset produces richer triangle neighbourhoods.</div>

<div class="q">Q19. Why did you choose sigma=5px for the Monte Carlo Gaussian noise? Is that realistic?</div>
<div class="a">5px at LRO NAC resolution (0.5 m/pixel from 50km orbit) = 2.5m position uncertainty in crater centre.
YOLO bounding box centres typically have 2&ndash;5px error due to: asymmetric crater morphology (shadow on one
side), rim vs. flat floor ambiguity, and subpixel aliasing. 5px is the value used in the base paper for
fair comparison. In practice, at lower altitudes (higher resolution), pixel noise decreases but crater
shapes become more variable, so 5px remains a reasonable representative value across landing phases.</div>

<div class="q">Q20. How did you compute the new sigma=0.20 for the Gaussian kernel after extending to 25D?</div>
<div class="a">The expected L2 distance between two random unit vectors in D dimensions is &radic;2 (for uniform random
on unit sphere). The &ldquo;effective spread&rdquo; of the distance distribution scales as &radic;D.
Original: &sigma;=0.15 for 12D descriptors.
To maintain the same fraction of &ldquo;nearby&rdquo; vs &ldquo;far&rdquo; descriptor pairs:
&sigma;_new = &sigma;_old &times; &radic;(D_new/D_old) = 0.15 &times; &radic;(25/12) = 0.15 &times; 1.443 = 0.2165 &rarr; rounded to 0.20.</div>

<h2>Section F: Alternatives &amp; Broader Concepts</h2>

<div class="q">Q21. What are alternatives to Delaunay triangulation for grouping craters?</div>
<div class="a">
<ul>
  <li><b>K-nearest neighbour graphs:</b> Connect each crater to its K closest neighbours. Less structured
  than Delaunay; produces more star-shaped patterns than triangles.</li>
  <li><b>Random triangles:</b> Sample random crater triples. Less efficient (many degenerate), requires
  many more candidates for same coverage.</li>
  <li><b>Regular grid partitioning:</b> Divide image into cells, use all craters within each cell.
  Sensitive to grid alignment; loses cross-cell neighbourhood information.</li>
  <li><b>Voronoi diagram:</b> Dual of Delaunay; equally valid but edges rather than faces are the natural
  groupings, making feature definition harder.</li>
  <li>Delaunay is standard because it is unique, deterministic, computationally efficient (O(N log N)),
  and produces the best (most equilateral) triangle shapes.</li>
</ul></div>

<div class="q">Q22. What are alternatives to RANSAC for geometric verification?</div>
<div class="a">
<ul>
  <li><b>PROSAC (Progressive Sample Consensus):</b> Prioritises sampling higher-quality correspondences first;
  converges faster than RANSAC when matches are sorted by quality (as ours are).</li>
  <li><b>GC-RANSAC (Graph-Cut RANSAC):</b> Uses spatial consistency graph to improve inlier/outlier separation.</li>
  <li><b>MAGSAC (Marginalisation in RANSAC):</b> Uses a distribution over inlier thresholds instead of fixed threshold.</li>
  <li><b>Least Median of Squares (LMedS):</b> Minimizes median residual; more robust than RANSAC at very
  high outlier rates but slower.</li>
  <li><b>Homography voting:</b> Instead of RANSAC, use a Hough-like voting in the homography parameter space.
  More expensive but can handle higher outlier rates.</li>
</ul></div>

<div class="q">Q23. Why not use deep learning (CNNs) for crater matching instead of geometric descriptors?</div>
<div class="a">Geometric descriptors have three critical advantages for space navigation:
(1) <b>Interpretability:</b> Engineers can understand exactly why a match was accepted/rejected.
Navigation safety systems require explainable decisions.
(2) <b>No training data needed:</b> CNN matchers require thousands of matched image pairs for training.
For new landing sites, no such pairs exist.
(3) <b>Robustness to domain shift:</b> A CNN trained on Chang&rsquo;E-4 images may fail at other lunar regions.
Geometric descriptors are universal &mdash; they work wherever Delaunay triangulation works.
(4) <b>Computational cost:</b> Running a CNN matcher onboard a spacecraft with limited power is impractical;
our algorithm runs in 0.157s on CPU.
CNNs could be used for crater detection (our YOLO step), where large training datasets exist. But for
matching, geometric methods remain superior in this domain.</div>

<div class="q">Q24. What is the Voronoi diagram and how does it relate to Delaunay triangulation?</div>
<div class="a">The Voronoi diagram partitions the plane into regions such that each region contains all points
closer to one crater centre than to any other. It is the dual graph of the Delaunay triangulation:
connect the circumcentres of adjacent Delaunay triangles and you get the Voronoi edges.
Delaunay edge (A,B) exists &harr; Voronoi regions of A and B share a boundary.
This duality means Delaunay neighbours are exactly the craters whose Voronoi cells touch &mdash;
they are the &ldquo;natural&rdquo; spatial neighbours.</div>

<div class="q">Q25. What would happen if you applied this algorithm to Mars or asteroids instead of the Moon?</div>
<div class="a">The algorithm is fundamentally transferable: (1) Craters exist on all rocky bodies (Mars, Vesta,
Ceres, etc.) with similar formation physics. (2) The geometric descriptor is body-agnostic &mdash; it only
uses crater positions and sizes. (3) You would need: a reference crater catalog for that body, a
detector trained on images from that body&rsquo;s camera. (4) Challenges: Mars has erosion and dust covering
ancient craters; asteroids have irregular shapes making homography estimation harder. The navigation
framework (homography + position error) would need adjustment for non-spherical bodies. But the core
triangle matching algorithm would work.</div>

<div class="q">Q26. What would you do to further improve performance beyond your current 5 improvements?</div>
<div class="a">
<ul>
  <li><b>Elliptical descriptor:</b> Fit an ellipse to each crater (not just a circle); include ellipticity
  and orientation as descriptor dimensions. More discriminative, especially for oblique-impact craters.</li>
  <li><b>CNN-based descriptor:</b> Train a Siamese network on crater patch pairs to learn an optimal
  descriptor embedding. Would require annotated matching pairs.</li>
  <li><b>Multi-scale triangulation:</b> Build triangles at different scales (small: nearby craters,
  large: widely-spaced craters) and vote across scales. Robust to scale-dependent detection failures.</li>
  <li><b>Graph neural networks:</b> Model the entire crater graph as a GNN and compute matching as
  message-passing. Recent work shows GNNs outperform hand-crafted second-order descriptors.</li>
  <li><b>YOLOv8l training on GPU:</b> With more training images and GPU, mAP@50 could reach 0.85+,
  giving higher recall and reducing the need for robustness at high error rates.</li>
</ul></div>

<div class="q">Q27. Why does the Objectives slide say &ldquo;DistilBERT, RoBERTa&rdquo;? (PPT Error)</div>
<div class="a"><b>This is a critical error in the presentation. The content on Slide 4 was accidentally left from a
different project (drug recommendation system). The correct objectives for our project are:
(1) Implement the triangle-based crater matching algorithm from the base IEEE paper.
(2) Evaluate using Monte Carlo simulation on LRO NAC dataset.
(3) Introduce 5 novel improvements achieving &gt;30% position error reduction.
Do NOT attempt to explain DistilBERT or drug recommendations in this viva &mdash; fix the slide immediately.</b></div>

<div class="q">Q28. What is the physical interpretation of the Gaussian kernel&rsquo;s sigma parameter?</div>
<div class="a">Sigma &sigma; defines the &ldquo;sensitivity&rdquo; of the similarity function to descriptor differences.
Geometrically: at distance ||d_i&ndash;d_j|| = &sigma;, the similarity drops to exp(&minus;0.5) &asymp; 0.607.
At distance = 2&sigma;, similarity = exp(&minus;2) &asymp; 0.135 (effectively zero).
So &sigma; defines the &ldquo;neighbourhood radius&rdquo; in descriptor space within which two triangles are
considered similar. Our &sigma;=0.20 in 25D space means triangles with L2 descriptor distance &lt; 0.20
are considered highly similar (S &gt; 0.607), while those with distance &gt; 0.40 are effectively
dissimilar (S &lt; 0.135). This corresponds to about 4&ndash;5% relative change in each of the 25 descriptor
dimensions before similarity drops below 60%.</div>


<!-- ================================================================ -->
<h1>PART 5 &mdash; QUICK REFERENCE SUMMARY TABLES</h1>
<!-- ================================================================ -->

<h2>5.1 Algorithm Pipeline Summary</h2>
<table>
  <tr><th>Step</th><th>What Happens</th><th>Input</th><th>Output</th><th>Key Parameter</th></tr>
  <tr><td>1. Detection</td><td>YOLOv8 finds craters in image</td><td>LRO image (800px)</td><td>N crater boxes + confidences</td><td>conf_threshold=0.25</td></tr>
  <tr><td>2. Triangulation</td><td>Delaunay mesh from centres</td><td>N crater centres</td><td>~2N triangles</td><td>area_threshold, aspect filter</td></tr>
  <tr><td>3. Descriptor</td><td>5D first-order per triangle</td><td>Triangle vertices + radii</td><td>5D vector per triangle</td><td>USE_RADIUS_DESCRIPTOR=True</td></tr>
  <tr><td>4. 2nd-order</td><td>Concatenate self+4 neighbours</td><td>5D vectors + adjacency graph</td><td>25D vector per triangle</td><td>MAX_NEIGHBORS=4</td></tr>
  <tr><td>5. Similarity</td><td>Gaussian kernel matrix</td><td>Obs 25D + Map 25D</td><td>M&times;N similarity matrix</td><td>sigma=0.20</td></tr>
  <tr><td>6. Adaptive thresh</td><td>Compute dynamic threshold</td><td>Similarity matrix</td><td>Effective threshold [0.50, 0.82]</td><td>ADAPTIVE_THRESHOLD=True</td></tr>
  <tr><td>7. Greedy match</td><td>One-to-one triangle assignment</td><td>Similarity + threshold</td><td>Matched triangle pairs</td><td>MATCH_THRESHOLD_FLOOR=0.50</td></tr>
  <tr><td>8. Conf voting</td><td>Weight-vote crater correspondences</td><td>Triangle pairs + YOLO conf</td><td>Crater pair votes</td><td>CONF_WEIGHTED_VOTING=True</td></tr>
  <tr><td>9. RANSAC</td><td>Geometric verification</td><td>Crater correspondences</td><td>Inlier correspondences</td><td>ransac_threshold=5px</td></tr>
  <tr><td>10. Homography</td><td>Estimate camera pose</td><td>4+ inlier pairs</td><td>H matrix, position error</td><td>cv2.findHomography</td></tr>
</table>

<h2>5.2 Our 5 Improvements at a Glance</h2>
<table>
  <tr><th>#</th><th>Improvement</th><th>Problem It Solves</th><th>Key Formula/Change</th><th>Impact</th></tr>
  <tr>
    <td>1</td><td>5D Radius Descriptor</td>
    <td>Geometric ambiguity: different-sized craters in same-shaped triangle had identical descriptors</td>
    <td>Add [r_min/r_max, r_mid/r_max] to 3D descriptor</td>
    <td>Position error &minus;31%</td>
  </tr>
  <tr>
    <td>2</td><td>Adaptive Threshold</td>
    <td>Fixed threshold too strict in noisy scenes, too lenient in clean ones</td>
    <td>thresh = clip(0.75 &times; mean(top-K row-maxima), 0.50, 0.82)</td>
    <td>Nav success +0.4%, better robustness</td>
  </tr>
  <tr>
    <td>3</td><td>Confidence-Weighted Voting</td>
    <td>Spurious detections (low YOLO confidence) contributed equal votes as true craters</td>
    <td>vote &times;= geomean(conf_A, conf_B, conf_C)</td>
    <td>Nav success at 10% error: +12.17%</td>
  </tr>
  <tr>
    <td>4</td><td>YOLOv8s Upgrade</td>
    <td>YOLOv8n has limited feature capacity for small craters</td>
    <td>3.2M params (n) &rarr; 11.2M params (s)</td>
    <td>Better detection quality</td>
  </tr>
  <tr>
    <td>5</td><td>Extended Adjacency (K=4)</td>
    <td>K=3 second-order descriptor had limited neighbourhood context</td>
    <td>12D (3&times;4) &rarr; 25D (5&times;5)</td>
    <td>Better descriptor uniqueness</td>
  </tr>
</table>

<h2>5.3 Full Results Comparison</h2>
<table>
  <tr><th>Metric</th><th>Base Paper</th><th>Our Baseline</th><th>Our Improved</th><th>Change</th></tr>
  <tr><td>Matching Accuracy (0% error)</td><td>~99%</td><td>96.57%</td><td class="metric-good">96.91%</td><td>+0.34%</td></tr>
  <tr><td>Navigation Success Rate</td><td>~100%</td><td>99.60%</td><td class="metric-good">100.0%</td><td>+0.40%</td></tr>
  <tr><td>Position Error X (% altitude)</td><td>0.44%</td><td>0.2834%</td><td class="metric-good">0.1949%</td><td class="metric-good">&minus;31.2%</td></tr>
  <tr><td>Position Error Y (% altitude)</td><td>0.44%</td><td>0.3224%</td><td class="metric-good">0.2157%</td><td class="metric-good">&minus;33.1%</td></tr>
  <tr><td>Reprojection Error Avg (px)</td><td>N/A</td><td>2.035</td><td class="metric-neutral">2.172</td><td>+6.7%</td></tr>
  <tr><td>Reprojection Error RMS (px)</td><td>N/A</td><td>2.309</td><td class="metric-neutral">2.434</td><td>+5.4%</td></tr>
  <tr><td>Matching Time (s/image)</td><td>~0.1s</td><td>0.073s</td><td class="metric-neutral">0.157s</td><td>2.1&times;</td></tr>
  <tr><td>Nav Success at 10% error</td><td>N/A</td><td>87.33%</td><td class="metric-good">99.5%</td><td class="metric-good">+12.17%</td></tr>
  <tr><td>Nav Success at 20% error</td><td>N/A</td><td>61.33%</td><td class="metric-good">99.0%</td><td class="metric-good">+37.67%</td></tr>
  <tr><td>mAP@50 (YOLO)</td><td>N/A (YOLOv7)</td><td>0.416</td><td>0.416</td><td>(same model trained)</td></tr>
  <tr><td>YOLO Precision</td><td>N/A</td><td>0.617</td><td>0.617</td><td>&mdash;</td></tr>
  <tr><td>YOLO Recall</td><td>N/A</td><td>0.232</td><td>0.232</td><td>&mdash;</td></tr>
</table>

<h2>5.4 Complete Glossary</h2>
<table>
  <tr><th>Term</th><th>Quick Definition</th></tr>
  <tr><td>LRO NAC</td><td>Lunar Reconnaissance Orbiter Narrow Angle Camera, 0.5 m/pixel</td></tr>
  <tr><td>TRN</td><td>Terrain-Relative Navigation: position estimation from landmark matching</td></tr>
  <tr><td>Delaunay triangulation</td><td>Triangulation maximising minimum angle, circumcircle property</td></tr>
  <tr><td>Circumcircle</td><td>Circle passing through all 3 triangle vertices; no other points inside for Delaunay</td></tr>
  <tr><td>First-order descriptor</td><td>Per-triangle shape signature (scale/rotation/translation invariant)</td></tr>
  <tr><td>Second-order descriptor</td><td>Triangle + sorted neighbour descriptors concatenated (25D improved)</td></tr>
  <tr><td>Gaussian kernel</td><td>S = exp(&minus;dist&sup2;/2&sigma;&sup2;), maps L2 distance to [0,1] similarity</td></tr>
  <tr><td>Greedy bipartite matching</td><td>One-to-one assignment by greedily accepting top scores</td></tr>
  <tr><td>RANSAC</td><td>Random Sample Consensus: robust model fitting by random subsampling</td></tr>
  <tr><td>Homography</td><td>3&times;3 projective transformation matrix mapping 2D point sets</td></tr>
  <tr><td>Reprojection error</td><td>Pixel distance: observed crater vs H &times; map crater position</td></tr>
  <tr><td>Monte Carlo simulation</td><td>Repeated random trials to estimate statistical performance</td></tr>
  <tr><td>Adaptive threshold</td><td>Threshold computed from top-K row-maxima of similarity matrix</td></tr>
  <tr><td>Confidence-weighted voting</td><td>Vote scaled by geometric mean of YOLO detection confidences</td></tr>
  <tr><td>mAP@50</td><td>Mean Average Precision: AP at 50% IoU threshold</td></tr>
  <tr><td>mAP@50-95</td><td>mAP averaged over IoU thresholds 50% to 95% (stricter)</td></tr>
  <tr><td>IoU</td><td>Intersection over Union: overlap / union of two bounding boxes</td></tr>
  <tr><td>Precision</td><td>TP/(TP+FP): fraction of detections that are real craters</td></tr>
  <tr><td>Recall</td><td>TP/(TP+FN): fraction of real craters that are detected</td></tr>
  <tr><td>C2f block</td><td>YOLOv8 backbone module: Cross Stage Partial with 2 sub-bottlenecks</td></tr>
  <tr><td>CraterDANet</td><td>Paper (Yang et al.) that created our LRO NAC dataset via domain adaptation</td></tr>
  <tr><td>Chang&rsquo;E-4</td><td>CNSA lunar lander (2019), first far-side landing, Von K&aacute;rm&aacute;n crater</td></tr>
  <tr><td>Robbins catalog</td><td>Global lunar crater database: 1.3M craters &gt;1 km (JGR Planets, 2019)</td></tr>
  <tr><td>CDR</td><td>Calibrated Data Record: radiometrically corrected LRO image product</td></tr>
  <tr><td>Anchor-free detection</td><td>Object detection without pre-defined anchor box sizes (YOLOv8)</td></tr>
  <tr><td>Scale invariance</td><td>Descriptor unchanged when all distances multiplied by a constant</td></tr>
  <tr><td>Rotation invariance</td><td>Descriptor unchanged when triangle is rotated (achieved by sorting)</td></tr>
  <tr><td>Degenerate triangle</td><td>Nearly collinear or very small triangle: produces unstable descriptors</td></tr>
  <tr><td>Bipartite graph</td><td>Graph with two disjoint vertex sets (obs triangles, map triangles); matching connects them</td></tr>
  <tr><td>Inlier (RANSAC)</td><td>Correspondence consistent with estimated model (reprojection error &lt; threshold)</td></tr>
  <tr><td>Outlier (RANSAC)</td><td>Correspondence inconsistent with model: wrong match from voting</td></tr>
  <tr><td>Voronoi diagram</td><td>Dual of Delaunay triangulation; region around each point closer to it than any other</td></tr>
  <tr><td>RMS</td><td>Root Mean Square: &radic;(mean(e&sup2;)), sensitive to large errors</td></tr>
</table>

<h2>5.5 References</h2>
<ol>
  <li>Base Paper: &ldquo;Lunar Crater Matching With Triangle-Based Global Second-Order Similarity for Precision Navigation&rdquo; &mdash; IEEE Xplore, 2024.</li>
  <li>Yang et al., &ldquo;CraterDANet: A CNN for Small-Scale Crater Detection via Synthetic-to-Real Domain Adaptation,&rdquo; IEEE TGRS.</li>
  <li>Robbins, S. J., &ldquo;A New Global Database of Lunar Impact Craters &gt;1&ndash;2 km,&rdquo; JGR Planets, 2019.</li>
  <li>Fischler, M. A. &amp; Bolles, R. C., &ldquo;Random Sample Consensus,&rdquo; Communications of the ACM, 1981.</li>
  <li>Delaunay, B., &ldquo;Sur la sphere vide,&rdquo; Bulletin of the Academy of Sciences of the USSR, 1934.</li>
  <li>Ultralytics YOLOv8 Documentation: https://docs.ultralytics.com/</li>
  <li>LRO NAC Data: https://lroc.im-ldi.com/images/downloads</li>
</ol>

</body>
</html>
"""

def convert_html_to_pdf(html_string, output_path):
    with open(output_path, "wb") as f:
        pisa_status = pisa.CreatePDF(html_string, dest=f)
    return not pisa_status.err

if __name__ == "__main__":
    print("Generating PDF...")
    success = convert_html_to_pdf(HTML, OUTPUT_PDF)
    if success:
        size_kb = os.path.getsize(OUTPUT_PDF) // 1024
        print(f"PDF created: {OUTPUT_PDF}")
        print(f"File size: {size_kb} KB")
    else:
        print("PDF generation failed. Check errors above.")
