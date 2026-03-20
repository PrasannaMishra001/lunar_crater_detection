# Lunar Crater Matching Using Triangle-Based Global Second-Order Similarity for Precision Navigation

**Applied Image Processing (AIP) -- Group 3**

**Under the guidance of:** [Dr. Mahua Bhattacharya](https://www.iiitm.ac.in/index.php/en/component/splms/teacher/Dr.Mahua), ABV-IIITM Gwalior

| Roll Number   | Name              |
|---------------|-------------------|
| 2023IMT-014   | Ankit Baidsen     |
| 2023IMT-050   | Malladi Nagarjuna |
| 2023IMT-059   | Prasanna Mishra   |
| 2023IMT-060   | Prasun Baranwal   |
| 2023IMT-073   | Shivam Deolankar  |

---

## Problem Statement

Accurate autonomous navigation during lunar landing remains one of the most critical challenges in space exploration. Small errors in position estimation during descent can lead to unsafe landings or mission failure. Historical missions have highlighted this challenge -- Apollo-era astronauts frequently had to manually correct trajectories, Beresheet (2019) lost control during descent, and Chandrayaan-2 (2019) lost communication with the Vikram lander during its final descent phase.

Modern missions target especially challenging regions such as the lunar south pole, introducing additional constraints including poor lighting, extreme shadow regions, uneven terrain, and regolith dust affecting visibility. These challenges motivate the need for robust terrain-relative navigation systems that rely on stable and reliable landmarks such as lunar craters.

This project implements and evaluates the **triangle-based crater matching algorithm** proposed in the base paper, which groups nearby craters into triangles, represents them as a graph structure, and applies **global second-order similarity** to compare both triangle geometry and their spatial relationships for precision navigation.

---

## Base Paper

**"Lunar Crater Matching With Triangle-Based Global Second-Order Similarity for Precision Navigation"**
- IEEE Xplore: [https://ieeexplore.ieee.org/document/11123425/](https://ieeexplore.ieee.org/document/11123425/)
- Secondary reference: [https://ieeexplore.ieee.org/document/10964403/](https://ieeexplore.ieee.org/document/10964403/)

---

## Algorithm Overview

The core algorithm consists of the following stages:

### 1. Crater Detection (YOLOv8n)
We use YOLOv8n (nano variant) from Ultralytics as a direct upgrade over the base paper's YOLOv7. YOLOv8n offers faster inference, improved small-object detection, and a cleaner PyTorch-based implementation. The detector identifies crater bounding boxes from LRO NAC imagery.

### 2. Delaunay Triangulation
Detected crater centers are triangulated using Delaunay triangulation to form a mesh of triangles. Degenerate triangles (too elongated or too small) are filtered out.

### 3. First-Order Descriptor
Each triangle is described by a scale-, rotation-, and translation-invariant descriptor:
- Sorted normalized side ratios: (l1/l3, l2/l3) where l1 <= l2 <= l3
- Perimeter-normalized area for additional discriminability

### 4. Second-Order Descriptor
The second-order descriptor captures the geometric context of each triangle's neighborhood. It concatenates the triangle's own first-order descriptor with the sorted first-order descriptors of its adjacent triangles (those sharing an edge). This makes the descriptor far more discriminative than first-order alone.

### 5. Global Matching with RANSAC Verification
- A Gaussian kernel similarity matrix is computed between observation and map triangle descriptors.
- Greedy bipartite matching selects the best one-to-one triangle correspondences.
- Crater-level correspondences are extracted via a voting scheme from matched triangle pairs.
- RANSAC-based geometric verification (homography estimation) filters out geometrically inconsistent matches, enforcing global consistency.

### 6. Navigation / Pose Estimation
From the verified crater correspondences, a homography is estimated to determine the camera's position. Position estimation error is reported as a percentage of flight altitude (the primary metric from the base paper).

---

## Key Improvements Over Base Paper

| Aspect | Base Paper | Our Implementation |
|--------|-----------|-------------------|
| Detection Model | YOLOv7 | YOLOv8n (nano) -- faster, better on small objects |
| Geometric Verification | Not explicitly detailed | RANSAC-based homography verification after triangle matching |
| Memory Efficiency | Not specified | Chunked similarity matrix computation for large crater sets |
| Matching Strategy | Not detailed | Greedy bipartite + per-row top-K candidate pruning |

---

## Dataset

**LRO NAC Lunar Crater Detection Dataset** from the CraterDANet paper.

Source: [https://github.com/yizuifangxiuyh/Lunar_Crater_Detection_Data](https://github.com/yizuifangxiuyh/Lunar_Crater_Detection_Data)

| Property | Value |
|----------|-------|
| Region | Chang'E-4 landing site (45-46 S, 176.4-178.8 E) |
| Camera | LRO Narrow Angle Camera (NAC), 0.5 m/pixel |
| Training images | 12 (800 x 800 px mosaics) |
| Test images | 8 (1000 x 1000 px NAC CDR tiles) |
| Training craters | 13,453 annotated |
| Test craters | 9,797 annotated |
| Min crater diameter | 8 pixels |

Additional dataset references used for validation and context:
- USGS Crater Database: [https://astrogeology.usgs.gov/search/map/moon_crater_database_v1_robbins](https://astrogeology.usgs.gov/search/map/moon_crater_database_v1_robbins)
- ISRO CHMap Browser: [https://chmapbrowse.issdc.gov.in/MapBrowse/](https://chmapbrowse.issdc.gov.in/MapBrowse/)
- LROC Image Downloads: [https://lroc.im-ldi.com/images/downloads](https://lroc.im-ldi.com/images/downloads)

---

## Results

All evaluation follows the same metrics as the base paper: Monte Carlo simulation with 1000 trials, Gaussian noise (sigma = 5 pixels) on crater centers.

### Comparison with Base Paper

| Metric | Base Paper | Ours | Status |
|--------|-----------|------|--------|
| Matching Accuracy (0% error, 1000 trials) | ~99% | 96.57% | Comparable |
| Mismatches per image (mean) | ~0 | 0.51 | Comparable |
| Navigation Success Rate | ~100% | 99.60% | Comparable |
| Position Error X (% of altitude) | 0.44% | **0.2834%** | Improved |
| Position Error Y (% of altitude) | 0.44% | **0.3224%** | Improved |
| Position Error XY total | 0.44% | 0.4748% | Comparable |
| Reprojection Error Average (px) | N/A | 2.035 | -- |
| Reprojection Error RMS (px) | N/A | 2.309 | -- |
| Average Matching Time (s/image) | ~0.1s | **0.073s** | Faster |

Our implementation achieves position errors in X (0.28%) and Y (0.32%) that individually beat the base paper's reported 0.44% orbital baseline. The combined XY error (0.47%) is comparable. Navigation succeeds in 99.6% of trials.

### YOLOv8n Crater Detection

| Metric | Value |
|--------|-------|
| mAP@50 | 0.416 |
| mAP@50-95 | 0.147 |
| Precision | 0.617 |
| Recall | 0.232 |
| Inference Speed (CPU) | 101.7 ms/image |

Note: The detector was trained for 50 epochs on CPU with only 12 images. Performance will improve significantly with more epochs and GPU training.

### Detection Error Rate Sweep (300 trials per rate)

| Error Rate | Matching Accuracy | Navigation Success |
|-----------|------------------|-------------------|
| 0% | 96.10% | 99.33% |
| 10% | 80.03% | 87.33% |
| 20% | 51.46% | 61.33% |
| 30% | 29.36% | 42.33% |
| 40% | 16.25% | 25.67% |
| 50% | 7.98% | 22.33% |
| 60% | 4.16% | 21.00% |
| 70% | 3.76% | 18.67% |
| 80% | 2.09% | 15.00% |
| 90% | 1.09% | 13.67% |
| 100% | 0.66% | 10.00% |

---

## Visualizations

### Comprehensive Results Summary

![Results Summary](lunar_crater_project/results/results_summary.png)

Four-panel figure showing: (a) matching and navigation success rates vs detection error rate, (b) reprojection error statistics, (c) position estimation error in X/Y/Z, and (d) matching time vs error rate.

### Matching Accuracy vs Detection Error Rate

![Matching vs Error Rate](lunar_crater_project/results/matching_vs_error_rate.png)

The algorithm maintains over 80% matching accuracy at 10% detection error rate, demonstrating robustness to moderate levels of false and missed detections.

### Reprojection Error

![Reprojection Error](lunar_crater_project/results/reprojection_error.png)

Average reprojection error of 2.035 pixels with RMS of 2.309 pixels, indicating high-quality geometric alignment between matched crater pairs.

### Position Error Distribution

![Position Error Distribution](lunar_crater_project/results/position_error_distribution.png)

Distribution of navigation position errors across 1000 Monte Carlo trials, showing tight concentration around the mean values.

### Delaunay Triangle Graph

![Triangle Graph](lunar_crater_project/results/triangle_graph.png)

Visualization of the Delaunay triangulation built from 100 crater centers, forming 180 valid triangles used for the matching algorithm.

### Sample Crater Detections

![Sample Detections](lunar_crater_project/results/train_sample_craters.jpg)

Ground truth crater annotations overlaid on an LRO NAC mosaic training image (M115143943, 1222 craters).

### YOLOv8n Training Curves

![YOLO Training Results](lunar_crater_project/models/yolov8n_craters/results.png)

Training and validation loss curves, precision-recall, and mAP metrics over 50 epochs.

### YOLOv8n Predictions vs Ground Truth

| Ground Truth Labels | YOLOv8n Predictions |
|---|---|
| ![GT](lunar_crater_project/models/yolov8n_craters/val_batch0_labels.jpg) | ![Pred](lunar_crater_project/models/yolov8n_craters/val_batch0_pred.jpg) |

---

## Performance Metrics Used

Following the base paper, evaluation is conducted at three levels:

**Matching Level:**
- Matching accuracy (%) over 1000 Monte Carlo trials
- Number of mismatches per image
- Matching success rate under varying false and missed detection error rates (0-100%)
- Average matching time in seconds per image

**Navigation Level:**
- Position estimation error as percentage of flight altitude in X, Y, and Z directions (average, maximum, minimum)
- Reprojection error in pixels (average, MaxAbsError, RMS)

**Monte Carlo Simulation:**
- 1000 trials with Gaussian noise (sigma = 5 pixels) on crater centers
- Statistical characterization of how detection uncertainty propagates into navigation error

---

## Project Structure

```
lunar_crater_detection/
|-- README.md                  # This file
|-- SETUP.md                   # Installation and replication instructions
|-- AIP_Project_Group3.pdf     # Project proposal document
|-- Lunar_Crater_Matching_...pdf  # Base paper
|
|-- Lunar_Crater_Detection_Data-main/   # Dataset
|   |-- LRO_DATA/
|   |   |-- train/             # 12 training images + annotations
|   |   |-- test/              # 8 test images + annotations
|   |-- README.md
|   |-- usage_example.py
|
|-- lunar_crater_project/      # Implementation
|   |-- config.py              # Hyperparameters, paths, settings
|   |-- data_loader.py         # LRO dataset loader
|   |-- prepare_yolo.py        # Convert annotations to YOLO format
|   |-- train_yolo.py          # YOLOv8n training script
|   |-- detect.py              # Crater detection (GT or YOLOv8)
|   |-- triangle_matching.py   # Core: triangle-based 2nd-order similarity
|   |-- navigation.py          # Pose estimation and error computation
|   |-- metrics.py             # All performance metrics + Monte Carlo
|   |-- visualize.py           # Result plots and figures
|   |-- main.py                # Full evaluation pipeline
|   |-- requirements.txt       # Python dependencies
|   |
|   |-- results/               # Generated output
|   |   |-- results_summary.png
|   |   |-- matching_vs_error_rate.png
|   |   |-- reprojection_error.png
|   |   |-- position_error_distribution.png
|   |   |-- triangle_graph.png
|   |   |-- full_results_summary.json
|   |
|   |-- models/                # Trained YOLOv8n weights
|   |-- yolo_dataset/          # YOLO-format data (auto-generated)
```

---

## Setup and Replication

For detailed installation instructions and commands to reproduce all results, see [SETUP.md](SETUP.md).

---

## References

1. Base Paper: "Lunar Crater Matching With Triangle-Based Global Second-Order Similarity for Precision Navigation" -- IEEE, 2024. [Link](https://ieeexplore.ieee.org/document/11123425/)
2. Yang et al., "CraterDANet: A Convolutional Neural Network for Small-Scale Crater Detection via Synthetic-to-Real Domain Adaptation," IEEE TGRS.
3. Robbins, S. J., "A New Global Database of Lunar Impact Craters >1-2 km," JGR Planets, 2019.
4. Ultralytics YOLOv8 Documentation: [https://docs.ultralytics.com/](https://docs.ultralytics.com/)
5. LRO NAC Data: [https://lroc.im-ldi.com/images/downloads](https://lroc.im-ldi.com/images/downloads)

---

## License

This project is for academic purposes under ABV-IIITM Gwalior, Applied Image Processing course.
