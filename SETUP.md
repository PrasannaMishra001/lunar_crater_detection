# Setup and Replication Guide

Instructions to install dependencies, prepare the dataset, and reproduce all results from the project.

---

## Prerequisites

- Python 3.10 or higher
- pip package manager
- Git
- (Optional) NVIDIA GPU with CUDA for faster YOLOv8 training

---

## 1. Clone the Repository

```bash
git clone https://github.com/PrasannaMishra001/lunar_crater_detection.git
cd lunar_crater_detection
```

---

## 2. Install Dependencies

```bash
cd lunar_crater_project
pip install -r requirements.txt
```

If you have an NVIDIA GPU and want to use CUDA-accelerated PyTorch, install the GPU version instead:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install ultralytics opencv-python scipy scikit-learn matplotlib Pillow tqdm pandas
```

For CPU-only systems:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install ultralytics opencv-python scipy scikit-learn matplotlib Pillow tqdm pandas
```

---

## 3. Verify Installation

```bash
python -c "import torch; print('PyTorch:', torch.__version__); import cv2; print('OpenCV:', cv2.__version__); import ultralytics; print('Ultralytics:', ultralytics.__version__); import scipy; print('SciPy:', scipy.__version__)"
```

---

## 4. Run the Full Evaluation Pipeline

### Quick mode (100 Monte Carlo trials, no error sweep -- takes about 2 minutes)

```bash
python main.py --quick --skip-sweep
```

### Standard mode (1000 trials + full error sweep -- takes about 30 minutes on CPU)

```bash
python main.py
```

### Full pipeline with YOLOv8 training (adds 30-60 minutes on CPU)

```bash
python main.py --train-yolo
```

### Use trained YOLOv8 model for detection instead of ground truth

```bash
python main.py --yolo-detect
```

---

## 5. Run Individual Components

### Prepare the YOLO-format dataset

```bash
python prepare_yolo.py
```

### Train YOLOv8n crater detector only

```bash
python train_yolo.py --epochs 100 --imgsz 800 --batch 4
```

### Validate a trained YOLOv8n model

```bash
python train_yolo.py --val-only
```

### Run triangle matching smoke test

```bash
python triangle_matching.py
```

### Run navigation module test

```bash
python navigation.py
```

### Run Monte Carlo metrics test (100 trials)

```bash
python metrics.py
```

---

## 6. Command-Line Options for main.py

| Flag | Description |
|------|-------------|
| `--quick` | Run 100 trials instead of 1000 (faster) |
| `--skip-sweep` | Skip the detection error rate sweep |
| `--train-yolo` | Train YOLOv8n before running the matching pipeline |
| `--yolo-detect` | Use trained YOLOv8 for crater detection (instead of ground truth annotations) |
| `--image-index N` | Use the Nth training image as the map (default: 0) |

---

## 7. Output Files

All results are saved to `lunar_crater_project/results/`:

| File | Description |
|------|-------------|
| `results_summary.png` | 4-panel comprehensive results figure |
| `matching_vs_error_rate.png` | Accuracy and navigation success vs error rate |
| `reprojection_error.png` | Reprojection error bar chart |
| `position_error_distribution.png` | Position error histograms |
| `triangle_graph.png` | Delaunay triangle mesh visualization |
| `train_sample_craters.jpg` | Crater detections on a training image |
| `full_results_summary.json` | All numeric results in JSON format |

Trained YOLOv8n weights are saved to `lunar_crater_project/models/yolov8n_craters/weights/best.pt`.

---

## 8. Expected Results (Baseline for Verification)

When running the full pipeline with default settings, you should see results close to the following:

| Metric | Expected Value |
|--------|---------------|
| Matching Accuracy (0% error) | 95-98% |
| Navigation Success Rate | 99-100% |
| Position Error X (% altitude) | 0.25-0.35% |
| Position Error Y (% altitude) | 0.30-0.40% |
| Reprojection Error Avg | 1.8-2.5 px |
| Matching Time per image | 0.05-0.30 s |

Minor variations are expected due to the stochastic nature of the Monte Carlo simulation.

---

## 9. Troubleshooting

**Memory errors during matching:**
The similarity matrix computation uses chunked processing by default. If you still encounter memory issues, reduce the number of craters by editing `config.py` and lowering `MAX_TRIANGLES`.

**YOLOv8 training is slow on CPU:**
Training on CPU takes 30-60 minutes for 50 epochs. For faster results, use a machine with an NVIDIA GPU, or reduce epochs: `python train_yolo.py --epochs 20`.

**Unicode encoding errors on Windows:**
If you see `UnicodeEncodeError` in the console output, set the environment variable before running:
```bash
set PYTHONIOENCODING=utf-8
python main.py --quick
```
