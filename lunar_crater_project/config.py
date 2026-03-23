"""
Configuration for Lunar Crater Matching Project
Triangle-Based Global Second-Order Similarity for Precision Navigation
Group 3 - AIP Project
"""
import os

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'Lunar_Crater_Detection_Data-main', 'LRO_DATA')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR  = os.path.join(DATA_DIR, 'test')

YOLO_DATA_DIR    = os.path.join(BASE_DIR, 'yolo_dataset')
YOLO_TRAIN_IMG   = os.path.join(YOLO_DATA_DIR, 'images', 'train')
YOLO_VAL_IMG     = os.path.join(YOLO_DATA_DIR, 'images', 'val')
YOLO_TRAIN_LABEL = os.path.join(YOLO_DATA_DIR, 'labels', 'train')
YOLO_VAL_LABEL   = os.path.join(YOLO_DATA_DIR, 'labels', 'val')
YOLO_YAML        = os.path.join(YOLO_DATA_DIR, 'data.yaml')

RESULTS_DIR = os.path.join(BASE_DIR, 'results')
MODELS_DIR  = os.path.join(BASE_DIR, 'models')

for d in [YOLO_TRAIN_IMG, YOLO_VAL_IMG, YOLO_TRAIN_LABEL, YOLO_VAL_LABEL,
          RESULTS_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

# ─── Dataset ──────────────────────────────────────────────────────────────────
# Pairs sharing the same scene (same M-number prefix): train → test
SHARED_SCENE_PAIRS = {
    'M115143943': ('M115143943_mosaic_train_small', 'M115143943RE_cal_echo_2_2'),
    'M1183658592': ('M1183658592_mosaic_train_small', 'M1183658592LE_cal_echo_23_0'),
    'M1229577647': ('M1229577647_mosaic_train_small', 'M1229577647LE_cal_echo_14_4'),
    'M161142858':  ('M161142858_mosaic_train_small',  'M161142858RE_cal_echo_1_2'),
}

# ─── YOLOv8 ───────────────────────────────────────────────────────────────────
YOLO_MODEL    = 'yolov8n.pt'     # nano — fastest, sufficient for small craters
YOLO_EPOCHS   = 100
YOLO_IMGSZ    = 800
YOLO_BATCH    = 8
YOLO_CONF     = 0.25             # detection confidence threshold
YOLO_IOU      = 0.45             # NMS IoU threshold
MIN_CRATER_PX = 8                # ignore craters smaller than 8px diameter

# ─── Triangle Matching ────────────────────────────────────────────────────────
MAX_NEIGHBORS        = 3         # max adjacent triangles per triangle
SIMILARITY_SIGMA     = 0.15      # Gaussian kernel bandwidth for similarity
MATCH_THRESHOLD      = 0.70      # min similarity score to accept a match
MIN_MATCH_CRATERS    = 4         # minimum matched crater pairs for navigation
RANSAC_REPROJ_THRESH = 5.0       # RANSAC reprojection threshold (pixels)
MAX_TRIANGLES        = 2000      # max triangles to build (for speed)

# ─── Monte Carlo Simulation ───────────────────────────────────────────────────
MC_TRIALS        = 1000          # number of trials
MC_SIGMA         = 5.0           # Gaussian noise on crater centers (pixels)
MC_FALSE_RATES   = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]  # FD + miss rates (%)
MC_MIN_CRATERS   = 10            # min craters in observation for valid trial

# ─── Navigation ───────────────────────────────────────────────────────────────
# Simulated camera parameters
IMAGE_WIDTH  = 1000              # pixels
IMAGE_HEIGHT = 1000              # pixels
ALTITUDE_FACTOR = 1.0            # km per 1000 pixels (for % error calc)
# Position error = translation_pixels / IMAGE_WIDTH * 100 (% of image = % of altitude)

# ─── Delaunay Triangulation ───────────────────────────────────────────────────
DELAUNAY_MAX_SIDE_RATIO = 10.0   # filter triangles with extreme aspect ratios
MIN_TRIANGLE_AREA       = 20.0   # minimum triangle area in pixels²
