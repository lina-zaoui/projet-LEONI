from pathlib import Path

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "model" / "modeling" / "model1_cable_seg.pt"

INPUT_DIR = BASE_DIR / "data" / "input-model1" /"segmentation"/ "valid" / "images"

OUTPUT_DIR = BASE_DIR / "data" / "output"/"valid_output"

MASK_DIR = OUTPUT_DIR / "masks"
CROP_DIR = OUTPUT_DIR / "crops"
VISUALIZATION_DIR = OUTPUT_DIR / "visualizations"
RESULTS_DIR = OUTPUT_DIR / "results"


# --------------------------------------------------
# YOLO PARAMETERS
# --------------------------------------------------

IMAGE_SIZE =  1024

# Start conservatively.
# We want to see weak detections during development.
CONFIDENCE_THRESHOLD = 0.20

# NMS IoU threshold.
IOU_THRESHOLD = 0.50


# ============================================================
# CABLE FILTERING
# ============================================================

# Your Model 1 should contain ONE class:
#
# 0 = cable

CABLE_CLASS_ID = 0


# ============================================================
# CROP
# ============================================================

# Extra pixels around the cable bounding box.
#
# This prevents Model 2 from receiving a crop that is too tight.

CROP_PADDING = 250


# ============================================================
# EXPECTED NUMBER OF CABLES
# ============================================================

EXPECTED_CABLES = 4