import os

# Repository root path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

# Common paths
RAW_DATA_DIR = os.path.join(REPO_ROOT, "rawdata")
DATASETS_DIR = os.path.join(REPO_ROOT, "datasets")
RUNS_DIR = os.path.join(REPO_ROOT, "runs")
DB_PATH = os.path.join(REPO_ROOT, "steel_defects.db")
LOG_FILE_PATH = os.path.join(REPO_ROOT, "defect_detection.log")

# Defect categories (Classes)
DEFECT_CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches"
]

def calculate_severity(confidence, box_width, box_height):
    """
    YOLOv8 좌표계는 0~1로 정규화되어 있으므로, box_width * box_height 가 면적 비율(area_ratio)이 됩니다.
    - 면적 비율이 10% 이상이거나 신뢰도(Confidence)가 0.9 이상이면 'HIGH'
    - 면적 비율이 5% 이상이거나 신뢰도(Confidence)가 0.7 이상이면 'MEDIUM'
    - 그 외에는 'LOW'로 판정합니다.
    """
    area_ratio = box_width * box_height
    
    if area_ratio >= 0.10 or confidence >= 0.90:
        return 'HIGH'
    elif area_ratio >= 0.05 or confidence >= 0.70:
        return 'MEDIUM'
    else:
        return 'LOW'
