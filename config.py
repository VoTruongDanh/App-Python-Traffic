"""
Configuration file cho Video Processing Application
"""
import os

# =============================================================================
# GPU SETTINGS
# =============================================================================
# Set to True để sử dụng GPU (nếu có), False để force CPU
USE_GPU = True  # Auto-enable GPU by default

# =============================================================================
# MODEL PATHS
# =============================================================================
MODEL_PERSON_PATH = os.path.join("..", "best.pt")  # Model đã train cho Person (Train1)
MODEL_TRAIN2_PATH = os.path.join("..", "Train2", "best.pt")  # Model Train2 (Multi-class)
MODEL_VEHICLE_DEFAULT = "yolov3.pt"  # Model gốc cho Vehicle

# =============================================================================
# CLASS DEFINITIONS
# =============================================================================
# COCO classes (for base YOLO models)
VEHICLE_CLASSES = [1, 2, 3, 5, 7]  # Bicycle, Car, Motorcycle, Bus, Truck
PERSON_CLASS = 0

CLASS_NAMES = {
    0: 'Person',
    1: 'Bicycle',
    2: 'Car',
    3: 'Motorcycle',
    5: 'Bus',
    7: 'Truck'
}

# Train2 classes (custom trained model)
TRAIN2_CLASSES = {
    0: 'Person',
    1: 'Car',
    2: 'Bus',
    3: 'Truck',
    4: 'Motorcycle'
}

TRAIN2_CLASS_IDS = {
    'person': 0,
    'car': 1,
    'bus': 2,
    'truck': 3,
    'motorcycle': 4
}

# =============================================================================
# DETECTION PARAMETERS
# =============================================================================
DEFAULT_CONFIDENCE = 0.55  # Tăng từ 0.5 → 0.55 (ít false positives hơn)
MIN_CONFIDENCE = 0.1
MAX_CONFIDENCE = 1.0

# FP16 Inference (requires GPU with Tensor Cores)
USE_FP16 = True  # Enable half-precision for 20-30% speedup

# NMS (Non-Maximum Suppression)
NMS_IOU = 0.5  # Giảm từ 0.6 → 0.5 (faster NMS)

# Inference size (fixed size for consistent performance)
INFERENCE_SIZE = 640  # Standard YOLO input size
# =============================================================================
# TRACKING PARAMETERS (DeepSORT)
# =============================================================================
TRACKER_MAX_AGE = 15  # Tăng từ 2 → 15 (giữ track lâu khi lag/skip frames)
TRACKER_N_INIT = 1    # Giữ 1 (confirm nhanh)
TRACKER_NMS_MAX_OVERLAP = 1.0

# =============================================================================
# VISUALIZATION
# =============================================================================
COLOR_PERSON = (0, 255, 0)      # Màu xanh lá cho Người
COLOR_VEHICLE = (0, 165, 255)   # Màu cam cho Xe
BBOX_THICKNESS = 1  # Giảm từ 2 → 1
TRAIL_LENGTH = 0  # TẮT trail để tăng FPS tối đa
TRAIL_THICKNESS = 1

# Font settings
FONT = 0  # cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.3  # Giảm thêm: 0.4 → 0.3
FONT_THICKNESS = 1
TEXT_COLOR = (255, 255, 255)

# Statistics panel
STATS_BG_COLOR = (0, 0, 0)
STATS_WIDTH = 250
STATS_HEIGHT_PER_ITEM = 30

# =============================================================================
# VIDEO PROCESSING
# =============================================================================
MAX_VIDEO_DURATION_SECONDS = 60  # Giới hạn demo
SUPPORTED_FORMATS = ['mp4', 'avi', 'mov', 'webm', 'mkv']

# Output video codec
OUTPUT_CODEC = 'mp4v'  # Codec cho file tạm
FINAL_CODEC = 'libx264'  # Codec cho file cuối (H.264)
PIXEL_FORMAT = 'yuv420p'  # Pixel format cho browser compatibility

# =============================================================================
# TEMP DIRECTORIES
# =============================================================================
TEMP_DIR = "temp"
OUTPUT_DIR = "output"
