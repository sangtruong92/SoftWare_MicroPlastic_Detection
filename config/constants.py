"""
Configuration constants for Microplastic Analyzer
Contains shape thresholds, categories, and other constant values
"""

# ============================================================================
# SHAPE CLASSIFICATION THRESHOLDS
# ============================================================================

# Based on research literature (morphometric parameters for microplastic classification)
# Reference: AR = Aspect Ratio, C = Circularity, E = Eccentricity
# UPDATED Feb 4, 2026: Relaxed thresholds to improve detection accuracy
SHAPE_THRESHOLDS = {
    'Microbead': {
        'circularity_min': 0.60,  # Relaxed to catch more spherical particles
        'circularity_max': 1.1,
        'eccentricity_max': 0.60,  # Relaxed from 0.50 to allow slight elongation
        'area_max': 800,
        'aspect_ratio_min': 1.0,
        'aspect_ratio_max': 1.5  # Relaxed from 1.3
    },
    'Pellet': {
        'circularity_min': 0.60,  # Relaxed to catch more spherical particles
        'circularity_max': 1.1,
        'eccentricity_max': 0.65,  # Relaxed from 0.50 - key fix
        'area_min': 800,
        'aspect_ratio_min': 1.0,
        'aspect_ratio_max': 1.8  # Relaxed from 1.6
    },
    'Fiber': {
        'circularity_max': 0.55,  # Relaxed from 0.30 - catch more fibers
        'eccentricity_min': 0.70,  # Relaxed from 0.75
        'aspect_ratio_min': 2.5,
        'solidity_max': 0.85,
        'rectangularity_max': 0.6
    },
    'Filament': {
        'circularity_max': 0.30,
        'eccentricity_min': 0.90,
        'aspect_ratio_min': 5.0,
        'aspect_ratio_max': 100.0,
        'solidity_min': 0.5
    },
    'Fragment': {
        'circularity_min': 0.38,  # Updated from 0.30
        'circularity_max': 0.72,  # Updated from 0.60
        'eccentricity_min': 0.20,  # Relaxed from 0.30
        'eccentricity_max': 0.75,  # Relaxed from 0.70
        'aspect_ratio_min': 1.0,  # Relaxed from 1.2
        'aspect_ratio_max': 3.0,  # Relaxed from 2.5
        'solidity_min': 0.5
    },
    'Irregular': {
        'circularity_max': 0.50,  # Relaxed from 0.40
        'eccentricity_max': 0.80,  # Relaxed from 0.70
        'aspect_ratio_max': 3.0,  # Relaxed from 2.0
        'solidity_max': 0.7
    }
}

# ============================================================================
# SHAPE CATEGORIES AND GROUPING
# ============================================================================

# All possible shape categories (grouped for simplified classification)
SHAPE_CATEGORIES = [
    'Microbead/Pellet',  # Spherical particles (small and large)
    'Fiber/Filament',    # Elongated thread-like structures
    'Fragment',          # Irregular broken pieces
    'Irregular'          # Highly irregular shapes
]

# Legacy individual categories (for internal thresholds)
INDIVIDUAL_CATEGORIES = [
    'Microbead', 'Pellet', 'Fiber', 'Filament', 
    'Fragment', 'Irregular'
]

# Shape equivalence groups for classification mapping
SHAPE_EQUIVALENCE_GROUPS = [
    {"Microbead", "Pellet"},     # → "Microbead/Pellet"
    {"Fiber", "Filament"},       # → "Fiber/Filament"
    {"Fragment", "Irregular"},   # → "Fragment" or "Irregular"
]

# Mapping from individual to grouped names
SHAPE_GROUP_MAPPING = {
    'Microbead': 'Microbead/Pellet',
    'Pellet': 'Microbead/Pellet',
    'Fiber': 'Fiber/Filament',
    'Filament': 'Fiber/Filament',
    'Fragment': 'Fragment',
    'Irregular': 'Irregular'
}

# ============================================================================
# COLOR CLASSIFICATION
# ============================================================================

COLOR_RANGES = {
    'Red': {'h_min': 0, 'h_max': 10, 's_min': 100},
    'Orange': {'h_min': 11, 'h_max': 25, 's_min': 100},
    'Yellow': {'h_min': 26, 'h_max': 34, 's_min': 100},
    'Green': {'h_min': 35, 'h_max': 85, 's_min': 40},
    'Cyan': {'h_min': 86, 'h_max': 95, 's_min': 100},
    'Blue': {'h_min': 96, 'h_max': 130, 's_min': 50},
    'Purple': {'h_min': 131, 'h_max': 155, 's_min': 40},
    'Magenta': {'h_min': 156, 'h_max': 170, 's_min': 100},
    'Pink': {'h_min': 171, 'h_max': 180, 's_min': 40},
    'White': {'s_max': 30, 'v_min': 200},
    'Gray': {'s_max': 30, 'v_min': 50, 'v_max': 200},
    'Black': {'v_max': 50}
}

# ============================================================================
# IMAGE PROCESSING PARAMETERS
# ============================================================================

# Default preprocessing parameters
DEFAULT_MIN_SIZE = 20
DEFAULT_MAX_SIZE = 10000
DEFAULT_BLUR_KERNEL = 15
DEFAULT_DISTANCE_THRESH = 0.5
DEFAULT_MARKER_SIZE = 3

# Background detection thresholds
BG_WHITE_THRESHOLD = 200
BG_BLACK_THRESHOLD = 50
BG_STD_THRESHOLD = 60

# ============================================================================
# YOLO PARAMETERS
# ============================================================================

YOLO_CLASS_NAMES = [
    'Microbead', 'Pellet', 'Fiber', 'Filament', 
    'Film', 'Fragment', 'Irregular', 'Unknown'
]

YOLO_CLASS_MAPPING = {name: idx for idx, name in enumerate(YOLO_CLASS_NAMES)}

# ============================================================================
# FILE PATHS AND DIRECTORIES
# ============================================================================

DEFAULT_MODEL_DIR = 'models'
DEFAULT_DATA_DIR = 'data'
DEFAULT_RESULTS_DIR = 'results'
DEFAULT_TEMP_DIR = 'temp'

# ============================================================================
# GUI PARAMETERS
# ============================================================================

DEFAULT_WINDOW_SIZE = (1400, 900)
DEFAULT_IMAGE_DISPLAY_SIZE = (600, 400)
