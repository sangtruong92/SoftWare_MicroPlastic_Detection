"""
Configuration settings for Microplastic Analyzer
Contains dataclass definitions for various parameter sets
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
import numpy as np


@dataclass
class SyntheticImageParams:
    """Parameters for synthetic image generation"""
    num_particles: int = 15
    shape_type: str = 'Mixed'
    color_type: str = 'Fluorescent'
    enable_blur: bool = True
    blur_kernel: int = 10
    enable_glow: bool = True
    glow_intensity: float = 0.4
    glow_sigma_min: float = 2.5
    glow_sigma_max: float = 6.0
    background_noise_min: float = 0.0
    background_noise_max: float = 0.02
    image_width: int = 2972
    image_height: int = 1980
    background_brightness: int = 0  # 0=dark (fluorescent), 255=white (brightfield)
    particle_brightness: int = 255  # Particle intensity (0=dark, 255=full brightness)


@dataclass
class PreprocessingParams:
    """Parameters for preprocessing"""
    method: str = 'advanced'
    min_area: int = 3  # Reduced from 10 to catch smaller particles
    max_area: int = 10000
    blur: int = 5
    distance_thresh: float = 0.5
    marker_size: int = 3
    brightness_threshold: int = 0  # Minimum mean brightness (0=disabled, 20-80 typical)
    min_blur_score: float = 0.0  # Minimum Laplacian variance (0=disabled, 50-300 typical)
    adaptive_c_value: int = 2  # C parameter for adaptive thresholding (lower = more sensitive)
    exclude_boundary: bool = True  # Exclude objects touching image edges (for accurate shape metrics)


@dataclass
class AnalysisResult:
    """Results from analysis"""
    features: List[Dict]
    mask: np.ndarray
    processing_time: float
    params: PreprocessingParams
    analysis_type: str
    num_detections: int = 0
    image_dimensions: Tuple[int, int] = (0, 0)
    background_color: str = 'unknown'
    fiber_count: int = 0
    avg_processing_per_object: float = 0.0
    boundary_objects_removed: int = 0  # Count of objects touching image edges (excluded from analysis)


@dataclass
class MLModelConfig:
    """Configuration for machine learning models"""
    model_type: str = 'YOLOv8'
    model_path: Optional[str] = None
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    device: str = 'cpu'  # 'cpu', 'cuda', or 'mps'
    image_size: int = 640


@dataclass
class TrainingConfig:
    """Configuration for model training"""
    epochs: int = 100
    batch_size: int = 16
    image_size: int = 640
    learning_rate: float = 0.01
    patience: int = 50
    device: str = 'cpu'
    workers: int = 8
    augmentation: bool = True
    save_period: int = 10
