"""
ML Benchmark Analyzer
Combines YOLO detection with traditional shape/color analysis for comparison
"""

import cv2
import numpy as np
import time
from typing import List, Dict, Optional

from config.settings import PreprocessingParams, AnalysisResult
from src.core import ImageProcessor, ShapeAnalyzer, ColorAnalyzer


class MLBenchmarkAnalyzer:
    """
    ML-based analyzer that combines YOLO detection with shape/color analysis.
    Provides comparable metrics to Quick and Deep analysis for benchmarking.
    """
    
    def __init__(self, yolo_model):
        """
        Initialize ML Benchmark Analyzer
        
        Args:
            yolo_model: Loaded YOLO model instance
        """
        self.yolo_model = yolo_model
        self.shape_analyzer = ShapeAnalyzer()
        self.color_analyzer = ColorAnalyzer()
        self.image_processor = ImageProcessor()
    
    def analyze(self, image: np.ndarray, params: PreprocessingParams = None, 
                confidence_threshold: float = 0.25) -> AnalysisResult:
        """
        Perform ML-based detection and extract detailed metrics.
        
        Args:
            image: Input RGB image
            params: Preprocessing parameters (for compatibility)
            confidence_threshold: Minimum confidence for detections
            
        Returns:
            AnalysisResult with ML detections and computed metrics
        """
        start_time = time.time()
        
        if image is None or image.size == 0:
            return AnalysisResult(
                features=[],
                mask=np.zeros((1, 1), dtype=np.uint8),
                processing_time=0,
                params=params or PreprocessingParams(),
                analysis_type='ml_benchmark'
            )
        
        # Detect background color
        bg_color = self.image_processor.detect_background_color(image)
        
        # Run YOLO detection
        results = self.yolo_model(image, conf=confidence_threshold, verbose=False)
        
        # Create combined mask from all detections
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        features = []
        
        for result in results:
            boxes = result.boxes
            for idx, box in enumerate(boxes):
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                
                # Skip low confidence detections
                if conf < confidence_threshold:
                    continue
                
                # Get class name (interpret as shape)
                class_name = result.names[cls] if cls < len(result.names) else f"Class_{cls}"
                
                # Create mask for this detection
                x, y, w, h = int(x1), int(y1), int(x2-x1), int(y2-y1)
                
                # Extract region of interest
                roi = image[y:y+h, x:x+w] if y+h <= image.shape[0] and x+w <= image.shape[1] else None
                
                if roi is None or roi.size == 0:
                    continue
                
                # Create particle mask using segmentation on ROI
                roi_mask, _ = self.image_processor.preprocess_segment(
                    roi, 
                    min_size=3,
                    method='adaptive'
                )
                
                # Place roi_mask in the full image mask
                if roi_mask.shape == (h, w):
                    mask[y:y+h, x:x+w] = np.maximum(mask[y:y+h, x:x+w], roi_mask)
                
                # Compute detailed shape metrics from the segmented region
                metrics = self.shape_analyzer.compute_shape_metrics_consistent(
                    roi_mask,
                    compute_solidity=True
                )
                
                if metrics is None:
                    # Fallback: use bounding box metrics
                    metrics = self._compute_bbox_metrics(x, y, w, h)
                else:
                    # Adjust centroid and bounding box to global coordinates
                    cx, cy = metrics['centroid']
                    metrics['centroid'] = (cx + x, cy + y)
                    bx, by, bw, bh = metrics['bounding_box']
                    metrics['bounding_box'] = (bx + x, by + y, bw, bh)
                
                # Classify shape based on metrics
                shape_class = self.shape_analyzer.classify_shape_unified(metrics)
                
                # Extract color from ROI
                roi_binary = np.zeros_like(roi_mask)
                roi_binary[roi_mask > 0] = 255
                color_name, h_val, s_val, v_val = self.color_analyzer.extract_color_from_region(
                    roi, roi_binary
                )
                
                # Build feature dictionary
                features.append({
                    'id': idx + 1,
                    'shape': shape_class,
                    'color': color_name,
                    'ml_class': class_name,
                    'ml_confidence': conf,
                    'area': metrics['area'],
                    'perimeter': metrics['perimeter'],
                    'circularity': metrics['circularity'],
                    'eccentricity': metrics['eccentricity'],
                    'aspect_ratio': metrics['aspect_ratio'],
                    'rectangularity': metrics['rectangularity'],
                    'solidity': metrics.get('solidity', 0),
                    'centroid': metrics['centroid'],
                    'bounding_box': metrics['bounding_box'],
                    'bbox_global': [x, y, w, h]
                })
        
        processing_time = time.time() - start_time
        
        return AnalysisResult(
            features=features,
            mask=mask,
            processing_time=processing_time,
            params=params or PreprocessingParams(),
            analysis_type='ml_benchmark',
            num_detections=len(features),
            background_color=bg_color
        )
    
    def _compute_bbox_metrics(self, x: int, y: int, w: int, h: int) -> Dict:
        """
        Compute basic metrics from bounding box when segmentation fails.
        
        Args:
            x, y, w, h: Bounding box coordinates
            
        Returns:
            Basic metrics dictionary
        """
        area = w * h
        perimeter = 2 * (w + h)
        circularity = (4.0 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
        aspect_ratio = max(w, h) / max(min(w, h), 1e-6)
        rectangularity = 1.0  # Perfect rectangle
        
        # Estimate eccentricity from aspect ratio
        # For rectangles: ecc ≈ sqrt(1 - (min/max)^2)
        ratio = min(w, h) / max(w, h)
        eccentricity = np.sqrt(max(0, 1 - ratio * ratio))
        
        return {
            'area': area,
            'perimeter': perimeter,
            'circularity': min(circularity, 1.0),
            'eccentricity': min(eccentricity, 1.0),
            'aspect_ratio': aspect_ratio,
            'rectangularity': rectangularity,
            'solidity': 1.0,  # Assume solid
            'centroid': (x + w/2, y + h/2),
            'bounding_box': (x, y, w, h)
        }
