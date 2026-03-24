"""
Analysis Module - Deep Analyzer
Detailed analysis with advanced metrics
"""

import cv2
import numpy as np
import time
from typing import List, Dict

from config.settings import PreprocessingParams, AnalysisResult
from src.core import ImageProcessor, ShapeAnalyzer, ColorAnalyzer


class DeepAnalyzer:
    """Perform deep analysis on microplastic images"""
    
    def __init__(self):
        """Initialize the deep analyzer"""
        self.image_processor = ImageProcessor()
        self.shape_analyzer = ShapeAnalyzer()
        self.color_analyzer = ColorAnalyzer()
    
    def analyze(self, image: np.ndarray, params: PreprocessingParams) -> AnalysisResult:
        """
        Perform deep analysis with detailed metrics.
        
        Args:
            image: Input RGB image
            params: Preprocessing parameters
            
        Returns:
            AnalysisResult with detailed features
        """
        start_time = time.time()
        
        if image is None or image.size == 0:
            return AnalysisResult(
                features=[],
                mask=np.zeros((1, 1), dtype=np.uint8),
                processing_time=0,
                params=params,
                analysis_type='deep'
            )
        
        # Preprocess image
        mask, thresh_val = self.image_processor.preprocess_segment(
            image,
            min_size=params.min_area,
            method=params.method
        )
        
        # Debug info
        mask_pixels = np.sum(mask > 0)
        print(f"DEBUG: Mask has {mask_pixels} white pixels, method={params.method}, min_area={params.min_area}")
        
        # Detect background
        bg_color = self.image_processor.detect_background_color(image)
        
        # Get image dimensions for boundary check
        h, w = mask.shape[:2]
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"DEBUG: Found {len(contours)} contours")
        
        # Extract detailed features (filter boundary objects for accurate metrics)
        features = []
        fiber_count = 0
        boundary_objects_removed = 0
        
        for idx, cnt in enumerate(contours):
            # === BOUNDARY CHECK: Skip objects actually touching image edges ===
            # Only filter if exclude_boundary is enabled (default: True)
            if params.exclude_boundary:
                # Get bounding box
                x, y, bw, bh = cv2.boundingRect(cnt)
                
                # Check if contour actually touches the image boundary (at edge, not near it)
                # Only remove objects that are truly at the edge (x=0, y=0, etc.)
                touches_boundary = (
                    x == 0 or              # Touches left edge
                    y == 0 or              # Touches top edge
                    (x + bw) >= w or       # Touches right edge
                    (y + bh) >= h          # Touches bottom edge
                )
                
                if touches_boundary:
                    boundary_objects_removed += 1
                    continue  # Skip - incomplete shape, perimeter cut off
            
            # Compute shape metrics with solidity (now on complete objects only)
            metrics = self.shape_analyzer.compute_shape_metrics_consistent(
                cv2.drawContours(np.zeros_like(mask), [cnt], -1, 255, -1),
                compute_solidity=True
            )
            
            if metrics is None:
                continue
            
            # Classify shape
            shape_class = self.shape_analyzer.classify_shape_unified(metrics)
            
            if shape_class == 'Fiber/Filament':  # Updated to grouped name
                fiber_count += 1
            
            # Extract color with full HSV
            particle_mask = np.zeros_like(mask)
            cv2.drawContours(particle_mask, [cnt], -1, 255, -1)
            color_name, hue, sat, val = self.color_analyzer.extract_color_from_region(image, particle_mask)
            
            features.append({
                'id': idx + 1,
                'shape': shape_class,
                'color': color_name,
                'area': metrics['area'],
                'perimeter': metrics['perimeter'],
                'circularity': metrics['circularity'],
                'eccentricity': metrics['eccentricity'],
                'aspect_ratio': metrics['aspect_ratio'],
                'rectangularity': metrics['rectangularity'],
                'solidity': metrics.get('solidity', 0),
                'centroid': metrics['centroid'],
                'bounding_box': metrics['bounding_box'],
                'Contour': cnt,
                'hsv': (hue, sat, val)
            })
        
        processing_time = time.time() - start_time
        avg_time_per_object = processing_time / len(features) if features else 0
        
        if boundary_objects_removed > 0:
            print(f"[Deep Analysis] Removed {boundary_objects_removed} boundary objects (touching edges)")
        
        return AnalysisResult(
            features=features,
            mask=mask,
            processing_time=processing_time,
            params=params,
            analysis_type='deep',
            num_detections=len(features),
            image_dimensions=image.shape[:2],
            background_color=bg_color,
            fiber_count=fiber_count,
            avg_processing_per_object=avg_time_per_object,
            boundary_objects_removed=boundary_objects_removed
        )
