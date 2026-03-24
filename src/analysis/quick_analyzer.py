"""
Analysis Module - Quick Analyzer
Fast analysis for real-time processing
"""

import cv2
import numpy as np
import time
from typing import List, Dict

from config.settings import PreprocessingParams, AnalysisResult
from src.core import ImageProcessor, ShapeAnalyzer, ColorAnalyzer


class QuickAnalyzer:
    """Perform quick analysis on microplastic images"""
    
    def __init__(self):
        """Initialize the quick analyzer"""
        self.image_processor = ImageProcessor()
        self.shape_analyzer = ShapeAnalyzer()
        self.color_analyzer = ColorAnalyzer()
    
    def analyze(self, image: np.ndarray, params: PreprocessingParams) -> AnalysisResult:
        """
        Perform quick analysis on an image.
        
        Args:
            image: Input RGB image
            params: Preprocessing parameters
            
        Returns:
            AnalysisResult with detected features
        """
        start_time = time.time()
        
        if image is None or image.size == 0:
            return AnalysisResult(
                features=[],
                mask=np.zeros((1, 1), dtype=np.uint8),
                processing_time=0,
                params=params,
                analysis_type='quick'
            )
        
        # Preprocess image
        mask, thresh_val = self.image_processor.preprocess_segment(
            image, 
            min_size=params.min_area,
            method=params.method
        )
        
        # Detect background
        bg_color = self.image_processor.detect_background_color(image)
        
        # Get image dimensions for boundary check
        h, w = mask.shape[:2]
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Extract features (filter boundary objects for accurate shape metrics)
        features = []
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
            
            # Compute shape metrics (now on complete objects only)
            metrics = self.shape_analyzer.compute_shape_metrics_consistent(
                cv2.drawContours(np.zeros_like(mask), [cnt], -1, 255, -1),
                compute_solidity=False
            )
            
            if metrics is None:
                continue
            
            # Classify shape
            shape_class = self.shape_analyzer.classify_shape_unified(metrics)
            
            # Extract color
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
                'centroid': metrics['centroid'],
                'bounding_box': metrics['bounding_box'],
                'Contour': cnt
            })
        
        processing_time = time.time() - start_time
        
        if boundary_objects_removed > 0:
            print(f"[Quick Analysis] Removed {boundary_objects_removed} boundary objects (touching edges)")
        
        return AnalysisResult(
            features=features,
            mask=mask,
            processing_time=processing_time,
            params=params,
            analysis_type='quick',
            num_detections=len(features),
            image_dimensions=image.shape[:2],
            background_color=bg_color,
            boundary_objects_removed=boundary_objects_removed
        )
