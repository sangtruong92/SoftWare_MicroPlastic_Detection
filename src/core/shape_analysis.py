"""
Shape Analysis Module
Contains algorithms for shape classification and metric computation
"""

import cv2
import numpy as np
import math
from typing import Dict, Optional

from config.constants import SHAPE_THRESHOLDS, SHAPE_EQUIVALENCE_GROUPS, SHAPE_GROUP_MAPPING


class ShapeAnalyzer:
    """Shape classification and analysis algorithms"""
    
    @staticmethod
    def classify_shape_unified(metrics: Dict) -> str:
        """
        Unified shape classification using consistent thresholds based on literature.
        Returns grouped categories for simplified classification.
        
        Grouped categories:
        - Microbead/Pellet: Spherical particles (C > 0.6, E < 0.65, AR < 1.8)
        - Fiber/Filament: Elongated structures (AR > 2.5, E ≥ 0.7 OR C < 0.55)
        - Fragment: Irregular pieces with moderate circularity
        - Irregular: Highly irregular shapes
        
        Args:
            metrics: dict containing:
                - circularity: float (0-1)
                - eccentricity: float (0-1)
                - aspect_ratio: float (≥1)
                - area: float (pixels)
                - rectangularity: float (0-1, optional)
                - solidity: float (0-1, optional)
        
        Returns:
            str: Grouped shape category
        """
        circ = metrics.get('circularity', 0)
        ecc = metrics.get('eccentricity', 0)
        aspect = metrics.get('aspect_ratio', 1.0)
        area = metrics.get('area', 0)
        rect = metrics.get('rectangularity', 0)
        solid = metrics.get('solidity', 1.0)
        
        # 1. Check for Filament first (most stringent: AR > 5.0, very elongated)
        if (aspect >= SHAPE_THRESHOLDS['Filament']['aspect_ratio_min'] and
            ecc >= SHAPE_THRESHOLDS['Filament']['eccentricity_min'] and
            circ <= SHAPE_THRESHOLDS['Filament']['circularity_max']):
            return "Fiber/Filament"
        
        # 2. Check for Fiber (elongated: AR > 2.5 with high eccentricity OR low circularity)
        if aspect >= SHAPE_THRESHOLDS['Fiber']['aspect_ratio_min']:
            # Relaxed criteria: High aspect ratio + (high eccentricity OR low circularity)
            if ecc >= SHAPE_THRESHOLDS['Fiber']['eccentricity_min'] or circ <= SHAPE_THRESHOLDS['Fiber']['circularity_max']:
                return "Fiber/Filament"
        
        # 3. IRREGULAR CHECK: Very high eccentricity indicates non-spherical shape
        # Only classify as Irregular if eccentricity is significantly high
        if ecc >= 0.68 and aspect < 2.5:  # Very high asymmetry, not elongated
            if circ < 0.85:  # Not nearly circular
                return "Irregular"
        
        # 4. Check for Fragment FIRST (before Microbead/Pellet)
        # Fragment has moderate circularity and lower solidity due to irregular edges
        # This prevents smooth synthetic Fragments from being misclassified
        if (circ >= SHAPE_THRESHOLDS['Fragment']['circularity_min'] and
            circ <= SHAPE_THRESHOLDS['Fragment']['circularity_max'] and
            aspect <= SHAPE_THRESHOLDS['Fragment']['aspect_ratio_max']):
            return "Fragment"
        
        # 5. Check for spherical particles (Microbead/Pellet)
        # Key criteria: VERY High circularity (C >= 0.85) + moderate eccentricity + very high solidity
        if circ >= 0.85:  # Increased from 0.82 to separate from Fragment range
            if ecc <= 0.55 and solid >= 0.96:  # Stricter: ecc from 0.60, solid from 0.95
                return "Microbead/Pellet"
        
        # 6. Check for lower circularity Irregular (standard check)
        if circ <= 0.55 or solid < 0.80:
            if aspect <= SHAPE_THRESHOLDS['Irregular']['aspect_ratio_max']:
                return "Irregular"
        
        # 7. Check for remaining Irregular (low circularity, variable shape)
        if (circ <= SHAPE_THRESHOLDS['Irregular']['circularity_max'] and
            aspect <= SHAPE_THRESHOLDS['Irregular']['aspect_ratio_max']):
            return "Irregular"
        
        # 8. Default: Fragment for anything that doesn't match above
        return "Fragment"
    
    @staticmethod
    def compute_shape_metrics_consistent(mask, compute_solidity=True) -> Optional[Dict]:
        """
        Consistent shape metric computation for both quick and deep analysis.
        
        Args:
            mask: Binary mask of the particle
            compute_solidity: Whether to compute solidity metric
            
        Returns:
            dict: Shape metrics or None if computation fails
        """
        if mask is None or mask.size == 0 or not np.any(mask):
            return None
        
        # Ensure binary mask
        mask_binary = (mask > 0).astype(np.uint8) * 255
        
        # Analyze if it's fiber-like and apply erosion if needed
        contours_temp, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours_temp:
            cnt_temp = max(contours_temp, key=cv2.contourArea)
            rect_temp = cv2.minAreaRect(cnt_temp)
            width_temp = rect_temp[1][0]
            height_temp = rect_temp[1][1]
            aspect_temp = max(width_temp, height_temp) / (min(width_temp, height_temp) + 1e-6)
            
            # If fiber-like, apply gentle erosion
            if aspect_temp > 3.0:
                kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
                mask_binary = cv2.erode(mask_binary, kernel_erode, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        
        # Use the largest contour
        cnt = max(contours, key=cv2.contourArea)
        
        # Simplify contour
        arc_len = cv2.arcLength(cnt, True)
        epsilon = max(0.5, 0.005 * arc_len)
        simplified = cv2.approxPolyDP(cnt, epsilon, True)
        
        # Basic metrics
        area = cv2.contourArea(simplified)
        perimeter = cv2.arcLength(simplified, True)
        
        # Circularity
        circularity = (4.0 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.0
        
        # Bounding box
        x, y, w, h = cv2.boundingRect(simplified)
        bounding_area = w * h
        rectangularity = area / bounding_area if bounding_area > 0 else 0.0
        aspect_ratio = max(w, h) / max(min(w, h), 1e-6)
        
        # Moments for centroid and eccentricity
        moments = cv2.moments(simplified)
        
        # Centroid
        centroid = (x + w / 2.0, y + h / 2.0)
        if moments['m00'] > 0:
            centroid = (moments['m10'] / moments['m00'], moments['m01'] / moments['m00'])
        
        # Eccentricity calculation
        eccentricity = 0.0
        if moments['m00'] > 0:
            mu20 = moments['mu20'] / moments['m00']
            mu02 = moments['mu02'] / moments['m00']
            mu11 = moments['mu11'] / moments['m00']
            
            eccentricity = ShapeAnalyzer.compute_eccentricity_safe(mu20, mu02, mu11)
        
        # Build metrics dictionary
        metrics = {
            'area': area,
            'perimeter': perimeter,
            'circularity': min(circularity, 1.0),
            'eccentricity': min(eccentricity, 1.0),
            'rectangularity': rectangularity,
            'aspect_ratio': aspect_ratio,
            'centroid': centroid,
            'bounding_box': (x, y, w, h),
            'contour': simplified
        }
        
        # Compute solidity if requested
        if compute_solidity:
            hull = cv2.convexHull(simplified)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0.0
            metrics['solidity'] = solidity
        
        # Try ellipse fitting for better eccentricity
        if len(simplified) >= 5:
            try:
                ellipse = cv2.fitEllipse(simplified)
                (_, _), (ma, mi), angle = ellipse
                ellipse_aspect = max(ma, mi) / max(min(ma, mi), 1e-6)
                if ellipse_aspect > aspect_ratio:
                    metrics['aspect_ratio'] = ellipse_aspect
                    
                # Ellipse eccentricity
                if max(ma, mi) > 0:
                    axis_ratio = min(ma, mi) / max(ma, mi)
                    ellipse_ecc = math.sqrt(max(0, 1 - axis_ratio * axis_ratio))
                    if ellipse_ecc > eccentricity:
                        metrics['eccentricity'] = min(ellipse_ecc, 1.0)
            except Exception:
                pass
        
        return metrics
    
    @staticmethod
    def compute_eccentricity_safe(mu20, mu02, mu11):
        """Safe eccentricity calculation that avoids math domain errors."""
        cov_matrix = np.array([[mu20, mu11], [mu11, mu02]])
        
        try:
            eigenvalues = np.linalg.eigvals(cov_matrix)
            eigenvalues = np.sort(eigenvalues)
            
            # Ensure eigenvalues are positive
            lambda_min = max(eigenvalues[0], 1e-10)
            lambda_max = max(eigenvalues[1], 1e-10)
            
            # Eccentricity = sqrt(1 - λ_min/λ_max)
            ratio = lambda_min / lambda_max
            ratio = max(0, min(1, ratio))
            eccentricity = math.sqrt(max(0, 1 - ratio))
            
            return min(eccentricity, 1.0)
        except:
            return 0.0
    
    @staticmethod
    def _shape_group_label(shape_name):
        """Convert shape name to its group label."""
        if not shape_name:
            return "Unknown"
        
        for group in SHAPE_EQUIVALENCE_GROUPS:
            if shape_name in group:
                return sorted(group)[0]
        
        return shape_name
    
    @staticmethod
    def enhanced_shape_class(circ, ecc, rectangularity=None, area=None, 
                            aspect_ratio=None, perimeter=None, solidity=None):
        """Enhanced shape classification matching synthetic generation after preprocessing effects.
        
        Key insight: Preprocessing (blur, morphology) significantly alters shapes:
        - Perfect rectangles (Film) become rounded → lose rectangularity
        - Spiky shapes (Irregular) get smoothed → look like mid-range fragments
        - Need to rely more on circularity + eccentricity + aspect_ratio combinations
        """
        
        # === FIBER/FILAMENT DETECTION (Priority 1: Most distinctive) ===
        is_fiber_candidate = False
        
        # Fiber requires BOTH high eccentricity AND elongated shape
        if aspect_ratio is not None and aspect_ratio > 3.0:  # Definitely elongated
            is_fiber_candidate = True
        elif aspect_ratio is not None and aspect_ratio > 2.5:
            # Moderately elongated - require high eccentricity too
            if ecc is not None and ecc > 0.85:
                is_fiber_candidate = True
        
        # Low solidity + elongated = fiber (broken strands)
        if solidity is not None and solidity < 0.75 and aspect_ratio is not None and aspect_ratio > 2.0:
            is_fiber_candidate = True
        
        # Fiber vs Filament classification
        if is_fiber_candidate:
            if aspect_ratio is not None and aspect_ratio > 5.0:
                return "Filament"
            else:
                return "Fiber"
        
        # === IRREGULAR SHAPES (Priority 2: Before spherical) ===
        # After preprocessing, spiky shapes smooth out but retain characteristics:
        # - True spheres: C > 0.90 and E < 0.35
        # - Irregulars after glow: C = 0.55-0.85 with higher E (> 0.40)
        is_irregular = False
        
        if circ is not None and circ < 0.55:  # Classic low circularity
            is_irregular = True
        elif circ is not None and 0.55 <= circ < 0.90:  # Mid-high range
            # Particles with high eccentricity are not true spheres
            if ecc is not None and ecc > 0.55:  # High asymmetry
                is_irregular = True
            elif solidity is not None and solidity < 0.88:  # Complex shape
                is_irregular = True
        
        if is_irregular:
            if aspect_ratio is not None and aspect_ratio < 2.5:  # Not elongated
                return "Irregular"
        
        # === SPHERICAL PARTICLES (Priority 3: Microbead/Pellet) ===
        # Very round shapes with high circularity AND low eccentricity AND high solidity
        if circ is not None and circ >= 0.85:
            if ecc is not None and ecc < 0.55:  # Low-moderate asymmetry
                if solidity is not None and solidity >= 0.92:
                    if area is not None and area < 800:
                        return "Microbead"
                    else:
                        return "Pellet"
        
        # === IRREGULAR SHAPES (Priority 4: Before Fragment) ===
        # Catch remaining low-circularity shapes
        if circ is not None and circ < 0.55:
            if aspect_ratio is not None and aspect_ratio < 2.0:
                return "Irregular"
        
        # === FRAGMENT CLASSIFICATION ===
        # Mid-range circularity, moderate properties
        if circ is not None and 0.45 <= circ < 0.75:
            if aspect_ratio is not None and aspect_ratio < 2.5:
                return "Fragment"
        
        # === DEFAULT: Assign based on dominant feature ===
        if circ is not None:
            if circ < 0.55:
                return "Irregular"
            elif circ < 0.75:
                return "Fragment"
            else:
                # High circ but didn't match strict Microbead criteria
                # Could be Irregular with smooth edges or borderline sphere
                if ecc is not None and ecc > 0.50:
                    return "Irregular"  # High asymmetry = not true sphere
                return "Microbead"
        
        return "Irregular"  # Ultimate fallback
