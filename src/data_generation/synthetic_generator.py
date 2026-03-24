"""
Data Generation Module
Generates synthetic microplastic images for training and testing
"""

import cv2
import numpy as np
import random
from typing import List, Dict, Tuple

from config.settings import SyntheticImageParams
from src.core.color_analysis import ColorAnalyzer


class SyntheticImageGenerator:
    """Generate synthetic microplastic images with ground truth"""
    
    # Fluorescent wavelengths palette - 4 natural fluorescent colors
    _FLUORESCENT_WAVELENGTHS = [
        ("Red", 630),
        ("Green", 525),
        ("Blue", 450),
        ("Yellow", 565),
    ]
    
    # Natural colors palette
    _NATURAL_COLORS = [
        ("White", (255, 255, 255)),
        ("Light Blue", (173, 216, 230)),
        ("Blue", (0, 0, 255)),
        ("Green", (0, 255, 0)),
        ("Red", (255, 0, 0)),
        ("Yellow", (255, 255, 0)),
        ("Orange", (255, 165, 0)),
        ("Brown", (165, 42, 42)),
        ("Black", (0, 0, 0)),
        ("Gray", (128, 128, 128)),
        ("Pink", (255, 192, 203)),
        ("Purple", (128, 0, 128)),
    ]
    
    def __init__(self):
        """Initialize the synthetic image generator"""
        self._build_palettes()
    
    def _build_palettes(self):
        """Build color palettes"""
        self.fluorescent_palette = []
        for label, wavelength in self._FLUORESCENT_WAVELENGTHS:
            rgb = ColorAnalyzer.wavelength_to_rgb(wavelength)
            self.fluorescent_palette.append((label, wavelength, rgb))
    
    def generate(self, params: SyntheticImageParams) -> Tuple[np.ndarray, List[Dict]]:
        """
        Generate synthetic image with ground truth.
        
        Args:
            params: SyntheticImageParams with generation settings
            
        Returns:
            Tuple of (image_array, ground_truth_info)
        """
        height, width = params.image_height, params.image_width
        
        # Create background based on brightness setting
        # 0 = dark (fluorescent), 255 = white (brightfield)
        bg_brightness = getattr(params, 'background_brightness', 0)
        if bg_brightness > 10:  # Brightfield mode
            base_value = bg_brightness / 255.0
            # Add slight variation for realism
            noise_range = 0.02
            base = np.full((height, width, 3), base_value, dtype=np.float32)
            noise = np.random.uniform(-noise_range, noise_range, (height, width, 1))
            base = np.clip(base + noise, 0, 1)
        else:  # Fluorescent mode (dark background)
            base_darkness = random.uniform(0.0, 0.01)
            base = np.full((height, width, 3), base_darkness, dtype=np.float32)
        ground_truth = []
        
        # Determine shapes to include (support both individual and grouped names)
        if params.shape_type == 'Mixed':
            shape_catalog = ['Microbead', 'Pellet', 'Fiber', 'Filament', 'Fragment', 'Irregular']
        elif 'Microbead/Pellet' in params.shape_type:
            shape_catalog = ['Microbead', 'Pellet']
        elif 'Fiber/Filament' in params.shape_type:
            shape_catalog = ['Fiber', 'Filament']
        elif 'only' in params.shape_type:
            shape_name = params.shape_type.replace(' only', '')
            # Map grouped names back to individual for generation
            if shape_name == 'Microbead/Pellet':
                shape_catalog = ['Microbead', 'Pellet']
            elif shape_name == 'Fiber/Filament':
                shape_catalog = ['Fiber', 'Filament']
            else:
                shape_catalog = [shape_name]
        else:
            shape_catalog = ['Microbead', 'Pellet', 'Fiber', 'Filament', 'Fragment', 'Irregular']
        
        # Increased margin to account for blur/glow effects spreading to edges
        # This prevents particles from appearing to touch image boundaries after effects
        min_margin = 50
        
        for i in range(params.num_particles):
            shape_label = random.choice(shape_catalog)
            
            # Select color based on color_type
            if params.color_type == 'Fluorescent':
                color_label, wavelength, rgb = random.choice(self.fluorescent_palette)
            elif params.color_type == 'Natural':
                color_label, rgb = random.choice(self._NATURAL_COLORS)
                wavelength = None
            else:  # Mixed Colors
                if random.random() > 0.5:
                    color_label, wavelength, rgb = random.choice(self.fluorescent_palette)
                else:
                    color_label, rgb = random.choice(self._NATURAL_COLORS)
                    wavelength = None
            
            # Enhanced intensity variation for fluorescent realism
            # Apply user-defined particle brightness scale
            particle_brightness_scale = getattr(params, 'particle_brightness', 255) / 255.0
            if params.color_type == 'Fluorescent' or (params.color_type == 'Mixed Colors' and wavelength):
                intensity_scale = random.uniform(0.3, 1.5) * particle_brightness_scale
            else:
                intensity_scale = random.uniform(0.75, 1.2) * particle_brightness_scale
            
            rgb_scaled = np.clip(np.array(rgb, dtype=np.float32) * intensity_scale, 0, 255)
            bgr_scaled = rgb_scaled[[2, 1, 0]]
            
            # Generate particle shape
            mask, cx, cy, size_hint, area = self._generate_particle_shape(
                shape_label, height, width, min_margin
            )
            
            # Calculate circularity from the mask for ground truth
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            circularity = 0.0
            if contours:
                cnt = max(contours, key=cv2.contourArea)
                perimeter = cv2.arcLength(cnt, True)
                if perimeter > 0:
                    circularity = (4.0 * np.pi * area) / (perimeter ** 2)
                    circularity = min(circularity, 1.0)
            
            # Apply effects
            mask_float = self._apply_effects(mask, params)
            
            # Add particle to base image
            # Brightfield: blend particle color onto white background
            # Fluorescent: add bright color onto dark background
            bg_brightness = getattr(params, 'background_brightness', 0)
            if bg_brightness > 10:  # Brightfield mode
                # Blend: base * (1 - mask) + particle_color * mask
                for channel in range(3):
                    base[:, :, channel] = (
                        base[:, :, channel] * (1 - mask_float) +
                        mask_float * (bgr_scaled[channel] / 255.0)
                    )
            else:  # Fluorescent mode (additive)
                for channel in range(3):
                    base[:, :, channel] = np.maximum(
                        base[:, :, channel],
                        mask_float * (bgr_scaled[channel] / 255.0)
                    )
            
            # Store ground truth with all metrics needed for statistics
            ground_truth.append({
                'id': i + 1,
                'position': (cx, cy),
                'size': size_hint,
                'area': area,
                'circularity': circularity,
                'shape': shape_label,
                'color': tuple(int(x) for x in bgr_scaled),
                'color_label': color_label,
                'wavelength_nm': wavelength
            })
        
        # Add realistic noise and artifacts
        base = self._add_noise_and_artifacts(base, width, height)
        
        # Apply optical blur
        base = self._apply_optical_blur(base)
        
        # Convert to uint8 RGB
        img_bgr = (base * 255).astype(np.uint8)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        return img_rgb, ground_truth
    
    def _generate_particle_shape(self, shape_label: str, height: int, width: int, 
                                 min_margin: int) -> Tuple[np.ndarray, int, int, int, float]:
        """
        Generate a particle shape mask based on literature morphometric parameters.
        
        Reference criteria:
        - Fiber: AR > 3.0, E ≥ 0.7 OR C ≤ 0.3 (relaxed from strict E > 0.9, C < 0.2)
        - Filament: AR > 5.0, C < 0.15, E > 0.95
        - Microbead: AR 1.0-1.2, C > 0.9, E < 0.2
        - Pellet: AR 1.0-1.5, C > 0.8, E < 0.4
        - Fragment: AR 1.0-3.0, C 0.38-0.72 (generation targets 0.4-0.7), E 0.20-0.75
        - Film: AR variable, C < 0.5, E variable
        """
        mask = np.zeros((height, width), dtype=np.uint8)
        cx = random.randint(min_margin, width - min_margin)
        cy = random.randint(min_margin, height - min_margin)
        size_hint = 10
        area = 0
        
        if shape_label == 'Microbead':
            # AR 1.0-1.2, C > 0.9 - nearly perfect circle
            # 10-24μm @ 2.2μm/pixel = 4.5-11 pixels diameter = radius 2-5.5
            radius = random.randint(2, 5)
            cv2.circle(mask, (cx, cy), radius, 255, -1)
            size_hint = radius
            area = np.pi * radius * radius
        
        elif shape_label == 'Pellet':
            # AR 1.0-1.5, C > 0.8 - slightly oval to circular
            # 30-100μm @ 2.2μm/pixel = 14-45 pixels diameter = radius 7-23
            radius = random.randint(7, 23)
            # Create slightly elliptical shape
            axes = (radius, int(radius * random.uniform(0.85, 1.0)))
            cv2.ellipse(mask, (cx, cy), axes, random.uniform(0, 360), 0, 360, 255, -1)
            size_hint = radius
            area = np.pi * axes[0] * axes[1]
        
        elif shape_label == 'Fiber':
            # AR > 3.0 (aim for 3-5), C < 0.2, E > 0.9 - very elongated
            # 20-100μm length @ 2.2μm/pixel = 9-45 pixels
            length = random.randint(15, 45)
            width_f = random.randint(2, 6)  # Keep thin for high AR, 4.4-13.2μm width
            angle = random.uniform(0, 180)
            margin = max(length, width_f) // 2 + 50  # Increased for glow effects
            cx = random.randint(margin, width - margin)
            cy = random.randint(margin, height - margin)
            rect = ((cx, cy), (length, width_f), angle)
            box = cv2.boxPoints(rect).astype(np.int32)
            cv2.fillPoly(mask, [box], 255)
            # Add slight irregularity to edges
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            if random.random() > 0.5:
                mask = cv2.erode(mask, kernel, iterations=1)
            size_hint = length / 2
            area = length * width_f
        
        elif shape_label == 'Filament':
            # AR > 5.0 (aim for 6-12), C < 0.15, E > 0.95 - extremely elongated
            # 30-100μm length @ 2.2μm/pixel = 14-45 pixels
            length = random.randint(25, 45)
            width_f = random.randint(2, 4)  # Very thin for very high AR, 4.4-8.8μm width
            angle = random.uniform(0, 180)
            margin = max(length, width_f) // 2 + 50  # Increased for glow effects
            cx = random.randint(margin, width - margin)
            cy = random.randint(margin, height - margin)
            rect = ((cx, cy), (length, width_f), angle)
            box = cv2.boxPoints(rect).astype(np.int32)
            cv2.fillPoly(mask, [box], 255)
            size_hint = length / 2
            area = length * width_f
        
        elif shape_label == 'Film':
            # AR variable, C < 0.5, rectangularity ≥ 0.85 - thin sheet-like, rectangular
            # 20-100μm dimensions @ 2.2μm/pixel = 9-45 pixels
            # Note: Rectangle shape provides high rectangularity; varied AR ensures C < 0.5
            length = random.randint(25, 45)
            width_f = random.randint(8, 20)  # Reduced width range to ensure C < 0.5
            angle = random.uniform(0, 180)
            margin = max(length, width_f) // 2 + 50  # Increased for glow effects
            cx = random.randint(margin, width - margin)
            cy = random.randint(margin, height - margin)
            rect = ((cx, cy), (length, width_f), angle)
            box = cv2.boxPoints(rect).astype(np.int32)
            cv2.fillPoly(mask, [box], 255)
            size_hint = max(length, width_f)
            area = length * width_f
        
        elif shape_label == 'Fragment':
            # AR 1.0-3.0, C 0.38-0.72, E 0.20-0.75 - irregular but somewhat rounded
            # CRITICAL: Must ensure circularity stays BELOW 0.72 to avoid Microbead/Pellet
            # Also need to reduce solidity to avoid matching solid >= 0.95 check
            base_size = random.randint(12, 30)
            cx = random.randint(base_size + 50, width - base_size - 50)
            cy = random.randint(base_size + 50, height - base_size - 50)
            
            # Create a more irregular polygon with CONCAVE indentations
            # This reduces both circularity AND solidity
            num_points = random.randint(10, 14)  # More points for more irregular shape
            points = []
            angle_step = 2 * np.pi / num_points
            
            # Create strong irregularity with guaranteed indentations
            for i in range(num_points):
                theta = i * angle_step + random.uniform(-0.35, 0.35)
                
                # Every 2nd or 3rd point is a deep indentation to break circularity
                if i % 3 == 0:
                    # Deep indentation - creates concavity and lowers circularity/solidity
                    r = base_size * random.uniform(0.35, 0.55)
                elif i % 3 == 1:
                    # Large protrusion
                    r = base_size * random.uniform(0.90, 1.15)
                else:
                    # Medium radius
                    r = base_size * random.uniform(0.55, 0.80)
                    
                px = int(cx + r * np.cos(theta))
                py = int(cy + r * np.sin(theta))
                points.append([px, py])
            
            cv2.fillPoly(mask, [np.array(points, dtype=np.int32)], 255)
            size_hint = base_size
            area = cv2.contourArea(np.array(points, dtype=np.int32))
        
        else:  # Irregular
            # C ≤ 0.50, E ≤ 0.70, AR ≤ 3.0 - highly irregular shape
            base_radius = random.randint(10, 32)
            cx = random.randint(base_radius + 50, width - base_radius - 50)
            cy = random.randint(base_radius + 50, height - base_radius - 50)
            
            # Create highly irregular shape with random spikes (reduced noise to prevent E > 0.70)
            angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
            noise = np.random.normal(0, base_radius * 0.30, size=angles.shape)  # Reduced from 0.45 to 0.30
            points = []
            for angle_val, offset in zip(angles, noise):
                r = max(5, base_radius + offset)
                px = int(cx + r * np.cos(angle_val))
                py = int(cy + r * np.sin(angle_val))
                points.append([px, py])
            cv2.fillPoly(mask, [np.array(points, dtype=np.int32)], 255)
            size_hint = base_radius
            area = np.pi * base_radius * base_radius * 0.6
        
        return mask, cx, cy, size_hint, area
    
    def _apply_effects(self, mask: np.ndarray, params: SyntheticImageParams) -> np.ndarray:
        """Apply blur and glow effects to mask"""
        mask_float = mask.astype(np.float32) / 255.0
        
        # Apply blurring if enabled
        if params.enable_blur:
            blur_kernel = params.blur_kernel
            if blur_kernel % 2 == 0:
                blur_kernel += 1
            mask_float = cv2.GaussianBlur(mask_float, (blur_kernel, blur_kernel), 0)
        
        # Apply glowing effect if enabled
        if params.enable_glow:
            glow_sigma = random.uniform(params.glow_sigma_min, params.glow_sigma_max)
            glow1 = cv2.GaussianBlur(mask_float, (0, 0), glow_sigma * 0.5)
            glow2 = cv2.GaussianBlur(mask_float, (0, 0), glow_sigma * 1.5)
            glow_combined = glow1 * 0.6 + glow2 * 0.4
            mask_float = np.clip(
                mask_float * (1 - params.glow_intensity * 0.7) + 
                glow_combined * params.glow_intensity * 1.2,
                0.0, 1.0
            )
        
        return mask_float
    
    def _add_noise_and_artifacts(self, base: np.ndarray, width: int, height: int) -> np.ndarray:
        """Add realistic noise and artifacts"""
        noise_level = random.uniform(0.001, 0.015)
        
        # Add shot noise
        background_noise = np.random.poisson(noise_level * 10, size=base.shape).astype(np.float32) / 255.0
        base += background_noise
        
        # Add random "hot pixels"
        num_hot_pixels = random.randint(5, 25)
        for _ in range(num_hot_pixels):
            hx = random.randint(0, width - 1)
            hy = random.randint(0, height - 1)
            hot_intensity = random.uniform(0.1, 0.4)
            base[hy, hx] = np.clip(base[hy, hx] + hot_intensity, 0.0, 1.0)
        
        return np.clip(base, 0.0, 1.0)
    
    def _apply_optical_blur(self, base: np.ndarray) -> np.ndarray:
        """Apply optical system blur"""
        blur_sigma = random.uniform(0.3, 0.8)
        blurred_base = cv2.GaussianBlur(base, (0, 0), blur_sigma)
        base = base * 0.85 + blurred_base * 0.15
        return np.clip(base, 0.0, 1.0)
